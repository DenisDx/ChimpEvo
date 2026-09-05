"""
Main simulation module for chimp evolution model
Core iterative year-by-year population dynamics with mutations
"""

import json
import os
import csv
import random
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid thread conflicts with tkinter
import matplotlib.pyplot as plt
from PIL import Image
import gc
from graph_style import clamp_values, render_series
from settings import DEFAULT_SETTINGS, PARAMETER_RANGES
from load_model import load_model_class
from metadata import validate_model_metadata
from experiment_manager import ExperimentNotSelectedError, resolve_experiment_paths, validate_path_component

# Global logger callback for GUI integration
_logger_callback = None

CORE_RUNTIME_SETTINGS = (
    "stat_generation_period",
    "graph_generation_period",
    "min_iterations",
    "max_iterations",
)


def set_logger(callback):
    """Set callback function for logging (used by GUI); callback(message: str)"""
    global _logger_callback
    _logger_callback = callback


def log(*args, **kwargs):
    """Log to console or GUI logger if set"""
    message = " ".join(str(arg) for arg in args)
    if _logger_callback:
        _logger_callback(message)
    else:
        print(message)


def _seed_random_generators(seed):
    """Seed application random generators when a seed is configured."""
    if seed is None or int(seed) == 0:
        return

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_runtime_config(settings, model_class=None):
    """Return model metadata after strict runtime configuration validation."""
    if not isinstance(settings, dict):
        raise TypeError("Configuration must be a JSON object")
    settings.setdefault("min_iterations", DEFAULT_SETTINGS["min_iterations"])
    settings.setdefault("beta_only_positive", False)
    for name in ("model", "tag", "device", *CORE_RUNTIME_SETTINGS):
        if name not in settings:
            raise ValueError(f"Missing required setting: {name}")
    if not isinstance(settings["model"], str) or not settings["model"]:
        raise ValueError("model must be a non-empty string")
    if not isinstance(settings["tag"], str) or not settings["tag"]:
        raise ValueError("tag must be a non-empty string")
    validate_path_component(settings["tag"], "tag")
    if settings["tag"] == "_models":
        raise ValueError("tag is reserved for batch model artifacts: _models")
    if settings["device"] not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")

    if model_class is None:
        model_class = load_model_class(settings["model"])
    model_metadata = validate_model_metadata(model_class)
    declarations = {
        **{
            name: {
                "type": "int" if isinstance(DEFAULT_SETTINGS[name], int) else "float",
                "min": PARAMETER_RANGES[name][0],
                "max": PARAMETER_RANGES[name][1],
            }
            for name in CORE_RUNTIME_SETTINGS
        },
        **model_metadata["settings"],
    }
    for name, declaration in declarations.items():
        if name not in settings:
            raise ValueError(f"Missing required setting for {settings['model']}: {name}")
        value = settings[name]
        expected_type = declaration["type"]
        valid_type = {
            "int": isinstance(value, int) and not isinstance(value, bool),
            "float": isinstance(value, (int, float)) and not isinstance(value, bool),
            "str": isinstance(value, str),
            "bool": isinstance(value, bool),
        }[expected_type]
        if not valid_type:
            raise ValueError(f"Invalid type for {name}: expected {expected_type}")
        if "min" in declaration and value < declaration["min"]:
            raise ValueError(f"{name} must be >= {declaration['min']}")
        if "max" in declaration and value > declaration["max"]:
            raise ValueError(f"{name} must be <= {declaration['max']}")
    if settings.get("initial_population", 0) > settings.get("max_population", float("inf")):
        raise ValueError("initial_population must not exceed max_population")
    if settings["min_iterations"] > settings["max_iterations"]:
        raise ValueError("min_iterations must not exceed max_iterations")
    return model_metadata


class PopulationSimulation:
    """Agent-based stochastic model: year-by-year population dynamics"""

    def __init__(self, settings, result_root="result"):
        """Initialize simulation with parameters
        
        Args:
            settings (dict): configuration with keys from DEFAULT_SETTINGS
            result_root: directory containing tag-specific result folders
        """
        self.settings = dict(settings)
        model_name = self.settings.get("model", "")
        model_class = load_model_class(model_name)
        self.model_metadata = validate_runtime_config(self.settings, model_class)
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.settings["device"] == "cuda" else "cpu")
        _seed_random_generators(self.settings.get("seed"))
        
        self._validate_settings()

        # Initialize configured dynamic model
        self.model = model_class(self.settings, self.device)
        self.has_age_field = "age" in self.model.population_fields
        self.has_beta_field = "beta" in self.model.population_fields
        
        # Simulation state
        self.year = 0
        self.results = []
        self.total_animals_processed = 0
        self.start_time = None
        self.output_dir = Path(result_root) / self.settings["tag"]
        self.min_survivorship_exponent = None  # Sticky lower bound exponent (10^x)
        self.stats_collected_count = 0  # Number of times stats have been collected
        # Distribution graph max age (sticky: only expands, never shrinks)
        self.max_age_distribution = None
        # Beta occurrence graph range (sticky: expands but never shrinks)
        self.beta_range_min = None
        self.beta_range_max = None
        self.last_generated_graph_year = None
        if self.has_beta_field and "beta_initial" in self.settings:
            beta_init = self.settings["beta_initial"]
            self.beta_range_min = -beta_init / 10.0
            self.beta_range_max = beta_init * 2.0
        
        self._prepare_output_dir()
        self._init_population()

    def _validate_settings(self):
        """Clamp configured numeric values to their supported ranges."""
        for key, (min_val, max_val) in PARAMETER_RANGES.items():
            if key not in self.settings:
                continue
            value = self.settings[key]
            if not (min_val <= value <= max_val):
                log(f"Warning: {key}={value} outside range [{min_val}, {max_val}], clamping")
                self.settings[key] = max(min_val, min(value, max_val))

    def _prepare_output_dir(self):
        """Create a new tag result directory without replacing prior data."""
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"Result directory is not empty: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _init_population(self):
        """Initialize population via model"""
        n = self.settings["initial_population"]
        self.model.initialize_population()
        log(f"Initialized population: {n} animals, device: {self.device}")

    def _calculate_yearly_stats(self):
        """Return the year and model values declared for annual output."""
        values = self._get_model_values()
        stats = {"year": self.year}
        for name, metadata in self.model_metadata["values"].items():
            if metadata["annual"]:
                stats[name] = values[name]
        return stats

    def _get_model_values(self):
        """Return validated public scalar values from the active model."""
        values = self.model.get_values()
        if not isinstance(values, dict):
            raise TypeError("get_values() must return a dict")

        declared_names = set(self.model_metadata["values"])
        returned_names = set(values)
        missing_names = declared_names - returned_names
        extra_names = returned_names - declared_names
        if missing_names or extra_names:
            raise ValueError(
                "get_values() names do not match declarations: "
                f"missing={sorted(missing_names)}, extra={sorted(extra_names)}"
            )

        normalized = {}
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                raise TypeError(f"get_values() must not return Torch tensors: {name}")
            if isinstance(value, np.generic):
                value = value.item()
            if value is not None and not isinstance(value, (str, bool, int, float)):
                raise TypeError(f"get_values() returned a non-scalar value: {name}")
            normalized[name] = value
        return normalized

    def _log_startup_info(self):
        """Log startup mode and initial simulation data"""
        mode = "CUDA" if self.device.type == "cuda" else "CPU"

        log(f"Run mode: {mode} (device={self.device})")
        log("Initial settings:")
        for key in sorted(self.settings.keys()):
            log(f"  {key} = {self.settings[key]}")

        log("Initial population data:")
        log(f"  count = {self.model.get_population_size()}")
        if self.has_age_field and self.model.get_population_size() > 0:
            ages = self.model.get_ages()
            log(f"  age: min={ages.min():.1f}, max={ages.max():.1f}, avg={ages.mean():.2f}")
        if self.has_beta_field and self.model.get_population_size() > 0:
            betas = self.model.get_tensor("beta")
            log(f"  beta: min={betas.min():.4f}, max={betas.max():.4f}, avg={betas.mean():.4f}")

    def _should_stop(self, model_stop_reason):
        """Check core iteration limit and a model-provided stop reason."""
        if model_stop_reason:
            log(f"Stop: {model_stop_reason}")
            return True

        max_iter = int(self.settings.get("max_iterations", 100000))
        if self.year >= max_iter:
            log(f"Stop: MAX_ITER ({max_iter}) reached")
            return True
        return False

    def _annotate_tag(self, fig):
        """Stamp the current run tag in a saved figure's top-right corner."""
        tag = self.settings.get("tag")
        if tag:
            fig.text(0.99, 0.99, tag, ha="right", va="top", fontsize=9, color="gray", alpha=0.8)

    def _save_distribution_graph(self, year):
        """Save per-year age distribution graph as bar chart

        Args:
            year: current simulation year index
        """
        if not self.has_age_field or self.model.get_population_size() == 0:
            return
        ages = self.model.get_ages().astype(int)
        ages = ages[ages > 0]
        if len(ages) == 0:
            return

        max_age = int(ages.max())
        # Sticky max_age: only expands, never shrinks (for consistent graph sizing)
        if self.max_age_distribution is None:
            self.max_age_distribution = max_age
        else:
            self.max_age_distribution = max(self.max_age_distribution, max_age)
        
        age_axis = np.arange(1, self.max_age_distribution + 1)
        age_counts = np.bincount(ages, minlength=self.max_age_distribution + 1)[1:]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(age_axis, age_counts, width=0.8, alpha=0.9)
        ax.set_title(f"Age Distribution (Year {year})")
        ax.set_xlabel("Age")
        ax.set_ylabel("Animal Count")
        ax.grid(True, alpha=0.3)

        distribution_file = self.output_dir / f"distribution{year}.png"
        fig.tight_layout()
        self._annotate_tag(fig)
        fig.savefig(distribution_file, dpi=100, bbox_inches="tight")
        plt.close(fig)
        del fig, ax
        gc.collect()

    def _save_survivorship_graph(self, year):
        """Save per-year smooth survivorship curve (log scale)

        Args:
            year: current simulation year index
        """
        if (
            not self.has_age_field
            or "lambda" not in self.settings
            or self.model.get_population_size() == 0
        ):
            return

        ages = self.model.get_ages().astype(int)
        ages = ages[ages > 0]
        if len(ages) == 0:
            return

        max_age = int(ages.max())
        total = len(ages)
        age_axis = np.arange(1, max_age + 1)

        survivorship = np.zeros(len(age_axis), dtype=float)
        for idx, age in enumerate(age_axis):
            survivorship[idx] = (np.sum(ages >= age) / total) * 100.0

        # Prepend point at age 0.5 with 100% survival (ensures curve starts at 100%)
        age_axis = np.concatenate([[0.5], age_axis])
        survivorship = np.concatenate([[100.0], survivorship])

        dense_x = np.linspace(0.5, max_age, num=max(120, max_age * 8))
        dense_y = np.interp(dense_x, age_axis, survivorship)
        if len(dense_y) >= 9:
            kernel = np.ones(9, dtype=float) / 9.0
            pad = len(kernel) // 2
            padded = np.pad(dense_y, (pad, pad), mode="edge")
            dense_y = np.convolve(padded, kernel, mode="valid")
        dense_y[dense_x <= 1.0] = 100.0
        dense_y = np.clip(dense_y, 0.01, 100.0)

        # Adaptive lower bound: start with ceiling power of 10, expand if >3 elements below
        min_val = np.min(dense_y)
        current_exponent = int(np.ceil(np.log10(min_val)))
        below_count = np.sum(dense_y < (10.0 ** current_exponent))
        if below_count > 3:
            current_exponent -= 1  # expand to 10^(exp-1)
        
        # Sticky bound: once set, it only expands further, never shrinks
        if self.min_survivorship_exponent is None:
            self.min_survivorship_exponent = current_exponent
        else:
            self.min_survivorship_exponent = min(self.min_survivorship_exponent, current_exponent)
        
        lower_bound = 10.0 ** self.min_survivorship_exponent

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(dense_x, dense_y, linewidth=2.0)
        
        # Add reference line: pure lambda-based exponential decay (no Gompertz effect)
        lambda_val = self.settings["lambda"]
        lambda_reference = 100.0 * np.exp(-lambda_val * dense_x)
        lambda_reference = np.clip(lambda_reference, lower_bound, 100.0)
        ax.plot(dense_x, lambda_reference, color='red', linestyle='--', linewidth=1.5, 
                label=f'Lambda-only (λ={lambda_val:.3f})')
        
        ax.set_yscale("log")
        ax.set_ylim(lower_bound, 100)
        ax.set_title(f"Survivorship Curve (Year {year})")
        ax.set_xlabel("Age")
        ax.set_ylabel("Survival (%)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        survivorship_file = self.output_dir / f"survivorship{year}.png"
        fig.tight_layout()
        self._annotate_tag(fig)
        fig.savefig(survivorship_file, dpi=100, bbox_inches="tight")
        plt.close(fig)
        del fig, ax
        gc.collect()

    def _save_beta_occurrence_graph(self, year):
        """Save per-year beta occurrence histogram

        Args:
            year: current simulation year index
        """
        if (
            not self.has_beta_field
            or self.beta_range_min is None
            or self.beta_range_max is None
            or self.model.get_population_size() == 0
        ):
            return

        betas = self.model.get_tensor("beta")
        if len(betas) == 0:
            return

        # Update range (sticky: expands but never shrinks)
        current_min = float(betas.min())
        current_max = float(betas.max())
        if current_min < self.beta_range_min:
            self.beta_range_min = current_min
        if current_max > self.beta_range_max:
            self.beta_range_max = current_max

        # Create histogram data with 50 bins
        fig, ax = plt.subplots(figsize=(9, 5))
        n_bins = 50
        counts, bin_edges = np.histogram(betas, bins=n_bins, 
                                         range=(self.beta_range_min, self.beta_range_max))
        
        # Calculate bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Filter out empty bins (where count == 0)
        non_zero_mask = counts > 0
        bin_centers_filtered = bin_centers[non_zero_mask]
        counts_filtered = counts[non_zero_mask]
        
        # Plot as circles (scatter points) - only non-zero bins
        ax.scatter(bin_centers_filtered, counts_filtered, s=80, alpha=0.7, edgecolors='black', linewidths=0.5)
        
        ax.set_title(f"Beta Distribution (Year {year})")
        ax.set_xlabel("Beta Value")
        ax.set_ylabel("Count")
        ax.set_xlim(self.beta_range_min, self.beta_range_max)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(bottom=0)
        
        # Mark initial beta_initial with vertical line
        beta_init = self.settings["beta_initial"]
        ax.axvline(beta_init, color='red', linestyle='--', linewidth=1.5, label=f'Initial β={beta_init:.2f}')
        ax.legend()

        beta_file = self.output_dir / f"betaoccurrence{year}.png"
        fig.tight_layout()
        self._annotate_tag(fig)
        fig.savefig(beta_file, dpi=100, bbox_inches="tight")
        plt.close(fig)
        del fig, ax
        gc.collect()

    def _generate_year_graphs(self, year):
        """Generate yearly distribution, survivorship, and beta occurrence files

        Args:
            year: current simulation year index
        """
        self._save_distribution_graph(year)
        self._save_survivorship_graph(year)
        self._save_beta_occurrence_graph(year)
        self._generate_model_graphs(annual=True, year=year)

    def _generate_model_graphs(self, annual=False, final=False, year=None, output_dir=None):
        """Render model-declared graphs for one requested output stage."""
        target_dir = Path(output_dir) if output_dir is not None else self.output_dir
        for graph in self.model_metadata["graphs"]:
            annual_output = annual and graph["annual"] and not graph["last"]
            final_output = final and (graph["final"] or graph["last"])
            if annual_output or final_output:
                suffix = f"_{int(year):07d}" if annual else ""
                output_path = target_dir / f"{graph['filename']}{suffix}.png"
                if graph["type"] == "time":
                    self._save_model_time_graph(graph, output_path)
                else:
                    self._save_model_distribution_graph(graph, output_path)

    def _save_model_time_graph(self, graph, output_path):
        """Render declared scalar histories to one time graph (lines/points/bars)."""
        fig, ax = plt.subplots(figsize=(10, 6))
        style = graph["style"]
        values2 = graph.get("values2")
        values3 = graph.get("values3")
        plot_filter = graph["filter"]
        plot_range = graph["range"]

        def row_passes_filter(row):
            """Return whether one stored statistic row passes all graph filters."""
            for field_name, bounds in plot_filter.items():
                value = row.get(field_name)
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    return False
                if not bounds[0] <= value <= bounds[1]:
                    return False
            return True

        filtered_results = [row for row in self.results if row_passes_filter(row)]
        for index, (value_name, label) in enumerate(zip(graph["values"], graph["labels"])):
            size_name = values2[index] if values2 else None
            color_name = values3[index] if values3 else None
            points = [
                (
                    row["year"],
                    row[value_name],
                    row.get(size_name) if size_name else None,
                    row.get(color_name) if color_name else None,
                )
                for row in filtered_results
                if row.get(value_name) is not None
                and (size_name is None or row.get(size_name) is not None)
                and (color_name is None or row.get(color_name) is not None)
            ]
            if not points:
                continue
            render_series(
                ax,
                clamp_values([point[0] for point in points], plot_range.get("year")),
                clamp_values([point[1] for point in points], plot_range.get(value_name)),
                label,
                style,
                size_values=[point[2] for point in points] if size_name else None,
                color_values=[point[3] for point in points] if color_name else None,
                max_point_size=graph["max_point_size"],
                size_range=plot_range.get(size_name) if size_name else None,
                color_range=plot_range.get(color_name) if color_name else None,
            )
        ax.set_xlabel(graph["xlabel"] or "Year")
        ax.set_title(graph["title"])
        if len(graph["values"]) > 1:
            ax.legend()
        ax.grid(True, alpha=0.3)
        x_range = graph["range"].get("year")
        y_range = graph["range"].get(graph["values"][0])
        if x_range:
            ax.set_xlim(*x_range)
        if y_range:
            ax.set_ylim(*y_range)
        fig.tight_layout()
        self._annotate_tag(fig)
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _save_model_distribution_graph(self, graph, output_path):
        """Render independently binned population fields to one distribution graph."""
        fig, ax = plt.subplots(figsize=(10, 6))
        for field_name, label in zip(graph["values"], graph["labels"]):
            bins = self.model.bin_values(
                field_name,
                graph["bin_count"],
                min=graph.get("min"),
                max=graph.get("max"),
                padding_min=graph["padding_min"],
                padding_max=graph["padding_max"],
                scale=graph["scale"],
            )
            if bins["min"] is None or bins["max"] is None:
                continue
            positions = np.linspace(0.0, 1.0, graph["bin_count"] + 1) ** graph["scale"]
            edges = bins["min"] + (bins["max"] - bins["min"]) * positions
            ax.stairs(bins["data"], edges, label=label, linewidth=2)
        ax.set_xlabel(graph["xlabel"])
        ax.set_ylabel("Count")
        ax.set_title(graph["title"])
        if len(graph["values"]) > 1:
            ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._annotate_tag(fig)
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _build_animation_gif(self, prefix, output_name, output_dir=None):
        """Build animation GIF from yearly PNG files

        Args:
            prefix: png file prefix (distribution or survivorship)
            output_name: output gif file name
            output_dir: directory containing frames and receiving the GIF
        """
        target_dir = Path(output_dir) if output_dir is not None else self.output_dir
        png_files = sorted(
            target_dir.glob(f"{prefix}*.png"),
            key=lambda p: int(p.stem.replace(prefix, "")) if p.stem.replace(prefix, "").isdigit() else -1,
        )
        if not png_files:
            log(f"No {prefix}*.png files found for {output_name} - skipping GIF generation")
            return

        log(f"Creating {output_name} from {len(png_files)} PNG files")
        frames = []
        for file_path in png_files:
            with Image.open(file_path) as source_image:
                frames.append(source_image.convert("P"))
        gif_file = target_dir / output_name
        try:
            frames[0].save(
                gif_file,
                save_all=True,
                append_images=frames[1:],
                duration=250,
                loop=0,
            )
        finally:
            for frame in frames:
                frame.close()
        log(f"Saved animation to {gif_file}")
        gc.collect()

    def step(self, should_finalize=None):
        """Execute one year and honor an optional successful-finalization request."""
        if self.model.get_population_size() == 0:
            return False

        # Track processed animals for speed metric
        self.total_animals_processed += self.model.get_population_size()
        
        # Step 1: Reproduction (fill empty niches)
        births = self.model.apply_reproduction()
        
        # Step 2: Aging (increment all ages by 1)
        self.model.age_population()
        
        # Step 3: Mortality (stochastic death)
        deaths = self.model.apply_mortality()
        model_stop_reason = self.model.should_stop()
        completed_iterations = self.year + 1
        min_iterations = int(self.settings.get("min_iterations", 0))
        effective_stop_reason = (
            model_stop_reason if completed_iterations >= min_iterations else None
        )
        if should_finalize is not None and should_finalize():
            effective_stop_reason = effective_stop_reason or "finalization requested"
        
        # Determine if we should collect statistics this year
        stat_period = int(self.settings.get("stat_generation_period", 1))
        if stat_period < 1:
            stat_period = 1
        
        max_iterations = int(self.settings.get("max_iterations", 100000))
        reaches_max_iterations = self.year + 1 >= max_iterations
        should_collect_stats = (
            self.year % stat_period == 0
            or bool(effective_stop_reason)
            or reaches_max_iterations
        )
        
        # Collect statistics only if it's the right period
        if should_collect_stats:
            stats = self._calculate_yearly_stats()
            if births > 0 or deaths > 0:
                stats["born"] = births
                stats["dead"] = deaths
            
            self.results.append(stats)
            
            # Generate graphs only during stats collection, and only every N stats
            graph_period = int(self.settings.get("graph_generation_period", 1))
            if graph_period < 1:
                graph_period = 1
            
            if self.stats_collected_count % graph_period == 0:
                self._generate_year_graphs(self.year)
                self.last_generated_graph_year = self.year
            
            self.stats_collected_count += 1

            value_text = ", ".join(
                f"{name}={value}"
                for name, value in stats.items()
                if name != "year"
            )
            log(f"Year {self.year}: {value_text}")
        
        self.year += 1

        if self._should_stop(effective_stop_reason):
            return False
        
        return True

    def run(
        self,
        should_cancel=None,
        should_finalize=None,
        graph_callback=None,
        performance_callback=None,
    ):
        """Run until completion and report generated graphs and performance snapshots."""
        log(f"Starting simulation: {self.settings['tag']}")
        self._log_startup_info()
        self.start_time = time.perf_counter()
        self.was_cancelled = False
        reported_graph_year = None
        next_performance_report = self.start_time

        while True:
            if should_cancel is not None and should_cancel():
                self.was_cancelled = True
                log("Stop: cancellation requested")
                break
            has_next = self.step(should_finalize=should_finalize)
            current_time = time.perf_counter()
            if performance_callback is not None and current_time >= next_performance_report:
                performance_callback(
                    current_time - self.start_time,
                    self.year,
                    self.total_animals_processed,
                )
                next_performance_report = current_time + 0.1
            if (
                graph_callback is not None
                and self.last_generated_graph_year != reported_graph_year
            ):
                graph_callback(self.output_dir, self.last_generated_graph_year)
                reported_graph_year = self.last_generated_graph_year
            if not has_next:
                break

        if performance_callback is not None:
            performance_callback(
                time.perf_counter() - self.start_time,
                self.year,
                self.total_animals_processed,
            )

        # Generate graph for final year if not already generated
        if self.results:
            graph_period = int(self.settings.get("graph_generation_period", 1))
            # Check if last statistics collection generated a graph
            last_graph_collected = (self.stats_collected_count - 1) % graph_period == 0 if self.stats_collected_count > 0 else False
            if not last_graph_collected:
                # Generate final year graph (use year from last stats collection)
                final_year = self.results[-1]["year"]
                log(f"Generating final year graph for year {final_year}")
                self._generate_year_graphs(final_year)
                self.last_generated_graph_year = final_year
        if (
            graph_callback is not None
            and self.last_generated_graph_year != reported_graph_year
        ):
            graph_callback(self.output_dir, self.last_generated_graph_year)

        total_time_sec = time.perf_counter() - self.start_time
        avg_iteration_time_sec = total_time_sec / self.year if self.year > 0 else 0.0
        avg_per_animal_time_sec = (
            total_time_sec / self.total_animals_processed
            if self.total_animals_processed > 0
            else 0.0
        )

        log(f"Simulation complete: {self.year} years, final population: {self.model.get_population_size()}")
        log("Speed statistics:")
        log(f"  total calculation time = {total_time_sec:.3f} s")
        log(f"  average iteration time = {avg_iteration_time_sec:.6f} s")
        log(f"  average per-animal time = {avg_per_animal_time_sec:.9f} s")
        return self.results

    def _generate_graphs(self, output_dir):
        """Generate summary graphs from simulation results
        
        Args:
            output_dir: output directory path (Path object)
        """
        if not self.results:
            return
        
        years = [r["year"] for r in self.results]
        counts = [r["count"] for r in self.results]
        avg_ages = [r["avg_age"] for r in self.results]
        beta_rows_available = all("avg_beta" in row for row in self.results)
        avg_betas = [r["avg_beta"] for r in self.results] if beta_rows_available else []
        births = [r.get("born", 0) for r in self.results]
        deaths = [r.get("dead", 0) for r in self.results]
        
        # Create figure with 4 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Population dynamics
        ax1.plot(years, counts, linewidth=2, color="blue")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Population Count")
        ax1.set_title("Population Dynamics")
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Average age
        ax2.plot(years, avg_ages, linewidth=2, color="green")
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Average Age")
        ax2.set_title("Average Age Over Time")
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Beta evolution when the active model declares the scalar.
        if beta_rows_available:
            ax3.plot(years, avg_betas, linewidth=2, color="red")
            ax3.set_xlabel("Year")
            ax3.set_ylabel("Average Beta")
            ax3.set_title("Genetic Parameter Evolution")
            ax3.grid(True, alpha=0.3)
        else:
            ax3.set_visible(False)
        
        # Plot 4: Birth/death rates
        ax4.bar([y - 0.2 for y in years], births, width=0.4, label="Births", color="green", alpha=0.7)
        ax4.bar([y + 0.2 for y in years], deaths, width=0.4, label="Deaths", color="red", alpha=0.7)
        ax4.set_xlabel("Year")
        ax4.set_ylabel("Count")
        ax4.set_title("Birth and Death Events")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        graph_file = output_dir / "results_summary.png"
        self._annotate_tag(fig)
        plt.savefig(graph_file, dpi=100, bbox_inches="tight")
        log(f"Saved graph to {graph_file}")
        plt.close()
    
    def export_results(self, output_dir=None, successful=False):
        """Export results to CSV and generate graphs
        
        Args:
            output_dir: output directory (default: ./result/tag/)
            successful: create final.csv only for a completed calculation
            
        Returns:
            output directory path (str)
        """
        if output_dir is None:
            output_dir = self.output_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        log(f"Exporting results to {output_dir}, total years: {self.year}")
        
        # Save CSV
        csv_file = output_dir / "result.csv"
        if self.results:
            keys = ["year"] + [
                name for name, metadata in self.model_metadata["values"].items()
                if metadata["annual"]
            ]
            with open(csv_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.results)
            log(f"Saved results to {csv_file}")

        final_file = output_dir / "final.csv"
        if successful:
            values = self._get_model_values()
            final_names = [
                name for name, metadata in self.model_metadata["values"].items()
                if metadata["final"]
            ]
            duration_seconds = 0.0
            if self.start_time is not None:
                duration_seconds = time.perf_counter() - self.start_time
            final_row = {
                "model": self.settings.get("model", "model_base"),
                "tag": self.settings["tag"],
                "year": max(self.year - 1, 0),
                "duration_seconds": duration_seconds,
            }
            final_row.update({name: values[name] for name in final_names})
            final_keys = ["model", "tag", "year", "duration_seconds", *final_names]
            with open(final_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=final_keys)
                writer.writeheader()
                writer.writerow(final_row)
            log(f"Saved final results to {final_file}")
        else:
            final_file.unlink(missing_ok=True)
        
        # Generate graphs
        self._generate_graphs(output_dir)
        self._build_animation_gif("distribution", "distribution.gif")
        self._build_animation_gif("survivorship", "survivorship.gif")
        if self.has_beta_field:
            self._build_animation_gif("betaoccurrence", "betaoccurrence.gif")
        if successful:
            self._generate_model_graphs(final=True, output_dir=output_dir)
            for graph in self.model_metadata["graphs"]:
                if graph["annual"] and graph["animated"] and not graph["last"]:
                    self._build_animation_gif(
                        f"{graph['filename']}_",
                        f"{graph['filename']}.gif",
                        output_dir=output_dir,
                    )
        
        return str(output_dir)


def run_simulation(
    config_path="config.json",
    should_cancel=None,
    should_finalize=None,
    return_completion=False,
    graph_callback=None,
    performance_callback=None,
    result_root="result",
):
    """Main entry point for simulation from command line or batch
    
    Args:
        config_path: configuration path or in-memory settings
        should_cancel: optional callback checked between simulation years
        should_finalize: optional callback requesting successful completion
        return_completion: include successful-completion status in the return value
        graph_callback: optional callback receiving output directory and generated year
        performance_callback: optional callback receiving elapsed seconds, years, and processed animals
        result_root: directory containing tag-specific result folders
        
    Returns:
        results (list of dicts)
    """
    # Load configuration
    if isinstance(config_path, str):
        with open(config_path) as f:
            settings = json.load(f)
    else:
        settings = config_path  # already a dict

    # Run simulation
    sim = PopulationSimulation(settings, result_root=result_root)
    results = sim.run(
        should_cancel=should_cancel,
        should_finalize=should_finalize,
        graph_callback=graph_callback,
        performance_callback=performance_callback,
    )
    
    # Export results
    completed = not sim.was_cancelled
    sim.export_results(successful=completed)
    
    if return_completion:
        return results, completed
    return results


if __name__ == "__main__":
    import sys

    try:
        paths = resolve_experiment_paths(Path.cwd())
    except ExperimentNotSelectedError as error:
        raise SystemExit(str(error)) from error

    config_file = paths["config_path"]
    config_source = str(config_file)
    if len(sys.argv) > 1:
        with config_file.open(encoding="utf-8") as source_file:
            config_source = json.load(source_file)
        config_source["tag"] = validate_path_component(sys.argv[1], "tag")

    run_simulation(config_source, result_root=paths["result_dir"])
