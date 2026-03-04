"""
Main simulation module for chimp evolution model
Core iterative year-by-year population dynamics with mutations
"""

import json
import os
import csv
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
from settings import DEFAULT_SETTINGS, PARAMETER_RANGES
from model import Model

# Global logger callback for GUI integration
_logger_callback = None


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


class PopulationSimulation:
    """Agent-based stochastic model: year-by-year population dynamics"""

    def __init__(self, settings):
        """Initialize simulation with parameters
        
        Args:
            settings (dict): configuration with keys from DEFAULT_SETTINGS
        """
        self.settings = {**DEFAULT_SETTINGS, **settings}
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.settings["device"] == "cuda" else "cpu")
        
        # Validate ranges
        for key, (min_val, max_val) in PARAMETER_RANGES.items():
            if key in self.settings:
                val = self.settings[key]
                if not (min_val <= val <= max_val):
                    log(f"Warning: {key}={val} outside range [{min_val}, {max_val}], clamping")
                    self.settings[key] = max(min_val, min(val, max_val))
        
        # Initialize model with population state
        self.model = Model(self.settings, self.device)
        
        # Simulation state
        self.year = 0
        self.results = []
        self.yearly_beta_changes = []
        self.total_animals_processed = 0
        self.start_time = None
        self.output_dir = Path("result") / self.settings["tag"]
        self.min_survivorship_exponent = None  # Sticky lower bound exponent (10^x)
        self.ema_beta_value = None  # EMA of avg_beta (k=0.03)
        self.consecutive_ema_below_threshold = 0  # Counter for years where ema_change < threshold
        self.stats_collected_count = 0  # Number of times stats have been collected
        # Distribution graph max age (sticky: only expands, never shrinks)
        self.max_age_distribution = None
        # Beta occurrence graph range (sticky: expands but never shrinks)
        beta_init = self.settings["beta_initial"]
        self.beta_range_min = -beta_init / 10.0
        self.beta_range_max = beta_init * 2.0
        
        self._prepare_output_dir()
        self._init_population()

    def _prepare_output_dir(self):
        """Prepare result folder: remove all old files from tag-specific directory"""
        # Remove entire directory if it exists
        if self.output_dir.exists():
            for file_path in self.output_dir.glob("*"):
                file_path.unlink(missing_ok=True)
        # Create fresh directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _init_population(self):
        """Initialize population via model"""
        n = self.settings["initial_population"]
        self.model.initialize_population(
            n,
            self.settings["initial_age_max"],
            self.settings["beta_initial"]
        )
        log(f"Initialized population: {n} animals, device: {self.device}")

    def _calculate_yearly_stats(self):
        """Calculate yearly population statistics
        
        Returns:
            dict with population metrics
        """
        if self.model.get_population_size() == 0:
            return {
                "year": self.year,
                "count": 0,
                "avg_age": 0,
                "born": 0,
                "dead": 0,
                "prop_aging": 0,
                "avg_beta": 0,
                "avg_beta_ema": 0,
            }
        
        ages = self.model.get_ages()
        betas = self.model.get_betas()
        
        # TODO: track births/deaths per year (requires state tracking)
        stats = {
            "year": self.year,
            "count": self.model.get_population_size(),
            "avg_age": float(ages.mean()),
            "born": 0,  # TODO: track from reproduction phase
            "dead": 0,  # TODO: track from mortality phase
            "prop_aging": 0.0,  # TODO: proportion dying of aging
            "avg_beta": float(betas.mean()),
            "avg_beta_ema": 0.0,  # Will be filled in step()
        }
        return stats

    def _log_startup_info(self):
        """Log startup mode and initial simulation data"""
        mode = "CUDA" if self.device.type == "cuda" else "CPU"
        ages = self.model.get_ages()
        betas = self.model.get_betas()

        log(f"Run mode: {mode} (device={self.device})")
        log("Initial settings:")
        for key in sorted(self.settings.keys()):
            log(f"  {key} = {self.settings[key]}")

        log("Initial population data:")
        log(f"  count = {self.model.get_population_size()}")
        log(f"  age: min={ages.min():.1f}, max={ages.max():.1f}, avg={ages.mean():.2f}")
        log(f"  beta: min={betas.min():.4f}, max={betas.max():.4f}, avg={betas.mean():.4f}")

    def _should_stop(self):
        """Check stopping conditions"""
        # Condition 1: population too small
        pop_size = self.model.get_population_size()
        if pop_size < 2:
            log(f"Stop: population too small ({pop_size} animals)")
            return True
        
        # Condition 2: MAX_ITER reached
        max_iter = int(self.settings.get("max_iterations", 100000))
        if self.year >= max_iter:
            log(f"Stop: MAX_ITER ({max_iter}) reached")
            return True
        
        # Condition 3: beta stabilization (EMA-based, 3 consecutive years below threshold)
        if len(self.yearly_beta_changes) > 10:
            avg_change_first_10 = np.mean(self.yearly_beta_changes[:10])
            multiplier = self.settings["stop_beta_change_threshold"]
            threshold = avg_change_first_10 * multiplier
            
            if self.year > 11:
                # Check change in EMA value from previous year
                ema_change = abs(self.results[-1]["avg_beta_ema"] - self.results[-2]["avg_beta_ema"])
                
                if ema_change < threshold:
                    self.consecutive_ema_below_threshold += 1
                    if self.consecutive_ema_below_threshold >= 3:
                        log(f"Stop: beta stabilized (ema_change {ema_change:.6f} < threshold {threshold:.6f} for 3 consecutive years)")
                        return True
                else:
                    # Reset counter if change exceeds threshold
                    self.consecutive_ema_below_threshold = 0
        
        return False

    def _save_distribution_graph(self, year):
        """Save per-year age distribution graph as bar chart

        Args:
            year: current simulation year index
        """
        if self.model.get_population_size() == 0:
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
        fig.savefig(distribution_file, dpi=100, bbox_inches="tight")
        plt.close(fig)
        del fig, ax
        gc.collect()

    def _save_survivorship_graph(self, year):
        """Save per-year smooth survivorship curve (log scale)

        Args:
            year: current simulation year index
        """
        if self.model.get_population_size() == 0:
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
        fig.savefig(survivorship_file, dpi=100, bbox_inches="tight")
        plt.close(fig)
        del fig, ax
        gc.collect()

    def _save_beta_occurrence_graph(self, year):
        """Save per-year beta occurrence histogram

        Args:
            year: current simulation year index
        """
        if self.model.get_population_size() == 0:
            return

        betas = self.model.get_betas()
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

    def _build_animation_gif(self, prefix, output_name):
        """Build animation GIF from yearly PNG files

        Args:
            prefix: png file prefix (distribution or survivorship)
            output_name: output gif file name
        """
        png_files = sorted(
            self.output_dir.glob(f"{prefix}*.png"),
            key=lambda p: int(p.stem.replace(prefix, "")) if p.stem.replace(prefix, "").isdigit() else -1,
        )
        if not png_files:
            log(f"No {prefix}*.png files found for {output_name} - skipping GIF generation")
            return

        log(f"Creating {output_name} from {len(png_files)} PNG files")
        frames = [Image.open(file_path).convert("P") for file_path in png_files]
        gif_file = self.output_dir / output_name
        frames[0].save(
            gif_file,
            save_all=True,
            append_images=frames[1:],
            duration=250,
            loop=0,
        )
        for frame in frames:
            frame.close()
        log(f"Saved animation to {gif_file}")
        gc.collect()

    def step(self):
        """Execute one year iteration: reproduction → aging → mortality"""
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
        
        # Determine if we should collect statistics this year
        stat_period = int(self.settings.get("stat_generation_period", 1))
        if stat_period < 1:
            stat_period = 1
        
        should_collect_stats = (self.year % stat_period == 0)
        
        # Collect statistics only if it's the right period
        if should_collect_stats:
            stats = self._calculate_yearly_stats()
            if births > 0 or deaths > 0:
                stats["born"] = births
                stats["dead"] = deaths
            
            # Calculate EMA of avg_beta with k=0.03
            if len(self.results) == 0:
                self.ema_beta_value = stats["avg_beta"]
                stats["avg_beta_ema"] = stats["avg_beta"]
            else:
                k = 0.03
                self.ema_beta_value = k * stats["avg_beta"] + (1.0 - k) * self.ema_beta_value
                stats["avg_beta_ema"] = self.ema_beta_value
            
            self.results.append(stats)
            
            # Track beta changes for stopping condition
            if len(self.results) > 1:
                change = abs(stats["avg_beta"] - self.results[-2]["avg_beta"])
            else:
                change = 0
            self.yearly_beta_changes.append(change)
            
            # Generate graphs only during stats collection, and only every N stats
            graph_period = int(self.settings.get("graph_generation_period", 1))
            if graph_period < 1:
                graph_period = 1
            
            if self.stats_collected_count % graph_period == 0:
                self._generate_year_graphs(self.year)
            
            self.stats_collected_count += 1
            
            log(f"Year {self.year}: pop={stats['count']}, avg_age={stats['avg_age']:.1f}, avg_beta={stats['avg_beta']:.4f}, avg_beta_ema={stats['avg_beta_ema']:.4f}, births={births}, deaths={deaths}")
        
        self.year += 1
        
        # Check stop conditions (only when we have stats)
        if should_collect_stats and self._should_stop():
            return False
        
        return True

    def run(self):
        """Run full simulation until stop condition"""
        log(f"Starting simulation: {self.settings['tag']}")
        self._log_startup_info()
        self.start_time = time.perf_counter()
        
        while self.step():
            pass

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
        avg_betas = [r["avg_beta"] for r in self.results]
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
        
        # Plot 3: Beta evolution
        ax3.plot(years, avg_betas, linewidth=2, color="red")
        ax3.set_xlabel("Year")
        ax3.set_ylabel("Average Beta")
        ax3.set_title("Genetic Parameter Evolution")
        ax3.grid(True, alpha=0.3)
        
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
        plt.savefig(graph_file, dpi=100, bbox_inches="tight")
        log(f"Saved graph to {graph_file}")
        plt.close()
    
    def export_results(self, output_dir=None):
        """Export results to CSV and generate graphs
        
        Args:
            output_dir: output directory (default: ./result/tag/)
            
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
            keys = self.results[0].keys()
            with open(csv_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.results)
            log(f"Saved results to {csv_file}")
        
        # Generate graphs
        self._generate_graphs(output_dir)
        self._build_animation_gif("distribution", "distribution.gif")
        self._build_animation_gif("survivorship", "survivorship.gif")
        self._build_animation_gif("betaoccurrence", "betaoccurrence.gif")
        
        return str(output_dir)


def run_simulation(config_path="config.json"):
    """Main entry point for simulation from command line or batch
    
    Args:
        config_path: path to config.json (tag as first CLI arg overrides)
        
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
    sim = PopulationSimulation(settings)
    results = sim.run()
    
    # Export results
    sim.export_results()
    
    return results


if __name__ == "__main__":
    import sys
    
    # Optional: tag as first argument
    config_file = "config.json"
    if len(sys.argv) > 1:
        tag = sys.argv[1]
        # Could be used to select config variant TODO: implement tag-based config loading
    
    run_simulation(config_file)
