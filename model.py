"""Generic population-model lifecycle and tensor access."""

import builtins
from types import MappingProxyType

import torch


class Model:
    """Provide shared age-based behavior for dynamic population models."""

    @staticmethod
    def description():
        """Return the model description as lightweight Markdown."""
        return """# Custom population model

## Purpose

Replace this text with a description of your model by overriding the static
`description()` method.

## Inheritance

This placeholder is defined by the generic `Model` lifecycle.

## Difference from its parent

Describe the biological rules, settings, outputs, and behavior that your model
adds or changes.
"""

    def __init__(self, settings, device):
        """Store settings, device, population schema, and annual counters."""
        self.settings = settings
        self.device = device
        field_declarations = self.add_population_fields()
        if not isinstance(field_declarations, dict):
            raise TypeError("add_population_fields() must return a dict")

        field_metadata = {}
        for field_name, metadata in field_declarations.items():
            if not isinstance(field_name, str) or not field_name:
                raise ValueError("Population field names must be non-empty strings")
            if not isinstance(metadata, dict):
                raise TypeError(f"Population field metadata must be a dict: {field_name}")
            field_metadata[field_name] = MappingProxyType(dict(metadata))

        self.population_field_metadata = MappingProxyType(field_metadata)
        self.population_fields = MappingProxyType({
            field_name: column
            for column, field_name in enumerate(field_declarations)
        })
        self.population = None
        self.last_born = 0
        self.last_death = 0

    @staticmethod
    def add_population_fields():
        """Declare ordered population fields and their public metadata."""
        return {"age": {"public": True}}

    @staticmethod
    def add_settings():
        """Declare generic model settings."""
        return {
            "seed": {
                "description": "Random seed; zero selects a random seed",
                "default": 0,
                "type": "int",
            },
        }

    @staticmethod
    def add_values():
        """Declare core scalar values returned by every population model."""
        return {
            "count": {"title": "Population", "annual": True, "final": True, "format": "d"},
            "avg_age": {"title": "Average age", "annual": True, "final": True, "format": ".4f"},
            "born": {"title": "Born", "annual": True, "final": False, "format": "d"},
            "dead": {"title": "Dead", "annual": True, "final": False, "format": "d"},
            "prop_aging": {"title": "Proportion aging", "annual": True, "final": False, "format": ".4f"},
        }

    @staticmethod
    def add_graphs():
        """Declare the shared age distribution graph."""
        return [{
            "filename": "age_distribution",
            "title": "Age Distribution",
            "values": ["age"],
            "labels": ["Age"],
            "type": "distr",
            "annual": True,
            "final": True,
            "animated": True,
        }]

    @staticmethod
    def add_metagraphs():
        """Declare no aggregate batch graphs for the generic model."""
        return []

    @staticmethod
    def add_batch():
        """Declare no default batch CSV for the generic model."""
        return ""

    def _validate_population_rows(self, rows):
        """Validate tensor shape, width, dtype, and device against the schema."""
        if not isinstance(rows, torch.Tensor):
            raise TypeError("Population rows must be a torch.Tensor")
        if rows.ndim != 2:
            raise ValueError("Population rows must be a 2D tensor")

        expected_width = len(self.population_fields)
        if rows.shape[1] != expected_width:
            raise ValueError(f"Population rows must contain {expected_width} columns")
        if not rows.is_floating_point():
            raise TypeError("Population rows must use a floating-point dtype")
        expected_device = torch.device(self.device)
        wrong_device_type = rows.device.type != expected_device.type
        wrong_explicit_index = (
            expected_device.index is not None
            and rows.device.index != expected_device.index
        )
        if wrong_device_type or wrong_explicit_index:
            raise ValueError(f"Population rows must use device {self.device}")

    def _set_population(self, rows):
        """Validate and set the complete population tensor."""
        self._validate_population_rows(rows)
        self.population = rows

    def _append_population_rows(self, rows):
        """Validate and append complete animal rows to the population."""
        self._validate_population_rows(rows)
        if self.population is None:
            self.population = rows
            return
        self.population = torch.cat([self.population, rows], dim=0)

    def initialize_population(self):
        """Initialize an age-only population from model settings."""
        initial_population = int(self.settings["initial_population"])
        initial_age_max = int(self.settings["initial_age_max"])
        ages = torch.randint(
            0,
            initial_age_max + 1,
            (initial_population, 1),
            dtype=torch.float32,
            device=self.device,
        )
        self._set_population(ages)

    def apply_mortality(self):
        """Keep all animals and return zero deaths."""
        self.last_death = 0
        return self.last_death

    def apply_reproduction(self):
        """Create no offspring and return zero births."""
        self.last_born = 0
        return self.last_born

    def age_population(self):
        """Increase the registered age field by one year."""
        self.population[:, self.population_fields["age"]] += 1

    def get_tensor(self, field_name):
        """Return a population field as a detached CPU NumPy array."""
        try:
            column = self.population_fields[field_name]
        except KeyError as error:
            raise KeyError(f"Unknown population field: {field_name}") from error
        return self.population[:, column].detach().cpu().numpy()

    def get_ages(self):
        """Return ages through the named population-field API."""
        return self.get_tensor("age")

    def bin_values(
        self,
        field_name,
        bin_count,
        min=None,
        max=None,
        padding_min=0.0,
        padding_max=0.0,
        scale=1.0,
    ):
        """Aggregate one public population field into device-side intervals."""
        if field_name not in self.population_fields:
            raise KeyError(f"Unknown population field: {field_name}")
        if not self.population_field_metadata[field_name].get("public", False):
            raise KeyError(f"Population field is not public: {field_name}")
        if not isinstance(bin_count, int) or bin_count < 1:
            raise ValueError("bin_count must be a positive integer")
        if scale <= 0:
            raise ValueError("scale must be greater than zero")
        if not (0.0 <= padding_min < 1.0 and 0.0 <= padding_max < 1.0):
            raise ValueError("padding values must be in [0, 1)")
        if padding_min + padding_max >= 1.0:
            raise ValueError("padding_min + padding_max must be less than one")

        column = self.population_fields[field_name]
        values = self.population[:, column]
        non_nan_values = values[~torch.isnan(values)]
        finite_values = non_nan_values[torch.isfinite(non_nan_values)]
        if finite_values.numel() == 0:
            return self._empty_bin_result(bin_count, min, max, non_nan_values)

        finite_min = finite_values.min()
        finite_max = finite_values.max()
        lower_bound = float(min) if min is not None else finite_min.item()
        upper_bound = float(max) if max is not None else finite_max.item()

        sorted_values = torch.sort(non_nan_values).values
        total_count = sorted_values.numel()
        lower_padding_count = int(total_count * padding_min)
        if lower_padding_count > 0:
            lower_index = builtins.min(lower_padding_count, total_count - 1)
            padded_lower = torch.clamp(sorted_values[lower_index], finite_min, finite_max).item()
            lower_bound = builtins.max(lower_bound, padded_lower)
        upper_padding_count = int(total_count * padding_max)
        if upper_padding_count > 0:
            upper_index = builtins.max(total_count - upper_padding_count, 0)
            upper_index = builtins.min(upper_index, total_count - 1)
            padded_upper = torch.clamp(sorted_values[upper_index], finite_min, finite_max).item()
            upper_bound = builtins.min(upper_bound, padded_upper)

        if lower_bound == upper_bound and min is None and max is None:
            lower_bound -= 0.5
            upper_bound += 0.5
        if lower_bound >= upper_bound:
            raise ValueError("Resolved bin range must have min less than max")

        positions = torch.linspace(
            0.0,
            1.0,
            bin_count + 1,
            dtype=values.dtype,
            device=values.device,
        ).pow(float(scale))
        edges = lower_bound + (upper_bound - lower_bound) * positions
        in_range = non_nan_values[
            (non_nan_values >= lower_bound) & (non_nan_values < upper_bound)
        ]
        interval_indices = torch.bucketize(in_range, edges[1:-1], right=True)
        counts = torch.bincount(interval_indices, minlength=bin_count)
        below_min = (non_nan_values < lower_bound).sum().item()
        above_max = (non_nan_values >= upper_bound).sum().item()
        return {
            "data": [int(count) for count in counts.cpu().tolist()],
            "min": float(lower_bound),
            "max": float(upper_bound),
            "below_min": int(below_min),
            "above_max": int(above_max),
        }

    def _empty_bin_result(self, bin_count, min_value, max_value, values):
        """Return zero bins and infinite overflow counts without finite data."""
        below_min = 0
        above_max = 0
        if min_value is not None:
            below_min = int((values < float(min_value)).sum().item())
        if max_value is not None:
            above_max = int((values >= float(max_value)).sum().item())
        return {
            "data": [0] * bin_count,
            "min": float(min_value) if min_value is not None else None,
            "max": float(max_value) if max_value is not None else None,
            "below_min": below_min,
            "above_max": above_max,
        }

    def get_population_size(self):
        """Return the current number of animals."""
        return len(self.population)

    def get_values(self):
        """Return current core scalar values as Python values."""
        population_size = self.get_population_size()
        average_age = None
        if population_size > 0:
            age_column = self.population_fields["age"]
            average_age = self.population[:, age_column].mean().item()
        return {
            "count": population_size,
            "avg_age": average_age,
            "born": self.last_born,
            "dead": self.last_death,
            "prop_aging": 0.0,
        }

    def should_stop(self):
        """Return no model-specific stop reason."""
        return None
