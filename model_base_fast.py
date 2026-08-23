"""Optimized beta-based chimp population model."""

import torch

from model_base import Model_base, mutation_interval


class Model_base_fast(Model_base):
    """Provide the beta model with batched device-side reproduction."""

    @staticmethod
    def description():
        """Return the optimized beta model description as lightweight Markdown."""
        return """# Fast beta/Gompertz model

## Purpose

Runs the baseline beta/Gompertz population model with faster reproduction for
large populations and CUDA workloads.

## Inheritance

Inherits all biological rules, settings, values, graphs, and stopping behavior
from `Model_base`.

## Difference from its parent

- Selects all parent pairs in one Torch operation on the configured device.
- Applies mutation and creates all offspring as one tensor batch.
- Concatenates offspring to the population once per year instead of once per
    birth.
- Keeps the same unlimited per-parent annual participation as `Model_base`.

The probability model is unchanged, but exact seeded trajectories can differ
because reproduction uses Torch random draws instead of Python and NumPy draws.
"""

    def apply_reproduction(self):
        """Create a batch of offspring up to fecundity and population limits."""
        self.last_born = 0
        max_population = int(self.settings["max_population"])
        mature_age = self.settings["mature_age"]
        fecundity = self.settings["fecundity"]
        age_column = self.population_fields["age"]
        beta_column = self.population_fields["beta"]
        mature_indices = torch.where(self.population[:, age_column] >= mature_age)[0]
        mature_count = mature_indices.numel()
        if mature_count < 2:
            return self.last_born

        current_population = self.population.shape[0]
        max_growth = int(mature_count * fecundity)
        birth_count = min(max_growth, max_population - current_population)
        if birth_count <= 0:
            return self.last_born

        parent_positions = torch.randint(
            mature_count,
            (birth_count, 2),
            device=self.device,
        )
        parent_indices = mature_indices[parent_positions]
        parent_betas = self.population[parent_indices, beta_column]
        child_betas = parent_betas.mean(dim=1)

        mutation_probability = self.settings["mutation_probability"]
        mutation_mask = torch.rand(birth_count, device=self.device) < mutation_probability
        if mutation_mask.any():
            mutation_x = self.settings["mutation_x"]
            mutation_s = self.settings["mutation_s"]
            lower, upper = mutation_interval(mutation_x, mutation_s)
            mutation_shifts = torch.empty(
                birth_count,
                dtype=self.population.dtype,
                device=self.device,
            ).uniform_(lower, upper)
            child_betas += mutation_shifts * mutation_mask

        children = torch.empty(
            (birth_count, len(self.population_fields)),
            dtype=self.population.dtype,
            device=self.device,
        )
        children[:, age_column] = 0.0
        children[:, beta_column] = child_betas
        self.population = torch.cat([self.population, children], dim=0)
        self.last_born = birth_count
        return self.last_born