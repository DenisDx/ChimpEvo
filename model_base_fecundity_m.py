"""Fixed-fecundity beta model with multiplicative mutation effects."""

import torch

from model_base import Model_base


class Model_base_fecundity_m(Model_base):
    """Provide batched fixed-fecundity reproduction with beta multipliers."""

    supports_beta_only_positive = False

    @staticmethod
    def description():
        """Return the multiplicative mutation model description as lightweight Markdown."""
        return """# Fixed-fecundity beta model with multiplicative mutations

## Purpose

Studies beta/Gompertz evolution with limited annual parent participation and
multiplicative, rather than additive, mutation effects.

## Inheritance

Inherits mortality, population fields, statistics, graphs, and stopping
behavior from `Model_base`.

## Difference from its parent

- Uses batched Torch reproduction and one population concatenation per year.
- Limits every mature animal to `floor(fecundity)` parent slots annually.
- Replaces the additive X/S beta mutation interval with a multiplier.
- Does not support `beta_only_positive`; negative beta values remain valid.

With N mature animals, annual births are bounded by:

$$births <= floor(N * floor(fecundity) / 2)$$

## Multiplicative mutation

An unmutated child receives the arithmetic mean of its two parental beta
values. Every child is then tested against `mutation_probability`. For a
mutated child, `mutation_s` sets the probability of an upward multiplier:

$$P(upward) = (mutation_s + 1) / 2$$

If upward is selected, beta is multiplied by `1 + mutation_x`; otherwise it
is divided by the same positive factor:

$$beta_child = beta_parent_mean * (1 + mutation_x)$$

or:

$$beta_child = beta_parent_mean / (1 + mutation_x)$$

Thus `mutation_x` is a relative mutation magnitude and `mutation_s` is the
upward-mutation bias. `mutation_x = 0` leaves beta unchanged regardless of
the selected direction.
"""

    @staticmethod
    def add_settings():
        """Describe multiplicative X and S semantics over inherited settings."""
        return {
            **{
                name: metadata
                for name, metadata in Model_base.add_settings().items()
                if name != "beta_only_positive"
            },
            "mutation_x": {
                "description": "Relative beta multiplier magnitude; mutations multiply or divide by 1 + X.",
                "default": 1.0,
                "type": "float",
                "min": 0.0,
                "max": 10.0,
            },
            "mutation_s": {
                "description": "Upward multiplier bias S; upward probability is (S + 1) / 2.",
                "default": 0.0,
                "type": "float",
                "min": -1.0,
                "max": 1.0,
            },
        }

    def apply_reproduction(self):
        """Create fixed-capacity offspring with multiplicative beta mutations."""
        self.last_born = 0
        max_population = int(self.settings["max_population"])
        mature_age = self.settings["mature_age"]
        parent_capacity = int(self.settings["fecundity"])
        age_column = self.population_fields["age"]
        beta_column = self.population_fields["beta"]
        mature_indices = torch.where(self.population[:, age_column] >= mature_age)[0]
        if mature_indices.numel() < 2 or parent_capacity < 1:
            return self.last_born

        available_slots = mature_indices.repeat_interleave(parent_capacity)
        max_births = available_slots.numel() // 2
        birth_count = min(max_births, max_population - self.population.shape[0])
        if birth_count <= 0:
            return self.last_born

        parent_slots = available_slots[torch.randperm(
            available_slots.numel(), device=self.device,
        )[:2 * birth_count]].reshape(birth_count, 2)
        child_betas = self.population[parent_slots, beta_column].mean(dim=1)
        mutation_mask = (
            torch.rand(birth_count, device=self.device)
            < self.settings["mutation_probability"]
        )
        if mutation_mask.any():
            upward_probability = self.settings["mutation_s"] / 2.0 + 0.5
            upward = torch.rand(birth_count, device=self.device) < upward_probability
            multiplier = 1.0 + self.settings["mutation_x"]
            factors = torch.where(
                upward,
                torch.full_like(child_betas, multiplier),
                torch.full_like(child_betas, 1.0 / multiplier),
            )
            child_betas *= torch.where(mutation_mask, factors, torch.ones_like(factors))
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