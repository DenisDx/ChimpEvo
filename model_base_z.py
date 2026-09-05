"""Fixed-fecundity beta model with independent mutation sign bias."""

import torch

from model_base_fast_fixed_fecundity import Model_base_fast_fixed_fecundity


def sample_z_mutation_shifts(
    mutation_x,
    mutation_s,
    mutation_z,
    count,
    dtype,
    device,
):
    """Sample mutation shifts with independent positive-branch bias Z."""
    positive_probability = mutation_z / 2.0 + 0.5
    positive_upper = mutation_x * (mutation_s + 1.0)
    negative_lower = mutation_x * (mutation_s - 1.0)
    positive_mask = torch.rand(count, device=device) < positive_probability
    unit_draws = torch.rand(count, dtype=dtype, device=device)
    positive_shifts = unit_draws * positive_upper
    negative_shifts = negative_lower + unit_draws * -negative_lower
    return torch.where(positive_mask, positive_shifts, negative_shifts)


class Model_base_z(Model_base_fast_fixed_fecundity):
    """Provide fixed-fecundity beta dynamics with mutation sign bias Z."""

    @staticmethod
    def description():
        """Return the mutation-sign-bias model description as lightweight Markdown."""
        return """# Fast fixed-fecundity beta model with mutation sign bias Z

## Purpose

Separates the probability of positive versus negative beta mutations from the
mutation interval asymmetry. This supports experiments where mutation direction
has its own evolvable or controlled bias.

## Inheritance

Inherits mortality, beta inheritance, fixed per-parent annual fecundity,
batched Torch reproduction, outputs, graphs, and stopping behavior from
`Model_base_fast_fixed_fecundity`.

## Difference from its parent

- Adds `mutation_z` in `[-1, 1]` as the mutation sign bias `Z`.
- `Z` controls how often the positive mutation branch is selected.
- `S` still controls the lengths of the positive and negative shift intervals.

For offspring selected to mutate:

$$P(positive) = (Z + 1) / 2$$

The mutation shift is sampled from the selected branch:

$$delta_beta ~ Uniform(0, X*(S+1))$$

or:

$$delta_beta ~ Uniform(X*(S-1), 0)$$

Then `beta_child = parental_mean + delta_beta`.

## Equivalence

When `Z = S`, this mixture is distributionally equivalent to the parent's
uniform mutation interval:

$$delta_beta ~ Uniform(X*(S-1), X*(S+1))$$

Exact seeded trajectories may still differ because this model consumes an
additional random draw to select the branch.
"""

    @staticmethod
    def add_settings():
        """Declare inherited settings and independent mutation sign bias Z."""
        return {
            **Model_base_fast_fixed_fecundity.add_settings(),
            "mutation_z": {
                "description": (
                    "Mutation sign bias Z. The positive branch probability is "
                    "(Z+1)/2; range [-1, 1]."
                ),
                "default": 0.0,
                "type": "float",
                "min": -1.0,
                "max": 1.0,
            },
        }

    def apply_reproduction(self):
        """Create fixed-capacity offspring using Z-biased mutation shifts."""
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
        current_population = self.population.shape[0]
        birth_count = min(max_births, max_population - current_population)
        if birth_count <= 0:
            return self.last_born

        parent_slots = available_slots[torch.randperm(
            available_slots.numel(),
            device=self.device,
        )[:2 * birth_count]].reshape(birth_count, 2)
        child_betas = self.population[parent_slots, beta_column].mean(dim=1)

        mutation_probability = self.settings["mutation_probability"]
        mutation_mask = torch.rand(birth_count, device=self.device) < mutation_probability
        if mutation_mask.any():
            mutation_shifts = sample_z_mutation_shifts(
                self.settings["mutation_x"],
                self.settings["mutation_s"],
                self.settings["mutation_z"],
                birth_count,
                self.population.dtype,
                self.device,
            )
            child_betas += mutation_shifts * mutation_mask
        if self.settings.get("beta_only_positive", False):
            child_betas.clamp_(min=0.0)

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


ModelBaseZ = Model_base_z
