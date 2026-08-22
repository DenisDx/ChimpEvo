"""Optimized beta model with per-parent annual fecundity limits."""

import torch

from model_base import Model_base, mutation_interval


class Model_base_fecundity(Model_base):
    """Provide beta dynamics with a fixed annual parent participation limit."""

    def apply_reproduction(self):
        """Create batched offspring while limiting every parent to annual capacity."""
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


class Model_base_fast_fixed_fecundity(Model_base_fecundity):
    """Expose the per-parent fecundity model through the loader naming contract."""