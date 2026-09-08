"""Diploid beta model with asymmetric multiplicative mutations."""

import torch

from model_base_diploid import Model_base_diploid


class Model_base_diploid_m(Model_base_diploid):
    """Provide diploid inheritance with per-allele multiplicative mutations."""

    supports_beta_only_positive = False

    @staticmethod
    def description():
        """Return the diploid multiplicative mutation model description."""
        return """# Diploid beta model with multiplicative mutations

## Purpose

Studies diploid beta inheritance where each inherited allele can mutate by a
proportional multiplier instead of receiving an additive beta shift.

## Inheritance

Inherits diploid population fields, codominant phenotype calculation, fixed
per-parent annual fecundity, batched Torch execution, mortality, outputs,
graphs, and stopping behavior from `Model_base_diploid`.

## Difference from its parent

- Retains the `beta`, `beta1`, and `beta2` population schema.
- Each child receives one randomly selected allele from each parent.
- Each inherited allele has an independent mutation check.
- Replaces additive X/S/Z shifts with asymmetric multiplicative changes.
- Does not support `beta_only_positive`; negative beta alleles remain valid.

## Codominant phenotype

The public beta phenotype remains the arithmetic mean of the two inherited
alleles:

$$beta = (beta1 + beta2) / 2$$

Mortality, beta statistics, graphs, and stopping behavior use this stored
phenotype exactly as in `Model_base_diploid`.

## Allele inheritance

For each child, one allele is sampled independently from each parent:

$$allele1 ~ Choice(parent1.beta1, parent1.beta2)$$

$$allele2 ~ Choice(parent2.beta1, parent2.beta2)$$

Each selected allele independently mutates with:

$$M_i ~ Bernoulli(mutation_probability)$$

Thus the probability that a child has at least one mutated allele is:

$$P(M1 or M2) = 1 - (1 - mutation_probability)^2$$

## Multiplicative X/S/Z mutation

For every allele selected to mutate, `mutation_z` chooses the mutation
direction independently:

$$P(multiply) = (Z + 1) / 2$$

`mutation_x` sets the base relative magnitude. `mutation_s` makes the upward
and downward magnitudes asymmetric while preserving the symmetric case at
`S = 0`:

$$upward_multiplier = 1 + X * (S + 1)$$

$$downward_divisor = 1 + X * (1 - S)$$

The final allele is therefore:

$$allele_new = allele_inherited * upward_multiplier$$

when multiplication is selected, or:

$$allele_new = allele_inherited / downward_divisor$$

when division is selected. Here $X >= 0$ and $-1 <= S, Z <= 1$, so both
multipliers are positive. At $S = 0$, the rules reduce to multiplication or
division by $1 + X$. At $Z = 0$, both directions have probability $0.5$.

After both allele mutations, the model stores the codominant phenotype:

$$beta_child = (beta1_child + beta2_child) / 2$$
"""

    @staticmethod
    def add_settings():
        """Declare X, S, and Z metadata for multiplicative allele mutations."""
        return {
            **{
                name: metadata
                for name, metadata in Model_base_diploid.add_settings().items()
                if name != "beta_only_positive"
            },
            "mutation_x": {
                "description": "Base relative multiplier magnitude X for mutated alleles.",
                "default": 1.0,
                "type": "float",
                "min": 0.0,
                "max": 10.0,
            },
            "mutation_s": {
                "description": "Multiplier magnitude asymmetry S; upward uses X*(S+1), downward uses X*(1-S).",
                "default": 0.0,
                "type": "float",
                "min": -1.0,
                "max": 1.0,
            },
            "mutation_z": {
                "description": "Multiply-direction bias Z; multiplication probability is (Z+1)/2.",
                "default": 0.0,
                "type": "float",
                "min": -1.0,
                "max": 1.0,
            },
        }

    def initialize_population(self):
        """Initialize homozygous diploid beta state without positive-beta constraints."""
        initial_population = int(self.settings["initial_population"])
        initial_age_max = int(self.settings["initial_age_max"])
        beta_initial = float(self.settings["beta_initial"])
        self.avg_beta_ema = None
        self._previous_avg_beta = None
        self._beta_changes = []
        self._consecutive_ema_below_threshold = 0
        ages = torch.randint(
            0,
            initial_age_max + 1,
            (initial_population,),
            dtype=torch.float32,
            device=self.device,
        )
        betas = torch.full(
            (initial_population,),
            beta_initial,
            dtype=torch.float32,
            device=self.device,
        )
        self._set_population(torch.stack([ages, betas, betas, betas], dim=1))

    def apply_reproduction(self):
        """Create offspring with independent multiplicative mutations per allele."""
        self.last_born = 0
        max_population = int(self.settings["max_population"])
        mature_age = self.settings["mature_age"]
        parent_capacity = int(self.settings["fecundity"])
        age_column = self.population_fields["age"]
        beta_column = self.population_fields["beta"]
        beta1_column = self.population_fields["beta1"]
        beta2_column = self.population_fields["beta2"]
        mature_indices = torch.where(self.population[:, age_column] >= mature_age)[0]
        if mature_indices.numel() < 2 or parent_capacity < 1:
            return self.last_born

        available_slots = mature_indices.repeat_interleave(parent_capacity)
        max_births = available_slots.numel() // 2
        birth_count = min(max_births, max_population - self.population.shape[0])
        if birth_count <= 0:
            return self.last_born

        parent_slots = available_slots[torch.randperm(
            available_slots.numel(),
            device=self.device,
        )[:2 * birth_count]].reshape(birth_count, 2)
        allele_columns = torch.tensor(
            [beta1_column, beta2_column],
            dtype=torch.long,
            device=self.device,
        )
        allele_choices = torch.randint(0, 2, (birth_count, 2), device=self.device)
        child_alleles = self.population[parent_slots, allele_columns[allele_choices]]

        mutation_mask = torch.rand(
            (birth_count, 2), device=self.device,
        ) < self.settings["mutation_probability"]
        multiply_mask = torch.rand(
            (birth_count, 2), device=self.device,
        ) < (self.settings["mutation_z"] / 2.0 + 0.5)
        mutation_x = self.settings["mutation_x"]
        mutation_s = self.settings["mutation_s"]
        upward_multiplier = 1.0 + mutation_x * (mutation_s + 1.0)
        downward_divisor = 1.0 + mutation_x * (1.0 - mutation_s)
        factors = torch.where(
            multiply_mask,
            torch.full_like(child_alleles, upward_multiplier),
            torch.full_like(child_alleles, 1.0 / downward_divisor),
        )
        child_alleles *= torch.where(
            mutation_mask,
            factors,
            torch.ones_like(factors),
        )

        children = torch.empty(
            (birth_count, len(self.population_fields)),
            dtype=self.population.dtype,
            device=self.device,
        )
        children[:, age_column] = 0.0
        children[:, beta1_column] = child_alleles[:, 0]
        children[:, beta2_column] = child_alleles[:, 1]
        children[:, beta_column] = child_alleles.mean(dim=1)
        self.population = torch.cat([self.population, children], dim=0)
        self.last_born = birth_count
        return self.last_born