"""Diploid beta model with a codominant mortality phenotype."""

import torch

from model import Model
from model_base_z import Model_base_z, sample_z_mutation_shifts


class Model_base_diploid(Model_base_z):
    """Provide diploid beta inheritance with a stored codominant phenotype."""

    @staticmethod
    def description():
        """Return the diploid beta model description as lightweight Markdown."""
        return """# Diploid beta model

## Purpose

Adds diploid inheritance to the fixed-fecundity, Z-biased beta model. Every
animal stores two inherited beta alleles and one expressed beta phenotype used
by mortality and existing outputs.

## Inheritance

Inherits settings, fixed per-parent annual fecundity, batched Torch execution,
Z-biased mutation shifts, mortality, outputs, graphs, and stopping behavior from
`Model_base_z`.

## Difference from its parent

- Stores two alleles, `beta1` and `beta2`, in addition to the derived `beta`.
- Each child independently receives one randomly selected allele from each
  parent; either parental allele has probability `0.5`.
- `mutation_probability` is checked independently twice per child, once for
  each inherited allele.
- Every allele selected to mutate receives its own independently sampled X/S/Z
  mutation shift.

## Codominant phenotype

The expressed phenotype is the arithmetic mean of both alleles:

$$beta = (beta1 + beta2) / 2$$

This codominant representation gives both inherited allele sets an equal,
additive contribution. It is appropriate here because beta represents the
combined behavior of many genes rather than dominance at one single gene.

The model intentionally stores the redundant triplet `beta`, `beta1`, and
`beta2` for simple state handling and compatibility. The public `beta` field
remains in the same column as in earlier models.

## Initialization

Every initial animal starts homozygous at the configured value:

$$beta1 = beta2 = beta = beta_initial$$

## Reproduction and mutation

Before mutation, the inherited alleles are:

$$allele1 ~ Choice(parent1.beta1, parent1.beta2)$$

$$allele2 ~ Choice(parent2.beta1, parent2.beta2)$$

Two independent mutation checks are then performed:

$$M1 ~ Bernoulli(mutation_probability)$$

$$M2 ~ Bernoulli(mutation_probability)$$

For each successful check, the model independently applies the inherited X/S/Z
shift rule from `Model_base_z`. The probability of at least one mutated allele
in a child is therefore `1 - (1 - mutation_probability)^2`.

After both possible mutations, `beta` is immediately stored as the arithmetic
mean of the final `beta1` and `beta2` values.

## Mortality compatibility

Mortality is not overridden. The inherited Gompertz mortality function reads
the stored, codominant `beta` phenotype exactly as in previous models. Existing
beta statistics, graphs, and stabilization rules also continue to use that
field.
"""

    @staticmethod
    def add_population_fields():
        """Declare age, compatible beta phenotype, and two private alleles."""
        return {
            **Model.add_population_fields(),
            "beta": {"public": True},
            "beta1": {"public": False},
            "beta2": {"public": False},
        }

    def initialize_population(self):
        """Initialize ages and homozygous beta alleles with their mean phenotype."""
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
        """Create offspring with independent allele inheritance and mutation."""
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
        current_population = self.population.shape[0]
        birth_count = min(max_births, max_population - current_population)
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
        allele_choices = torch.randint(
            0,
            2,
            (birth_count, 2),
            device=self.device,
        )
        selected_columns = allele_columns[allele_choices]
        child_alleles = self.population[parent_slots, selected_columns]

        mutation_probability = self.settings["mutation_probability"]
        mutation_mask = (
            torch.rand((birth_count, 2), device=self.device) < mutation_probability
        )
        if mutation_mask.any():
            mutation_shifts = sample_z_mutation_shifts(
                self.settings["mutation_x"],
                self.settings["mutation_s"],
                self.settings["mutation_z"],
                birth_count * 2,
                self.population.dtype,
                self.device,
            ).reshape(birth_count, 2)
            child_alleles += mutation_shifts * mutation_mask

        child_betas = child_alleles.mean(dim=1)
        children = torch.empty(
            (birth_count, len(self.population_fields)),
            dtype=self.population.dtype,
            device=self.device,
        )
        children[:, age_column] = 0.0
        children[:, beta_column] = child_betas
        children[:, beta1_column] = child_alleles[:, 0]
        children[:, beta2_column] = child_alleles[:, 1]
        self.population = torch.cat([self.population, children], dim=0)
        self.last_born = birth_count
        return self.last_born
