"""Multi-locus diploid beta model with optional dominance and delta alleles."""

import torch

from model import Model
from model_base_z import Model_base_z, sample_z_mutation_shifts


INITIAL_DOMINANCE = 1.0
DELTA_INITIAL_MAX = 0.0


class Model_alleles(Model_base_z):
    """Provide vectorized multi-locus diploid beta inheritance."""

    population_fields_depend_on_settings = True

    @staticmethod
    def description():
        """Return the multi-locus model description as lightweight Markdown."""
        return """# Multi-locus diploid beta model

## Purpose

Models `N_alleles` independent diploid beta loci while retaining one public,
effective `beta` phenotype for inherited mortality and output compatibility.

## Inheritance

Inherits fixed fecundity, Z-biased beta mutation, mortality, graphs, and
stopping behavior from `Model_base_z`. Every child independently inherits one
allele per locus from each parent. A mutation check applies to every inherited
allele and jointly mutates its beta, dominance, and delta values when present.

## Difference from its parent

- Stores two private beta alleles for every locus.
- Optionally selects the more dominant allele in every pair.
- Optionally shifts each allele's age effect by a mutable delta.

## Settings and modes

`N_alleles` selects the number N of independent diploid loci. Each animal
therefore stores `beta1_i` and `beta2_i` for every index i from 0 to N-1.
These allele fields are private; the public `beta` field remains the
single mortality phenotype.

`use_dominance` selects how the two alleles at each locus contribute. When it
is false, both alleles contribute. When it is true, the model also stores
`dom1_i` and `dom2_i` and selects the beta belonging to the greater dominance
value. The comparison is strict, so equality selects allele 2 consistently.

`delta_x` controls whether delayed allele effects are active. With
`delta_x = 0`, delta fields do not exist and beta is calculated only at birth.
With `delta_x > 0`, the model stores `delta1_i` and `delta2_i`, mutates them
with beta, and recalculates effective beta before every mortality step.

`delta_reversion` controls the tendency of a delta mutation to move toward
zero. A value of zero makes upward and downward shifts equally likely. For a
positive reversion value, the probability of an upward shift decreases linearly
from 0.5 at delta = 0 to 0 at delta = delta_reversion; larger deltas can only
decrease.

## Effective beta

Without dominance or delta, both alleles from all loci contribute equally:

$$beta = mean(beta1_0, ..., beta1_(N-1), beta2_0, ..., beta2_(N-1))$$

With dominance enabled, one beta is selected per locus and the phenotype is:

$$beta = mean(select(beta1_i, beta2_i), i = 0, ..., N-1)$$

where `select` chooses `beta1_i` only when dominance 1 is greater than
dominance 2; otherwise it chooses `beta2_i`.

With delta enabled, the selected or codominant set is recalculated at current
age t. The per-allele contribution is:

$$beta_i^effective(t) = beta_i * (t - delta_i) / t$$

and public beta is the mean of those contributions. At t = 0, an allele with
positive delta contributes zero; an allele with zero delta contributes its beta.
The inherited mortality calculation then remains:

$$m(t, beta) = clamp(alpha * exp(beta * t) + Lambda, 0, 1)$$

This effective-beta construction is an approximation. It is not equivalent to
averaging individual Gompertz mortality probabilities.

## Compatibility modes

To reproduce the biological rules of `model_base_diploid`, set:

$$N_alleles = 1, delta_x = 0, use_dominance = false$$

The two stored alleles then form one codominant pair and public beta is:

$$beta = (beta1_0 + beta2_0) / 2$$

This follows from the general multi-locus formula; it is not a separate
special case. Keep the inherited `fecundity`, `mutation_probability`,
`mutation_x`, `mutation_s`, and `mutation_z` settings the same to match the
diploid model's biological parameters. The models can still produce different
seeded trajectories because their vectorized random draws occur in a different
order.

To enable delayed effects while keeping one locus, use `N_alleles = 1` with
`delta_x > 0`. To compare codominant and dominant expression for the same
multi-locus configuration, change only `use_dominance`; enabling it creates
dominance fields and selects one allele from each pair.

## Mutation and constraints

Each inherited allele is independently tested against `mutation_probability`.
When selected, beta receives the inherited X/S/Z mutation shift. If present,
the same mutation also changes dominance by a uniform shift from -0.5 to 0.5
and changes delta by a signed uniform shift from 0 to `delta_x`, using
`delta_reversion`.
Dominance has no bounds; delta is clamped to zero after mutation.

When `beta_only_positive` is enabled, inherited beta alleles are clamped at
zero. In delta mode, every negative per-allele effective-beta contribution is
also clamped to zero before averaging. A negative `beta_initial` is invalid in
this mode.

## Statistics

Alongside inherited population statistics, the model reports the mean
within-animal variance of all beta alleles. Dominance mode additionally reports
the corresponding dominant-allele variance and dominance aggregates. Delta
mode reports mean delta and the mean count of high-delta alleles. These values
are reduced on the selected Torch device; only final scalar results are moved
to Python for output.
"""

    @staticmethod
    def add_settings():
        """Declare multi-locus settings in addition to inherited beta settings."""
        return {
            **Model_base_z.add_settings(),
            "N_alleles": {
                "description": "Independent diploid beta loci", "default": 100,
                "type": "int", "min": 1, "max": 10000,
            },
            "delta_x": {
                "description": "Maximum absolute delta mutation shift in years", "default": 0.0,
                "type": "float", "min": 0.0, "max": 1000.0,
            },
            "delta_reversion": {
                "description": "Delta value where upward mutation probability reaches zero", "default": 12.0,
                "type": "float", "min": 0.0, "max": 1000.0,
            },
            "use_dominance": {
                "description": "Use the higher-dominance allele in each locus", "default": False,
                "type": "bool",
            },
        }

    @staticmethod
    def add_population_fields(config=None):
        """Declare public phenotype and private dynamic allele fields."""
        config = config or {}
        count = int(config.get("N_alleles", 100))
        fields = {**Model.add_population_fields(), "beta": {"public": True}}
        for name in ("beta1", "beta2"):
            fields.update({f"{name}_{index}": {"public": False} for index in range(count)})
        if config.get("use_dominance", False):
            for name in ("dom1", "dom2"):
                fields.update({f"{name}_{index}": {"public": False} for index in range(count)})
        if float(config.get("delta_x", 0.0)) != 0.0:
            for name in ("delta1", "delta2"):
                fields.update({f"{name}_{index}": {"public": False} for index in range(count)})
        return fields

    @staticmethod
    def add_values():
        """Declare aggregate allele statistics alongside inherited scalar results."""
        return {
            **Model_base_z.add_values(),
            "avg_allele_beta_variance": {"title": "Average allele beta variance", "annual": True, "final": True, "format": ".6f"},
            "avg_dominant_beta_variance": {"title": "Average dominant beta variance", "annual": True, "final": True, "format": ".6f"},
            "avg_dominance": {"title": "Average dominance", "annual": True, "final": True, "format": ".4f"},
            "dominance_variance": {"title": "Dominance variance", "annual": True, "final": True, "format": ".6f"},
            "avg_delta": {"title": "Average delta", "annual": True, "final": True, "format": ".4f"},
            "avg_high_delta_alleles": {"title": "Average high-delta alleles", "annual": True, "final": True, "format": ".4f"},
        }

    @staticmethod
    def get_estimated_memory_consumption(config):
        """Return a compact peak-memory estimate for dynamic allele tensors."""
        count = int(config.get("N_alleles", 100))
        columns = 2 + 2 * count
        if config.get("use_dominance", False):
            columns += 2 * count
        if float(config.get("delta_x", 0.0)) != 0.0:
            columns += 2 * count
        return int(config.get("max_population", 0)) * columns * 4 * 2

    def _columns(self, prefix):
        """Return ordered schema columns for one dynamic allele field prefix."""
        return [self.population_fields[f"{prefix}_{index}"] for index in range(self.settings["N_alleles"])]

    def _alleles(self, prefix):
        """Return one population-by-locus dynamic allele tensor view."""
        return self.population[:, self._columns(prefix)]

    def _selected_alleles(self):
        """Return effective per-locus beta and optional matching delta tensors."""
        beta1 = self._alleles("beta1")
        beta2 = self._alleles("beta2")
        if self.settings["use_dominance"]:
            choose_first = self._alleles("dom1") > self._alleles("dom2")
            betas = torch.where(choose_first, beta1, beta2)
            deltas = None
            if self.settings["delta_x"] != 0.0:
                deltas = torch.where(choose_first, self._alleles("delta1"), self._alleles("delta2"))
            return betas, deltas
        betas = torch.cat([beta1, beta2], dim=1)
        if self.settings["delta_x"] == 0.0:
            return betas, None
        return betas, torch.cat([self._alleles("delta1"), self._alleles("delta2")], dim=1)

    def _update_effective_beta(self):
        """Store the current age-dependent effective beta phenotype."""
        betas, deltas = self._selected_alleles()
        if deltas is None:
            effective = betas.mean(dim=1)
        else:
            ages = self.population[:, self.population_fields["age"]].unsqueeze(1)
            nonzero_age = ages != 0.0
            contributions = torch.where(
                nonzero_age,
                betas * (ages - deltas) / torch.where(nonzero_age, ages, torch.ones_like(ages)),
                torch.where(deltas > 0.0, torch.zeros_like(betas), betas),
            )
            if self.settings.get("beta_only_positive", False):
                contributions.clamp_(min=0.0)
            effective = contributions.mean(dim=1)
        if self.settings.get("beta_only_positive", False):
            effective.clamp_(min=0.0)
        self.population[:, self.population_fields["beta"]] = effective

    def initialize_population(self):
        """Initialize ages, homozygous beta loci, and optional allele attributes."""
        if self.settings.get("beta_only_positive", False) and self.settings["beta_initial"] < 0:
            raise ValueError("beta_initial must be nonnegative when beta_only_positive is enabled")
        count = int(self.settings["N_alleles"])
        population_size = int(self.settings["initial_population"])
        ages = torch.randint(0, int(self.settings["initial_age_max"]) + 1, (population_size,), dtype=torch.float32, device=self.device)
        beta = torch.full((population_size, count), float(self.settings["beta_initial"]), dtype=torch.float32, device=self.device)
        columns = [ages, beta.mean(dim=1), beta, beta]
        if self.settings["use_dominance"]:
            dominance = torch.zeros((population_size, count), dtype=torch.float32, device=self.device)
            columns.extend([dominance, dominance])
        if self.settings["delta_x"] != 0.0:
            delta = torch.full((population_size, count), DELTA_INITIAL_MAX, dtype=torch.float32, device=self.device)
            columns.extend([delta, delta])
        self.avg_beta_ema = None
        self._previous_avg_beta = None
        self._beta_changes = []
        self._consecutive_ema_below_threshold = 0
        self._set_population(torch.cat([column.reshape(population_size, -1) for column in columns], dim=1))
        self._update_effective_beta()

    def apply_reproduction(self):
        """Create offspring with vectorized per-locus inheritance and mutation."""
        self.last_born = 0
        age_column = self.population_fields["age"]
        mature = torch.where(self.population[:, age_column] >= self.settings["mature_age"])[0]
        capacity = int(self.settings["fecundity"])
        if mature.numel() < 2 or capacity < 1:
            return 0
        slots = mature.repeat_interleave(capacity)
        births = min(slots.numel() // 2, int(self.settings["max_population"]) - self.population.shape[0])
        if births <= 0:
            return 0
        parents = slots[torch.randperm(slots.numel(), device=self.device)[:2 * births]].reshape(births, 2)
        count = int(self.settings["N_alleles"])
        choices = torch.randint(0, 2, (births, 2, count), device=self.device)
        beta_pairs = torch.stack([self._alleles("beta1"), self._alleles("beta2")], dim=2)
        inherited = torch.stack([
            torch.gather(beta_pairs[parents[:, side]], 2, choices[:, side].unsqueeze(2)).squeeze(2)
            for side in range(2)
        ], dim=1)
        mutation_mask = torch.rand((births, 2, count), device=self.device) < self.settings["mutation_probability"]
        shifts = sample_z_mutation_shifts(self.settings["mutation_x"], self.settings["mutation_s"], self.settings["mutation_z"], births * 2 * count, self.population.dtype, self.device).reshape(births, 2, count)
        inherited += shifts * mutation_mask
        if self.settings.get("beta_only_positive", False):
            inherited.clamp_(min=0.0)
        child_fields = [torch.zeros((births, 1), dtype=self.population.dtype, device=self.device), inherited[:, 0], inherited[:, 1]]
        for prefix in ("dom", "delta"):
            enabled = prefix == "dom" and self.settings["use_dominance"] or prefix == "delta" and self.settings["delta_x"] != 0.0
            if not enabled:
                continue
            pairs = torch.stack([self._alleles(f"{prefix}1"), self._alleles(f"{prefix}2")], dim=2)
            values = torch.stack([
                torch.gather(pairs[parents[:, side]], 2, choices[:, side].unsqueeze(2)).squeeze(2)
                for side in range(2)
            ], dim=1)
            if prefix == "dom":
                values += (torch.rand_like(values) - 0.5) * mutation_mask
            else:
                magnitudes = torch.rand_like(values) * self.settings["delta_x"]
                reversion = self.settings["delta_reversion"]
                upward_probability = torch.full_like(values, 0.5) if reversion == 0 else 0.5 * (1.0 - torch.clamp(values / reversion, max=1.0))
                signs = torch.where(torch.rand_like(values) < upward_probability, 1.0, -1.0)
                values += signs * magnitudes * mutation_mask
                values.clamp_(min=0.0)
            child_fields.extend([values[:, 0], values[:, 1]])
        beta_effective = inherited.reshape(births, -1).mean(dim=1, keepdim=True)
        children = torch.cat([child_fields[0], beta_effective, *[field.reshape(births, -1) for field in child_fields[1:]]], dim=1)
        self.population = torch.cat([self.population, children], dim=0)
        self._update_effective_beta()
        self.last_born = births
        return births

    def apply_mortality(self):
        """Refresh delta-dependent beta immediately before inherited mortality."""
        if self.settings["delta_x"] != 0.0:
            self._update_effective_beta()
        return super().apply_mortality()

    def get_values(self):
        """Return inherited values plus device-side allele aggregate statistics."""
        values = super().get_values()
        empty = {"avg_allele_beta_variance": None, "avg_dominant_beta_variance": None, "avg_dominance": None, "dominance_variance": None, "avg_delta": None, "avg_high_delta_alleles": None}
        if not self.get_population_size():
            values.update(empty)
            return values
        all_betas = torch.cat([self._alleles("beta1"), self._alleles("beta2")], dim=1)
        values.update(empty)
        values["avg_allele_beta_variance"] = torch.var(all_betas, dim=1, correction=0).mean().item()
        if self.settings["use_dominance"]:
            dominant, _ = self._selected_alleles()
            dominance = torch.cat([self._alleles("dom1"), self._alleles("dom2")], dim=1)
            values["avg_dominant_beta_variance"] = torch.var(dominant, dim=1, correction=0).mean().item()
            dominance_variance, average_dominance = torch.var_mean(dominance, correction=0)
            values["avg_dominance"] = average_dominance.item()
            values["dominance_variance"] = dominance_variance.item()
        if self.settings["delta_x"] != 0.0:
            deltas = torch.cat([self._alleles("delta1"), self._alleles("delta2")], dim=1)
            values["avg_delta"] = deltas.mean().item()
            threshold = self.settings["delta_reversion"] * 0.5 if self.settings["delta_reversion"] else 0.0
            values["avg_high_delta_alleles"] = (deltas > threshold).sum(dim=1).float().mean().item()
        return values