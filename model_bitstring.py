"""Bit-string-inspired diploid model with delayed dominant allele effects."""

import torch

from model import Model
from model_alleles import Model_alleles


class Model_bitstring(Model_alleles):
    """Provide a bit-string-inspired model with mutable allele activation ages."""

    @staticmethod
    def description():
        """Return the bit-string-inspired model description as lightweight Markdown."""
        return """# Bit-string-inspired delayed beta model

## Purpose

Studies whether evolution of dominant allele activation ages produces a
Gompertz-like mortality curve. Each locus has a large beta effect centered on
`beta_central`, but it emerges only after its inherited delta activation age.
The model supports experiments on the effect of background mortality `lambda`
and other parameters on the resulting effective beta.

## Inheritance

Inherits vectorized diploid-locus storage, fixed parent fecundity, mortality,
graphs, stopping, dominance mutation, delta mutation, and aggregate statistics
from `Model_alleles`.

## Difference from its parent

- Dominance and delta fields are always present; each locus uses its strictly
  more dominant allele, with allele 2 winning dominance ties.
- Every allele has a delayed effect, including when `delta_x` is zero.
- A beta mutation resets beta from the fixed `beta_central` value instead of
  modifying the inherited beta.
- Negative pre-activation contributions are permanently clamped to zero.

## Effective beta

For selected dominant beta $b_i$, activation age $d_i$, and current age $t$,
the public beta is the fitted Gompertz coefficient:

$$beta_effective(t) = mean_i(max(0, b_i * (t - d_i) / t))$$

For $t=0$, an allele with positive delta contributes zero and an allele with
zero delta contributes its beta. Mortality uses:

$$m(t) = clamp(alpha * exp(beta_effective(t) * t) + Lambda, 0, 1)$$

The public `beta` is retained for graphs and CSV output, but is recomputed from
the allele state before mortality rather than inherited directly.

## Mutation

Each child inherits one allele from each parent at every locus. Without
mutation, beta, dominance, and delta are inherited together. With probability
`mutation_probability`, dominance receives $Uniform(-0.5, 0.5)$ and delta
receives its inherited signed, reversion-biased shift before clamping to zero.

A beta mutation ignores the parental beta. With $P(multiply)=(Z+1)/2$:

$$b_new = beta_central * (1 + X(S+1))$$

for multiplication, or:

$$b_new = beta_central / (1 + X(1-S))$$

otherwise. The same central-beta rule independently initializes every allele.
With default $X=S=Z=0$, all beta alleles equal `beta_central`.

## Relation to the Bit-String Model

This is bit-string-inspired rather than a literal implementation of the Penna
Bit-String Model. It stores continuous beta, dominance, and delta values rather
than binary loci; beta mutations are reversible resets around a central value;
and delta mutations may increase or decrease activation age. Its shared idea is
that inherited deleterious effects activate only after locus-specific ages.
"""

    @staticmethod
    def add_settings():
        """Declare fixed-mode allele settings and central beta initialization."""
        settings = dict(Model_alleles.add_settings())
        for name in (
            "beta_initial",
            "beta_only_positive",
            "use_dominance",
            "use_multiplication",
        ):
            settings.pop(name)
        settings["mutation_x"] = {
            "description": "Central-beta multiplicative mutation magnitude X",
            "default": 0.0,
            "type": "float",
            "min": 0.0,
            "max": 10.0,
        }
        settings["mutation_s"] = {
            "description": "Central-beta multiplicative mutation asymmetry S",
            "default": 0.0,
            "type": "float",
            "min": -1.0,
            "max": 1.0,
        }
        settings["mutation_z"] = {
            "description": "Central-beta multiplication probability bias Z",
            "default": 0.0,
            "type": "float",
            "min": -1.0,
            "max": 1.0,
        }
        settings["beta_central"] = {
            "description": "Central beta used to initialize and reset mutated alleles",
            "default": 2.7,
            "type": "float",
            "min": 0.0001,
            "max": 10.0,
        }
        settings["delta_initial"] = {
            "description": "Initial activation age applied to every allele",
            "default": 20.0,
            "type": "float",
            "min": 0.0,
            "max": 1000.0,
        }
        return settings

    @staticmethod
    def add_population_fields(config=None):
        """Declare public age/beta and permanent private allele attributes."""
        config = config or {}
        count = int(config.get("N_alleles", 100))
        fields = {**Model.add_population_fields(), "beta": {"public": True}}
        for name in ("beta1", "beta2", "dom1", "dom2", "delta1", "delta2"):
            fields.update({f"{name}_{index}": {"public": False} for index in range(count)})
        return fields

    @staticmethod
    def get_estimated_memory_consumption(config):
        """Return the peak-memory estimate for permanent allele attribute columns."""
        count = int(config.get("N_alleles", 100))
        return int(config.get("max_population", 0)) * (2 + 6 * count) * 4 * 2

    def _uses_dominance(self):
        """Return that every locus uses dominant allele selection."""
        return True

    def _uses_delta(self):
        """Return that every locus always has a delayed effect."""
        return True

    def _uses_multiplication(self):
        """Return that beta mutations use multiplicative central-beta draws."""
        return True

    def _clamps_effective_beta(self):
        """Return that inactive allele contributions never become negative."""
        return True

    def _central_beta_values(self, shape, dtype):
        """Return independent central-beta multiplicative draws with the requested shape."""
        multiply_mask = torch.rand(shape, device=self.device) < (
            self.settings["mutation_z"] / 2.0 + 0.5
        )
        mutation_x = self.settings["mutation_x"]
        mutation_s = self.settings["mutation_s"]
        upward_multiplier = 1.0 + mutation_x * (mutation_s + 1.0)
        downward_divisor = 1.0 + mutation_x * (1.0 - mutation_s)
        return torch.where(
            multiply_mask,
            torch.full(shape, self.settings["beta_central"] * upward_multiplier, dtype=dtype, device=self.device),
            torch.full(shape, self.settings["beta_central"] / downward_divisor, dtype=dtype, device=self.device),
        )

    def initialize_population(self):
        """Initialize independent beta alleles and fixed initial activation ages."""
        count = int(self.settings["N_alleles"])
        population_size = int(self.settings["initial_population"])
        ages = torch.randint(
            0,
            int(self.settings["initial_age_max"]) + 1,
            (population_size,),
            dtype=torch.float32,
            device=self.device,
        )
        beta_pairs = self._central_beta_values(
            (population_size, 2, count),
            torch.float32,
        )
        dominance = torch.zeros((population_size, count), dtype=torch.float32, device=self.device)
        delta = torch.full(
            (population_size, count),
            self.settings["delta_initial"],
            dtype=torch.float32,
            device=self.device,
        )
        columns = [ages, beta_pairs.mean(dim=(1, 2)), beta_pairs[:, 0], beta_pairs[:, 1], dominance, dominance, delta, delta]
        self.avg_beta_ema = None
        self._previous_avg_beta = None
        self._beta_changes = []
        self._consecutive_ema_below_threshold = 0
        self._set_population(torch.cat([column.reshape(population_size, -1) for column in columns], dim=1))
        self._update_effective_beta()

    def _mutate_beta_alleles(self, inherited, mutation_mask):
        """Replace each mutated inherited beta allele with a central-beta draw."""
        central_betas = self._central_beta_values(inherited.shape, inherited.dtype)
        inherited.copy_(torch.where(mutation_mask, central_betas, inherited))