"""Default beta-based chimp population model."""

import random
import math

import numpy as np
import torch

from model import Model


def mutation_interval(mutation_x, mutation_s):
    """Return mutation-shift bounds from effect size X and asymmetry S."""
    shift = mutation_s * mutation_x
    return -mutation_x + shift, mutation_x + shift


class Model_base(Model):
    """Provide the current beta model through the dynamic-model naming contract."""

    @staticmethod
    def description():
        """Return the beta/Gompertz model description as lightweight Markdown."""
        return """# Baseline beta/Gompertz model

## Purpose

Simulates year-by-year evolution of an age-structured population whose inherited
`beta` value controls age-dependent mortality. It is the simplest complete
biological model bundled with ChimpEvo.

## Inheritance

Inherits the generic population lifecycle from `Model`.

## Difference from its parent

- Adds a public `beta` field to every animal.
- Adds stochastic mortality, sexual-maturity-based reproduction, inheritance,
    mutation, carrying capacity, and beta-stabilization stopping.
- Selects mature parents with replacement, so one animal may participate in any
    number of births during the same year.

## Core rules

Annual mortality at age `t` is clamped to a valid probability:

$$m(t, beta) = clamp(alpha * exp(beta * t) + Lambda, 0, 1)$$

Without mutation, offspring inherit the parental mean:

$$beta_child = (beta_parent1 + beta_parent2) / 2$$

With probability `mutation_probability`, an additive shift is sampled:

$$delta_beta ~ Uniform(-X + S*X, X + S*X)$$

and `beta_child = parental_mean + delta_beta`. Each year runs reproduction,
aging, and mortality in that order. Births are limited by mature population,
`fecundity`, and `max_population`.
"""

    @staticmethod
    def add_settings():
        """Declare inherited and beta-model biological settings."""
        return {
            **Model.add_settings(),
            "max_population": {
                "description": "Population carrying capacity", "default": 2000,
                "type": "int", "min": 100, "max": 100000000,
            },
            "initial_population": {
                "description": "Starting population count", "default": 2000,
                "type": "int", "min": 1, "max": 100000000,
            },
            "initial_age_max": {
                "description": "Maximum random initial age", "default": 10,
                "type": "int", "min": 0, "max": 100,
            },
            "lambda": {
                "description": "Background mortality rate", "default": 0.043,
                "type": "float", "min": 0.0, "max": 0.25,
            },
            "alpha": {
                "description": "Age-related mortality multiplier", "default": 0.001,
                "type": "float", "min": 0.0, "max": 0.1,
            },
            "beta_initial": {
                "description": "Initial genetic parameter", "default": 0.11,
                "type": "float", "min": 0.0, "max": 1.0,
            },
            "beta_only_positive": {
                "description": "Prevent inherited and expressed beta values below zero",
                "default": False, "type": "bool",
            },
            "mature_age": {
                "description": "Minimum reproduction age", "default": 12,
                "type": "int", "min": 1, "max": 50,
            },
            "fecundity": {
                "description": "Maximum offspring per mature animal and year", "default": 1.0,
                "type": "float", "min": 0.0, "max": 10.0,
            },
            "mutation_probability": {
                "description": "Mutation probability per birth", "default": 0.1,
                "type": "float", "min": 0.0, "max": 0.5,
            },
            "mutation_x": {
                "description": "Mutation half-width X. The beta shift is sampled from [-X + S*X, X + S*X].",
                "default": 1.0,
                "type": "float", "min": 0.0, "max": 10.0,
            },
            "mutation_s": {
                "description": "Mutation asymmetry S. It moves the interval center by S*X: [-X + S*X, X + S*X].",
                "default": 0.0,
                "type": "float", "min": -1.0, "max": 1.0,
            },
            "stop_beta_change_threshold": {
                "description": "Beta stabilization multiplier (used only for auto-stop)", "default": 0.1,
                "type": "float", "min": 0.0001, "max": 1.0,
            },
            "oldest_death_percent": {
                "description": "Oldest population share used for death-age statistics (for avg_oldest_death_age calculation)", "default": 0.1,
                "type": "float", "min": 0.0001, "max": 1.0,
            },
        }

    @staticmethod
    def add_population_fields():
        """Declare inherited age and public beta population fields."""
        return {
            **Model.add_population_fields(),
            "beta": {"public": True},
        }

    @staticmethod
    def add_values():
        """Declare inherited core values and beta aggregate scalars."""
        return {
            **Model.add_values(),
            "avg_beta": {"title": "Average beta", "annual": True, "final": True, "format": ".4f"},
            "beta_variance": {"title": "Beta variance", "annual": True, "final": True, "format": ".6f"},
            "avg_beta_ema": {"title": "Average beta EMA", "annual": True, "final": True, "format": ".4f"},
            "beta_min": {"title": "Minimum beta", "annual": True, "final": True, "format": ".4f"},
            "beta_max": {"title": "Maximum beta", "annual": True, "final": True, "format": ".4f"},
            "beta_median": {"title": "Median beta", "annual": True, "final": True, "format": ".4f"},
            "avg_oldest_death_age": {
                "title": "Average oldest-subset death age", "annual": True, "final": True, "format": ".4f",
            },
            "avg_years_not_lived": {
                "title": "Average beta-zero lifespan shortfall", "annual": True, "final": True, "format": ".4f",
            },
        }

    @staticmethod
    def add_graphs():
        """Declare inherited age and beta distribution/evolution graphs."""
        return [
            *Model.add_graphs(),
            {
                "filename": "beta_distribution",
                "title": "Beta Distribution",
                "values": ["beta"],
                "labels": ["Beta"],
                "type": "distr",
                "annual": True,
                "final": True,
                "animated": True,
            },
            {
                "filename": "beta_evolution",
                "title": "Beta Evolution",
                "values": ["avg_beta", "avg_beta_ema"],
                "labels": ["Average beta", "Average beta EMA"],
                "type": "time",
                "annual": True,
                "final": True,
                "animated": True,
            },
        ]

    @staticmethod
    def add_metagraphs():
        """Declare final beta evolution across the mutation-effect batch sweep."""
        return [{
            "filename": "beta_by_mutation",
            "title": "Beta by Mutation Effect",
            "xlabel": "Mutation effect size",
            "xvalue": "mutation_x",
            "values": ["avg_beta"],
            "values2": ["beta_variance"],
            "values3": ["avg_age"],
            "labels": ["Average beta"],
            "animated": True,
            "data_label": "tag",
            "style": "points", #could be "lines", "points", "bars"
            "last": True, #default false, if true, the graph will be rendered only for the last row (so it will be a single file)
        },
        {
            "filename": "beta_factors",
            "title": "Beta dependency on lambda, P mutation, mutation_x",
            "xlabel": "Beta factor",
            "xvalue": "avg_beta",
            "values": ["lambda"],
            "values2": ["mutation_probability"],
            "values3": ["mutation_x"],
            "labels": ["Lambda"],
            "animated": True,
            "data_label": "tag",
            "style": "points", #could be "lines", "points", "bars"
            "last": True,
        },
        {
            "filename": "beta_factors lambda bigger than",
            "title": "Beta dependency on lambda, P mutation, mutation_x",
            "xlabel": "Beta factor",
            "xvalue": "avg_beta",
            "values": ["lambda"],
            "values2": ["mutation_probability"],
            "values3": ["mutation_x"],
            "labels": ["Lambda"],
            "animated": True,
            "data_label": "tag",
            "style": "points", #could be "lines", "points", "bars"
            "last": True,
            "filter": {
                "lambda": [0.01, 0.05],
                "mutation_probability": [0.05, 0.15],
            },
            "range": {
                "avg_beta": [0.0, 0.25],
            }
        }
        ]

    @staticmethod
    def add_batch():
        """Declare the default mutation-effect sweep as CSV text."""
        return """tag,mutation_x
x_0.05,0.05
x_0.1,0.1
x_0.2,0.2
x_0.3,0.3
x_0.5,0.5
x_0.75,0.75
x_1.0,1.0
x_1.5,1.5
x_2.0,2.0
"""

    def initialize_population(self):
        """Initialize ages and beta values from model settings."""
        initial_population = int(self.settings["initial_population"])
        initial_age_max = int(self.settings["initial_age_max"])
        beta_initial = float(self.settings["beta_initial"])
        if self.settings.get("beta_only_positive", False) and beta_initial < 0.0:
            raise ValueError("beta_initial must be nonnegative when beta_only_positive is enabled")
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
        self._set_population(torch.stack([ages, betas], dim=1))

    @staticmethod
    def get_estimated_memory_consumption(config):
        """Return a compact peak-memory estimate for the beta population tensor."""
        return int(config.get("max_population", 0)) * 2 * 4 * 2

    def calculate_mortality_probability(self, ages, betas):
        """Return clamped Gompertz mortality probabilities."""
        alpha = self.settings["alpha"]
        lambda_param = self.settings["lambda"]
        mortality = alpha * torch.exp(betas * ages) + lambda_param
        return torch.clamp(mortality, 0.0, 1.0)

    def apply_mortality(self):
        """Remove animals selected by their mortality probabilities."""
        self.last_death = 0
        self.last_oldest_death_age = None
        self.last_years_not_lived = None
        if len(self.population) == 0:
            return self.last_death

        ages = self.population[:, self.population_fields["age"]]
        betas = self.population[:, self.population_fields["beta"]]
        death_probs = self.calculate_mortality_probability(ages, betas)
        survivors = torch.rand_like(death_probs) >= death_probs
        death_mask = ~survivors
        death_count = death_mask.sum().item()
        if death_count:
            death_ages = ages[death_mask]
            oldest_percent = float(self.settings.get("oldest_death_percent", 0.1))
            oldest_count = max(1, math.ceil(len(ages) * oldest_percent))
            oldest_indices = torch.topk(ages, oldest_count).indices
            oldest_mask = torch.zeros_like(survivors, dtype=torch.bool)
            oldest_mask[oldest_indices] = True
            oldest_death_ages = ages[death_mask & oldest_mask]
            if oldest_death_ages.numel():
                self.last_oldest_death_age = oldest_death_ages.mean().item()

            beta_zero_mortality = min(
                1.0,
                max(0.0, float(self.settings["alpha"]) + float(self.settings["lambda"])),
            )
            expected_beta_zero_lifespan = (
                float("inf") if beta_zero_mortality == 0.0 else 1.0 / beta_zero_mortality
            )
            self.last_years_not_lived = (
                expected_beta_zero_lifespan - death_ages.mean().item()
            )
        self.population = self.population[survivors]
        self.last_death = death_count
        return self.last_death

    def mutate_beta(self, parent_beta1, parent_beta2):
        """Return a parental-average beta with an optional mutation shift."""
        base_beta = (parent_beta1 + parent_beta2) / 2.0
        if random.random() >= self.settings["mutation_probability"]:
            return max(0.0, base_beta) if self.settings.get("beta_only_positive", False) else base_beta

        mutation_x = self.settings["mutation_x"]
        mutation_s = self.settings["mutation_s"]
        lower, upper = mutation_interval(mutation_x, mutation_s)
        child_beta = base_beta + random.uniform(lower, upper)
        return max(0.0, child_beta) if self.settings.get("beta_only_positive", False) else child_beta

    def apply_reproduction(self):
        """Create offspring up to fecundity and population limits."""
        self.last_born = 0
        births = 0
        max_population = self.settings["max_population"]
        mature_age = self.settings["mature_age"]
        fecundity = self.settings["fecundity"]
        mature_mask = self.population[:, 0] >= mature_age
        mature_indices = torch.where(mature_mask)[0].cpu().numpy()
        if len(mature_indices) < 2:
            return self.last_born

        current_population = len(self.population)
        max_growth = int(len(mature_indices) * fecundity)
        target_population = min(current_population + max_growth, max_population)
        while len(self.population) < target_population:
            parent_index_1, parent_index_2 = np.random.choice(
                mature_indices,
                size=2,
                replace=True,
            )
            parent_1 = self.population[parent_index_1]
            parent_2 = self.population[parent_index_2]
            child_beta = self.mutate_beta(parent_1[1].item(), parent_2[1].item())
            child = torch.tensor([0.0, child_beta], device=self.device)
            self._append_population_rows(child.unsqueeze(0))
            births += 1

        self.last_born = births
        return self.last_born

    def get_values(self):
        """Return core values extended with current beta aggregates."""
        values = super().get_values()
        beta_values = {
            "avg_beta": None,
            "beta_variance": None,
            "avg_beta_ema": None,
            "beta_min": None,
            "beta_max": None,
            "beta_median": None,
            "avg_oldest_death_age": getattr(self, "last_oldest_death_age", None),
            "avg_years_not_lived": getattr(self, "last_years_not_lived", None),
        }
        if self.get_population_size() > 0:
            betas = self.population[:, self.population_fields["beta"]]
            beta_variance, average_beta = torch.var_mean(betas, correction=0)
            average_beta = average_beta.item()
            average_beta_ema = getattr(self, "avg_beta_ema", None)
            if average_beta_ema is None:
                average_beta_ema = average_beta
            beta_values = {
                "avg_beta": average_beta,
                "beta_variance": beta_variance.item(),
                "avg_beta_ema": average_beta_ema,
                "beta_min": betas.min().item(),
                "beta_max": betas.max().item(),
                "beta_median": betas.median().item(),
                "avg_oldest_death_age": getattr(self, "last_oldest_death_age", None),
                "avg_years_not_lived": getattr(self, "last_years_not_lived", None),
            }
        values.update(beta_values)
        return values

    def should_stop(self):
        """Update beta EMA state and return a model-specific stop reason."""
        population_size = self.get_population_size()
        if population_size < 2:
            return f"population too small ({population_size} animals)"

        betas = self.population[:, self.population_fields["beta"]]
        average_beta = betas.mean().item()
        previous_ema = getattr(self, "avg_beta_ema", None)
        previous_average = getattr(self, "_previous_avg_beta", None)
        if previous_ema is None:
            self.avg_beta_ema = average_beta
            beta_change = 0.0
        else:
            self.avg_beta_ema = 0.03 * average_beta + 0.97 * previous_ema
            beta_change = abs(average_beta - previous_average)

        self._previous_avg_beta = average_beta
        self._beta_changes.append(beta_change)
        if len(self._beta_changes) <= 12:
            return None

        initial_change = float(np.mean(self._beta_changes[:10]))
        threshold = initial_change * self.settings["stop_beta_change_threshold"]
        ema_change = abs(self.avg_beta_ema - previous_ema)
        if ema_change < threshold:
            self._consecutive_ema_below_threshold += 1
            if self._consecutive_ema_below_threshold >= 3:
                return (
                    "beta stabilized "
                    f"(ema_change {ema_change:.6f} < threshold {threshold:.6f} "
                    "for 3 consecutive years)"
                )
        else:
            self._consecutive_ema_below_threshold = 0
        return None
