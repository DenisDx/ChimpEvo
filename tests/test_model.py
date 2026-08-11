import random

import numpy as np
import pytest
import torch

from model_base import Model_base as Model
from settings import DEFAULT_SETTINGS


def make_settings(**overrides):
    """Return isolated model settings with optional overrides."""
    return {**DEFAULT_SETTINGS, **overrides}


@pytest.fixture(autouse=True)
def deterministic_randomness():
    """Reset random generators before each model test."""
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)


@pytest.mark.smoke
def test_initialize_population_contract():
    """Create the expected CPU population tensor and public arrays."""
    model = Model(
        make_settings(initial_population=12, initial_age_max=7, beta_initial=0.25),
        torch.device("cpu"),
    )

    model.initialize_population()

    assert model.population.shape == (12, 2)
    assert model.population.dtype == torch.float32
    assert model.population.device.type == "cpu"
    assert model.get_population_size() == 12
    assert np.all((model.get_ages() >= 0) & (model.get_ages() <= 7))
    np.testing.assert_allclose(model.get_tensor("beta"), 0.25)


@pytest.mark.smoke
def test_age_population_changes_only_age_column():
    """Increment ages while preserving beta values."""
    model = Model(make_settings(), torch.device("cpu"))
    model.population = torch.tensor(
        [[0.0, 0.10], [4.0, -0.20], [9.0, 0.30]],
        dtype=torch.float32,
    )
    original_betas = model.population[:, 1].clone()

    model.age_population()

    torch.testing.assert_close(model.population[:, 0], torch.tensor([1.0, 5.0, 10.0]))
    torch.testing.assert_close(model.population[:, 1], original_betas)


@pytest.mark.smoke
def test_mortality_probability_uses_gompertz_formula_and_clamps():
    """Calculate Gompertz mortality and clamp invalid probabilities."""
    model = Model(make_settings(alpha=0.1, **{"lambda": 0.2}), torch.device("cpu"))
    ages = torch.tensor([0.0, 2.0, 100.0])
    betas = torch.tensor([0.0, 0.1, 1.0])

    probabilities = model.calculate_mortality_probability(ages, betas)

    expected = torch.clamp(0.1 * torch.exp(betas * ages) + 0.2, 0.0, 1.0)
    torch.testing.assert_close(probabilities, expected)
    assert probabilities[-1].item() == 1.0


@pytest.mark.smoke
def test_reproduction_respects_fecundity_and_population_limit():
    """Add only the offspring allowed by fecundity and capacity."""
    settings = make_settings(
        max_population=10,
        mature_age=2,
        fecundity=0.5,
        mutation_probability=0.0,
    )
    model = Model(settings, torch.device("cpu"))
    model.population = torch.tensor(
        [[2.0, 0.10], [3.0, 0.10], [4.0, 0.10], [5.0, 0.10]],
        dtype=torch.float32,
    )

    births = model.apply_reproduction()

    assert births == 2
    assert model.get_population_size() == 6
    torch.testing.assert_close(model.population[-2:, 0], torch.zeros(2))
    torch.testing.assert_close(model.population[-2:, 1], torch.full((2,), 0.10))


@pytest.mark.smoke
def test_empty_population_mortality_is_noop():
    """Return zero deaths for an empty initialized population."""
    model = Model(make_settings(), torch.device("cpu"))
    model.population = torch.empty((0, 2), dtype=torch.float32)

    deaths = model.apply_mortality()

    assert deaths == 0
    assert model.population.shape == (0, 2)