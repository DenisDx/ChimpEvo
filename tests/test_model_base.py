import random

import numpy as np
import pytest
import torch

from model import Model
from model_base import Model_base, mutation_interval
from model_base_fast import Model_base_fast
from model_base_fast_fixed_fecundity import Model_base_fecundity
from model_base_fecundity_m import Model_base_fecundity_m
from settings import DEFAULT_SETTINGS


def make_settings():
    """Return deterministic settings for beta-model compatibility checks."""
    return {
        **DEFAULT_SETTINGS,
        "device": "cpu",
        "initial_population": 12,
        "initial_age_max": 8,
        "mutation_probability": 0.4,
        "max_population": 20,
        "mature_age": 2,
        "fecundity": 0.5,
    }


def run_model_year():
    """Run one deterministic default-model year and return its observable state."""
    random.seed(86420)
    np.random.seed(86420)
    torch.manual_seed(86420)
    model = Model_base(make_settings(), torch.device("cpu"))
    model.initialize_population()
    births = model.apply_reproduction()
    model.age_population()
    deaths = model.apply_mortality()
    return model.population, births, deaths


@pytest.mark.smoke
def test_mutation_interval_scales_asymmetry_by_mutation_x():
    """Center the mutation interval at S times X with half-width X."""
    assert mutation_interval(2.0, 0.5) == pytest.approx((-1.0, 3.0))
    assert mutation_interval(4.0, 0.5) == pytest.approx((-2.0, 6.0))


@pytest.mark.smoke
def test_model_base_uses_dynamic_model_class_contract():
    """Expose the default implementation as a Model subclass."""
    assert issubclass(Model_base, Model)

    model = Model_base(make_settings(), torch.device("cpu"))
    assert model.population_fields == {"age": 0, "beta": 1}
    assert model.population_field_metadata["age"]["public"] is True
    assert model.population_field_metadata["beta"]["public"] is True
    with pytest.raises(TypeError):
        model.population_fields["beta"] = 2


@pytest.mark.smoke
def test_model_base_preserves_current_beta_model_behavior():
    """Preserve the seeded v1 result after one population year."""
    population, births, deaths = run_model_year()

    expected_ages = torch.tensor([
        3, 9, 4, 5, 5, 4, 8, 1, 8, 1, 2, 8, 1, 1, 1, 1,
    ], dtype=torch.float32)
    expected_betas = torch.tensor([
        0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11,
        0.11, 0.11, 0.11, 0.11, 0.11, 0.0676848, 0.2739522, 0.11,
    ])
    assert births == 4
    assert deaths == 0
    torch.testing.assert_close(population[:, 0], expected_ages)
    torch.testing.assert_close(population[:, 1], expected_betas)


@pytest.mark.smoke
def test_model_base_fast_creates_batched_offspring_up_to_capacity():
    """Create one device-side child batch without exceeding population capacity."""
    settings = make_settings()
    settings.update({
        "max_population": 5,
        "mature_age": 2,
        "fecundity": 2.0,
        "mutation_probability": 0.0,
    })
    model = Model_base_fast(settings, torch.device("cpu"))
    model._set_population(torch.tensor([
        [2.0, 0.10],
        [4.0, 0.30],
        [1.0, 0.50],
    ]))

    births = model.apply_reproduction()

    assert births == 2
    assert model.last_born == 2
    assert model.population.shape == (5, 2)
    torch.testing.assert_close(model.population[-2:, 0], torch.zeros(2))
    assert torch.all((model.population[-2:, 1] >= 0.10) & (model.population[-2:, 1] <= 0.30))


@pytest.mark.smoke
def test_model_base_fecundity_limits_each_parent_to_annual_capacity():
    """Limit selected parent slots while creating one batch of children."""
    settings = make_settings()
    settings.update({
        "max_population": 20,
        "mature_age": 2,
        "fecundity": 1.0,
        "mutation_probability": 0.0,
    })
    model = Model_base_fecundity(settings, torch.device("cpu"))
    model._set_population(torch.tensor([
        [2.0, 0.10],
        [2.0, 0.20],
        [2.0, 0.30],
        [2.0, 0.40],
    ]))

    births = model.apply_reproduction()

    assert births == 2
    assert model.last_born == 2
    assert model.population.shape == (6, 2)
    torch.testing.assert_close(model.population[-2:, 0], torch.zeros(2))


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("mutation_s", "expected_beta"),
    [(-1.0, 0.1), (1.0, 0.4)],
)
def test_fecundity_m_applies_requested_multiplicative_mutation(
    monkeypatch,
    mutation_s,
    expected_beta,
):
    """Divide or multiply the parental mean by 1 + X according to S."""
    settings = make_settings()
    settings.update({
        "max_population": 3,
        "mature_age": 2,
        "fecundity": 1.0,
        "mutation_probability": 1.0,
        "mutation_x": 1.0,
        "mutation_s": mutation_s,
    })
    model = Model_base_fecundity_m(settings, torch.device("cpu"))
    model._set_population(torch.tensor([[2.0, 0.1], [2.0, 0.3]]))
    monkeypatch.setattr(
        torch,
        "randperm",
        lambda count, device=None: torch.arange(count, device=device),
    )

    assert model.apply_reproduction() == 1
    assert model.population[-1, 1].item() == pytest.approx(expected_beta)


@pytest.mark.smoke
def test_fecundity_m_suppresses_beta_only_positive():
    """Ignore the inherited positive-beta flag and omit it from model metadata."""
    settings = make_settings()
    settings.update({"beta_initial": -0.2, "beta_only_positive": True})
    model = Model_base_fecundity_m(settings, torch.device("cpu"))

    model.initialize_population()

    assert "beta_only_positive" not in Model_base_fecundity_m.add_settings()
    assert torch.all(model.population[:, 1] == -0.2)


@pytest.mark.smoke
def test_model_base_fecundity_ignores_fractional_parent_capacity():
    """Avoid giving an animal a partial indivisible reproductive slot."""
    settings = make_settings()
    settings.update({"max_population": 20, "mature_age": 2, "fecundity": 0.5})
    model = Model_base_fecundity(settings, torch.device("cpu"))
    model._set_population(torch.tensor([[2.0, 0.10], [2.0, 0.20]]))

    assert model.apply_reproduction() == 0


@pytest.mark.smoke
def test_model_base_declares_and_returns_beta_scalar_values():
    """Return beta aggregates as scalars separate from the beta field."""
    model = Model_base(make_settings(), torch.device("cpu"))
    model._set_population(torch.tensor([
        [2.0, 0.10],
        [4.0, 0.20],
        [6.0, 0.40],
    ]))

    values = model.get_values()

    assert {"avg_beta", "beta_variance", "beta_min", "beta_max", "beta_median"} <= set(model.add_values())
    assert values["avg_beta"] == pytest.approx(0.7 / 3)
    assert values["beta_variance"] == pytest.approx(0.0155555556)
    assert values["beta_min"] == pytest.approx(0.10)
    assert values["beta_max"] == pytest.approx(0.40)
    assert values["beta_median"] == pytest.approx(0.20)
    assert "avg_beta" not in model.population_fields


@pytest.mark.smoke
def test_model_base_returns_none_for_empty_beta_aggregates():
    """Return null beta aggregates when no animals remain."""
    model = Model_base(make_settings(), torch.device("cpu"))
    model._set_population(torch.empty((0, 2)))

    values = model.get_values()

    assert values["avg_beta"] is None
    assert values["beta_variance"] is None
    assert values["beta_min"] is None
    assert values["beta_max"] is None
    assert values["beta_median"] is None


@pytest.mark.smoke
def test_model_base_singleton_beta_variance_is_zero():
    """Use population variance semantics without a singleton NaN."""
    model = Model_base(make_settings(), torch.device("cpu"))
    model._set_population(torch.tensor([[2.0, 0.25]]))

    assert model.get_values()["beta_variance"] == pytest.approx(0.0)


@pytest.mark.smoke
def test_model_base_reports_current_year_mortality_metrics():
    """Report oldest-subset death age and beta-zero lifespan shortfall."""
    settings = make_settings()
    settings.update({"alpha": 1.0, "lambda": 0.0, "oldest_death_percent": 0.1})
    model = Model_base(settings, torch.device("cpu"))
    model._set_population(torch.tensor([
        [1.0, 0.10],
        [5.0, 0.20],
        [10.0, 0.30],
    ]))

    deaths = model.apply_mortality()
    values = model.get_values()

    assert deaths == 3
    assert values["avg_oldest_death_age"] == pytest.approx(10.0)
    assert values["avg_years_not_lived"] == pytest.approx(1.0 - (16.0 / 3.0))


@pytest.mark.smoke
def test_model_base_returns_null_mortality_metrics_without_deaths():
    """Keep mortality metrics null when the current year has no deaths."""
    settings = make_settings()
    settings.update({"alpha": 0.0, "lambda": 0.0})
    model = Model_base(settings, torch.device("cpu"))
    model._set_population(torch.tensor([[4.0, 0.10], [9.0, 0.20]]))

    model.apply_mortality()

    assert model.get_values()["avg_oldest_death_age"] is None
    assert model.get_values()["avg_years_not_lived"] is None


@pytest.mark.smoke
def test_model_base_requests_stop_after_beta_ema_stabilizes():
    """Own beta EMA state and return a reason after sustained stabilization."""
    settings = make_settings()
    settings.update(initial_population=2, initial_age_max=0, beta_initial=0.10)
    model = Model_base(settings, torch.device("cpu"))
    model.initialize_population()

    for beta in torch.linspace(0.10, 0.30, 12):
        model.population[:, model.population_fields["beta"]] = beta
        assert model.should_stop() is None

    reason = None
    for _ in range(60):
        reason = model.should_stop()
        if reason:
            break

    assert "beta stabilized" in reason
    assert model.get_values()["avg_beta_ema"] is not None


@pytest.mark.smoke
def test_model_base_declares_default_mutation_batch_csv():
    """Expose the current mutation sweep as model-owned batch CSV text."""
    lines = Model_base.add_batch().strip().splitlines()

    assert lines[0] == "tag,mutation_x"
    assert lines[1] == "x_0.05,0.05"