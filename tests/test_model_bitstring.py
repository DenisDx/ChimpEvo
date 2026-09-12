"""Focused behavior tests for the bit-string-inspired allele model."""

import pytest
import torch

from model_bitstring import Model_bitstring
from settings import DEFAULT_SETTINGS


def make_settings(**overrides):
    """Return compact CPU settings for the bit-string-inspired model."""
    defaults = {
        name: metadata["default"]
        for name, metadata in Model_bitstring.add_settings().items()
    }
    return {
        **DEFAULT_SETTINGS,
        **defaults,
        "device": "cpu",
        "initial_population": 4,
        "initial_age_max": 0,
        **overrides,
    }


@pytest.mark.smoke
def test_bitstring_model_declares_fixed_dominance_and_delta_schema():
    """Always create dominance/delta fields and expose only the delta mutation mode."""
    settings = Model_bitstring.add_settings()
    fields = Model_bitstring.add_population_fields({"N_alleles": 1})

    assert {"beta_initial", "beta_only_positive", "use_dominance"}.isdisjoint(settings)
    assert settings["use_multiplication"]["default"] is False
    assert settings["beta_central"]["default"] == 2.7
    assert settings["delta_initial"]["default"] == 20.0
    assert list(fields) == ["age", "beta", "beta1_0", "beta2_0", "dom1_0", "dom2_0", "delta1_0", "delta2_0"]


@pytest.mark.smoke
def test_bitstring_model_initializes_central_beta_and_delta():
    """Initialize every independent beta allele and delta from central settings."""
    model = Model_bitstring(make_settings(N_alleles=2, mutation_x=0.0), torch.device("cpu"))
    model.initialize_population()

    torch.testing.assert_close(model._alleles("beta1"), torch.full((4, 2), 2.7))
    torch.testing.assert_close(model._alleles("beta2"), torch.full((4, 2), 2.7))
    torch.testing.assert_close(model._alleles("delta1"), torch.full((4, 2), 20.0))
    assert model.get_values()["avg_beta"] == 0.0


@pytest.mark.smoke
def test_bitstring_model_clamps_inactive_beta_and_selects_dominant_allele():
    """Use the dominant allele and keep its contribution zero before activation."""
    model = Model_bitstring(make_settings(N_alleles=1), torch.device("cpu"))
    model._set_population(torch.tensor([[10.0, 0.0, 2.0, 9.0, 1.0, 0.0, 20.0, 0.0]]))

    model._update_effective_beta()
    assert model.population[0, model.population_fields["beta"]].item() == 0.0

    model.population[0, model.population_fields["delta1_0"]] = 5.0
    model._update_effective_beta()
    assert model.population[0, model.population_fields["beta"]].item() == pytest.approx(1.0)


@pytest.mark.smoke
def test_bitstring_mutation_resets_beta_from_central_value(monkeypatch):
    """Replace a mutated parental beta with a central-beta multiplicative draw."""
    model = Model_bitstring(make_settings(
        N_alleles=1,
        beta_central=2.0,
        mutation_x=1.0,
        mutation_s=0.0,
        mutation_z=1.0,
    ), torch.device("cpu"))
    monkeypatch.setattr(
        torch,
        "rand",
        lambda shape, device=None: torch.zeros(shape, dtype=torch.float32, device=device),
    )
    inherited = torch.full((1, 2, 1), 99.0)
    model._mutate_beta_alleles(inherited, torch.tensor([[[True], [False]]]))

    assert inherited[0, 0, 0].item() == pytest.approx(4.0)
    assert inherited[0, 1, 0].item() == pytest.approx(99.0)


@pytest.mark.smoke
def test_bitstring_multiplies_or_divides_delta_toward_reversion(monkeypatch):
    """Scale mutated deltas with the fixed divisor and reversion-biased direction."""
    model = Model_bitstring(make_settings(
        delta_x=20.0,
        delta_reversion=20.0,
        use_multiplication=True,
    ), torch.device("cpu"))
    random_values = iter([
        torch.full((1, 3), 0.5),
        torch.tensor([[0.0, 1.0, 0.0]]),
    ])
    monkeypatch.setattr(torch, "rand_like", lambda values: next(random_values).to(values.device))
    deltas = torch.tensor([[5.0, 5.0, 5.0]])

    model._mutate_delta_alleles(deltas, torch.tensor([[True, True, False]]))

    assert Model_bitstring.DELTA_X_DIVIDER_FOR_MULTIPLICATION == 10.0
    torch.testing.assert_close(deltas, torch.tensor([[10.0, 2.5, 5.0]]))


@pytest.mark.smoke
def test_bitstring_memory_estimate_always_includes_delta_and_dominance():
    """Account for permanent per-locus dominance and delta attributes."""
    assert Model_bitstring.get_estimated_memory_consumption({"N_alleles": 2, "max_population": 10}) == 10 * 14 * 4 * 2


@pytest.mark.smoke
def test_bitstring_model_inherits_delta_and_age_evolution_graphs():
    """Expose shared delta and age time graphs for permanently delayed alleles."""
    graphs = {graph["filename"]: graph for graph in Model_bitstring.add_graphs()}

    assert graphs["delta_evolution"]["values"] == ["avg_delta", "delta_min", "delta_max"]
    assert graphs["age_evolution"]["values"] == ["avg_age", "avg_oldest_death_age"]