import pytest
import torch

from model_alleles import Model_alleles
from settings import DEFAULT_SETTINGS


def make_settings(**overrides):
    """Return compact CPU settings for the multi-locus beta model."""
    defaults = {
        name: metadata["default"]
        for name, metadata in Model_alleles.add_settings().items()
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
def test_allele_model_declares_dynamic_private_schema():
    """Create only the optional field sets requested by the active configuration."""
    model = Model_alleles(make_settings(N_alleles=2, use_dominance=True, delta_x=1.0), torch.device("cpu"))

    assert list(model.population_fields) == [
        "age", "beta", "beta1_0", "beta1_1", "beta2_0", "beta2_1",
        "dom1_0", "dom1_1", "dom2_0", "dom2_1",
        "delta1_0", "delta1_1", "delta2_0", "delta2_1",
    ]
    assert all(not model.population_field_metadata[name]["public"] for name in list(model.population_fields)[2:])


@pytest.mark.smoke
def test_allele_model_initializes_homozygous_loci_and_zero_optional_values():
    """Set initial beta loci homogeneously with zero dominance and delta values."""
    model = Model_alleles(make_settings(N_alleles=2, use_dominance=True, delta_x=1.0, beta_initial=0.25), torch.device("cpu"))
    model.initialize_population()

    torch.testing.assert_close(model._alleles("beta1"), torch.full((4, 2), 0.25))
    torch.testing.assert_close(model._alleles("beta2"), torch.full((4, 2), 0.25))
    torch.testing.assert_close(model._alleles("dom1"), torch.zeros((4, 2)))
    torch.testing.assert_close(model._alleles("delta1"), torch.zeros((4, 2)))
    torch.testing.assert_close(model.population[:, model.population_fields["beta"]], torch.full((4,), 0.25))


@pytest.mark.smoke
def test_allele_model_uses_second_allele_for_equal_dominance():
    """Resolve equal dominance with strict greater-than by selecting beta2."""
    model = Model_alleles(make_settings(N_alleles=2, use_dominance=True), torch.device("cpu"))
    model._set_population(torch.tensor([[2.0, 0.0, 1.0, 2.0, 10.0, 20.0, 0.0, 0.0, 0.0, 0.0]]))
    model._update_effective_beta()

    assert model.population[0, model.population_fields["beta"]].item() == pytest.approx(15.0)


@pytest.mark.smoke
def test_allele_model_clamps_individual_beta_and_delta_contributions():
    """Keep inherited alleles and pre-delta effective contributions nonnegative."""
    model = Model_alleles(make_settings(N_alleles=1, delta_x=2.0, beta_only_positive=True), torch.device("cpu"))
    model._set_population(torch.tensor([[1.0, 0.0, 0.5, 0.5, 2.0, 2.0]]))
    model._update_effective_beta()

    assert model.population[0, model.population_fields["beta"]].item() == 0.0
    with pytest.raises(ValueError, match="beta_initial"):
        Model_alleles(make_settings(beta_only_positive=True, beta_initial=-0.1), torch.device("cpu")).initialize_population()


@pytest.mark.smoke
def test_allele_model_statistics_stay_scalar_and_memory_estimate_includes_options():
    """Compute declared allele aggregates and scale memory by active field count."""
    settings = make_settings(N_alleles=2, use_dominance=True, delta_x=1.0, max_population=10)
    model = Model_alleles(settings, torch.device("cpu"))
    model.initialize_population()
    values = model.get_values()

    assert values["avg_allele_beta_variance"] == 0.0
    assert values["avg_dominant_beta_variance"] == 0.0
    assert values["avg_delta"] == 0.0
    assert Model_alleles.get_estimated_memory_consumption(settings) == 10 * 14 * 4 * 2