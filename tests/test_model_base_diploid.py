import pytest
import torch

import model_base_diploid as diploid_module
from load_model import discover_models, load_model_class
from model_base import Model_base
from model_base_diploid import Model_base_diploid
from model_base_z import Model_base_z
from settings import DEFAULT_SETTINGS


def make_diploid_settings():
    """Return complete CPU settings for the diploid beta model."""
    declared_defaults = {
        name: metadata["default"]
        for name, metadata in Model_base_diploid.add_settings().items()
    }
    return {
        **DEFAULT_SETTINGS,
        **declared_defaults,
        "device": "cpu",
        "initial_population": 6,
        "initial_age_max": 4,
    }


def set_deterministic_parent_selection(monkeypatch, allele_choices):
    """Select parent slots in order and return specified allele indices."""
    monkeypatch.setattr(
        diploid_module.torch,
        "randperm",
        lambda count, device=None: torch.arange(count, device=device),
    )
    monkeypatch.setattr(
        diploid_module.torch,
        "randint",
        lambda low, high, size, device=None: torch.tensor(
            allele_choices,
            dtype=torch.long,
            device=device,
        ),
    )


@pytest.mark.smoke
def test_model_base_diploid_uses_loader_contract_and_no_new_settings():
    """Discover the loader class and preserve every inherited setting unchanged."""
    assert "model_base_diploid" in discover_models()
    loaded_class = load_model_class("model_base_diploid")
    assert loaded_class.__name__ == "Model_base_diploid"
    assert issubclass(loaded_class, Model_base_z)
    assert loaded_class.add_settings() == Model_base_z.add_settings()
    assert "description" in loaded_class.__dict__
    description = loaded_class.description()
    assert "codominant" in description
    assert "independently twice per child" in description
    assert "Mortality is not overridden" in description


@pytest.mark.smoke
def test_diploid_schema_keeps_beta_in_compatible_column_one():
    """Keep public beta at column one and store both alleles privately after it."""
    model = Model_base_diploid(make_diploid_settings(), torch.device("cpu"))

    assert model.population_fields == {
        "age": 0,
        "beta": 1,
        "beta1": 2,
        "beta2": 3,
    }
    assert model.population_field_metadata["beta"]["public"] is True
    assert model.population_field_metadata["beta1"]["public"] is False
    assert model.population_field_metadata["beta2"]["public"] is False


@pytest.mark.smoke
def test_diploid_initialization_is_homozygous_with_mean_phenotype():
    """Initialize beta1, beta2, and beta to beta_initial for every animal."""
    settings = make_diploid_settings()
    settings["beta_initial"] = 0.37
    model = Model_base_diploid(settings, torch.device("cpu"))

    model.initialize_population()

    assert model.population.shape == (6, 4)
    expected = torch.full((6,), 0.37, dtype=torch.float32)
    torch.testing.assert_close(model.population[:, 1], expected)
    torch.testing.assert_close(model.population[:, 2], expected)
    torch.testing.assert_close(model.population[:, 3], expected)


@pytest.mark.smoke
def test_diploid_reproduction_selects_one_allele_from_each_parent(monkeypatch):
    """Inherit independently selected parental alleles and store their arithmetic mean."""
    settings = make_diploid_settings()
    settings.update({
        "max_population": 20,
        "mature_age": 2,
        "fecundity": 1.0,
        "mutation_probability": 0.0,
    })
    model = Model_base_diploid(settings, torch.device("cpu"))
    model._set_population(torch.tensor([
        [2.0, 2.0, 1.0, 3.0],
        [2.0, 15.0, 10.0, 20.0],
        [2.0, 150.0, 100.0, 200.0],
        [2.0, 1500.0, 1000.0, 2000.0],
    ]))
    set_deterministic_parent_selection(monkeypatch, [[1, 0], [0, 1]])

    births = model.apply_reproduction()

    assert births == 2
    torch.testing.assert_close(model.population[-2:, 2:], torch.tensor([
        [3.0, 10.0],
        [100.0, 2000.0],
    ]))
    torch.testing.assert_close(model.population[-2:, 1], torch.tensor([6.5, 1050.0]))


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("mutation_draws", "expected_alleles"),
    [
        ([[0.9, 0.9]], [1.0, 10.0]),
        ([[0.1, 0.9]], [3.0, 10.0]),
        ([[0.9, 0.1]], [1.0, 14.0]),
        ([[0.1, 0.1]], [3.0, 14.0]),
    ],
)
def test_diploid_checks_and_shifts_each_allele_independently(
    monkeypatch,
    mutation_draws,
    expected_alleles,
):
    """Support zero, either one, or both independently mutated child alleles."""
    settings = make_diploid_settings()
    settings.update({
        "max_population": 10,
        "mature_age": 2,
        "fecundity": 1.0,
        "mutation_probability": 0.5,
    })
    model = Model_base_diploid(settings, torch.device("cpu"))
    model._set_population(torch.tensor([
        [2.0, 1.5, 1.0, 2.0],
        [2.0, 15.0, 10.0, 20.0],
    ]))
    set_deterministic_parent_selection(monkeypatch, [[0, 0]])
    monkeypatch.setattr(
        diploid_module.torch,
        "rand",
        lambda size, device=None: torch.tensor(
            mutation_draws,
            dtype=torch.float32,
            device=device,
        ),
    )
    monkeypatch.setattr(
        diploid_module,
        "sample_z_mutation_shifts",
        lambda *args, **kwargs: torch.tensor([2.0, 4.0]),
    )

    assert model.apply_reproduction() == 1

    child = model.population[-1]
    torch.testing.assert_close(child[2:], torch.tensor(expected_alleles))
    assert child[1].item() == pytest.approx(sum(expected_alleles) / 2.0)


@pytest.mark.smoke
def test_diploid_inherits_mortality_using_stored_beta_phenotype():
    """Use inherited mortality and aggregates from beta regardless of allele decomposition."""
    settings = make_diploid_settings()
    model = Model_base_diploid(settings, torch.device("cpu"))
    model._set_population(torch.tensor([
        [5.0, 0.20, -0.60, 1.00],
        [5.0, 0.20, 0.10, 0.30],
    ]))

    probabilities = model.calculate_mortality_probability(
        model.population[:, model.population_fields["age"]],
        model.population[:, model.population_fields["beta"]],
    )

    assert "apply_mortality" not in Model_base_diploid.__dict__
    assert Model_base_diploid.apply_mortality is Model_base.apply_mortality
    torch.testing.assert_close(probabilities[0], probabilities[1])
    assert model.get_values()["avg_beta"] == pytest.approx(0.20)


@pytest.mark.smoke
def test_diploid_preserves_fixed_fecundity_dtype_device_and_beta_invariant():
    """Keep parent capacity and beta-mean consistency in batched reproduction."""
    settings = make_diploid_settings()
    settings.update({
        "max_population": 20,
        "mature_age": 2,
        "fecundity": 1.0,
        "mutation_probability": 1.0,
        "mutation_x": 1.0,
        "mutation_s": 0.0,
        "mutation_z": 1.0,
    })
    model = Model_base_diploid(settings, torch.device("cpu"))
    model._set_population(torch.tensor([
        [2.0, 0.15, 0.10, 0.20],
        [2.0, 0.35, 0.30, 0.40],
        [2.0, 0.55, 0.50, 0.60],
        [2.0, 0.75, 0.70, 0.80],
    ], dtype=torch.float32))

    births = model.apply_reproduction()

    assert births == 2
    assert model.population.shape == (6, 4)
    assert model.population.dtype == torch.float32
    assert model.population.device.type == "cpu"
    torch.testing.assert_close(model.population[-2:, 0], torch.zeros(2))
    torch.testing.assert_close(
        model.population[-2:, 1],
        model.population[-2:, 2:].mean(dim=1),
    )
