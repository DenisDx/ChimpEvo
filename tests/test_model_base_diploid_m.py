import pytest
import torch

import model_base_diploid_m as diploid_m_module
from load_model import discover_models, load_model_class
from model_base_diploid import Model_base_diploid
from model_base_diploid_m import Model_base_diploid_m
from settings import DEFAULT_SETTINGS


def make_diploid_m_settings(**overrides):
    """Return complete CPU settings for the diploid multiplicative model."""
    defaults = {
        name: metadata["default"]
        for name, metadata in Model_base_diploid_m.add_settings().items()
    }
    return {
        **DEFAULT_SETTINGS,
        **defaults,
        "device": "cpu",
        "initial_population": 2,
        "initial_age_max": 0,
        **overrides,
    }


def select_first_parent_alleles(monkeypatch):
    """Select parent slots and their first alleles in a deterministic order."""
    monkeypatch.setattr(
        diploid_m_module.torch,
        "randperm",
        lambda count, device=None: torch.arange(count, device=device),
    )
    monkeypatch.setattr(
        diploid_m_module.torch,
        "randint",
        lambda low, high, size, device=None: torch.zeros(
            size,
            dtype=torch.long,
            device=device,
        ),
    )


@pytest.mark.smoke
def test_diploid_m_uses_loader_contract_and_multiplicative_metadata():
    """Discover the model and expose complete X/S/Z multiplier documentation."""
    assert "model_base_diploid_m" in discover_models()
    loaded_class = load_model_class("model_base_diploid_m")

    assert loaded_class.__name__ == "Model_base_diploid_m"
    assert issubclass(loaded_class, Model_base_diploid)
    assert "beta_only_positive" not in loaded_class.add_settings()
    description = loaded_class.description()
    assert "upward_multiplier" in description
    assert "downward_divisor" in description
    assert "P(multiply)" in description


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("mutation_s", "mutation_z", "expected_alleles"),
    [
        (0.5, 1.0, [2.5, 10.0]),
        (-0.5, -1.0, [0.4, 1.6]),
    ],
)
def test_diploid_m_mutates_each_inherited_allele_with_x_s_and_z(
    monkeypatch,
    mutation_s,
    mutation_z,
    expected_alleles,
):
    """Apply asymmetric multiplication or division to both inherited alleles."""
    settings = make_diploid_m_settings(
        max_population=3,
        mature_age=2,
        fecundity=1.0,
        mutation_probability=1.0,
        mutation_x=1.0,
        mutation_s=mutation_s,
        mutation_z=mutation_z,
    )
    model = Model_base_diploid_m(settings, torch.device("cpu"))
    model._set_population(torch.tensor([
        [2.0, 1.0, 1.0, 3.0],
        [2.0, 4.0, 4.0, 8.0],
    ]))
    select_first_parent_alleles(monkeypatch)

    assert model.apply_reproduction() == 1

    child = model.population[-1]
    torch.testing.assert_close(child[2:], torch.tensor(expected_alleles))
    assert child[1].item() == pytest.approx(sum(expected_alleles) / 2.0)


@pytest.mark.smoke
def test_diploid_m_ignores_legacy_positive_beta_option():
    """Allow negative initial alleles even when an old configuration retains the flag."""
    model = Model_base_diploid_m(
        make_diploid_m_settings(beta_initial=-0.2, beta_only_positive=True),
        torch.device("cpu"),
    )

    model.initialize_population()

    assert torch.all(model.population[:, 1:] == -0.2)