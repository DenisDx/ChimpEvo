import pytest
import torch

from load_model import discover_models, load_model_class
from model_base_fast_fixed_fecundity import Model_base_fast_fixed_fecundity
from model_base_z import Model_base_z, ModelBaseZ, sample_z_mutation_shifts
from settings import DEFAULT_SETTINGS


def make_z_settings():
    """Return complete CPU settings for the Z-biased model."""
    declared_defaults = {
        name: metadata["default"]
        for name, metadata in Model_base_z.add_settings().items()
    }
    return {
        **DEFAULT_SETTINGS,
        **declared_defaults,
        "device": "cpu",
    }


@pytest.mark.smoke
def test_model_base_z_uses_loader_contract_and_declares_z_metadata():
    """Discover the loader class, alias, parent, and bounded mutation_z setting."""
    assert "model_base_z" in discover_models()
    loaded_class = load_model_class("model_base_z")
    assert loaded_class.__name__ == "Model_base_z"
    assert loaded_class is not Model_base_z
    assert issubclass(loaded_class, Model_base_fast_fixed_fecundity)
    assert ModelBaseZ is Model_base_z
    assert loaded_class.add_settings()["mutation_z"] == {
        "description": (
            "Mutation sign bias Z. The positive branch probability is "
            "(Z+1)/2; range [-1, 1]."
        ),
        "default": 0.0,
        "type": "float",
        "min": -1.0,
        "max": 1.0,
    }
    assert "description" in loaded_class.__dict__
    assert "When `Z = S`" in loaded_class.description()


@pytest.mark.smoke
def test_z_extremes_select_only_the_requested_mutation_sign():
    """Use only nonnegative shifts at Z=1 and nonpositive shifts at Z=-1."""
    positive = sample_z_mutation_shifts(
        2.0, 0.25, 1.0, 20000, torch.float32, torch.device("cpu"),
    )
    negative = sample_z_mutation_shifts(
        2.0, 0.25, -1.0, 20000, torch.float32, torch.device("cpu"),
    )

    assert torch.all((positive >= 0.0) & (positive <= 2.5))
    assert torch.all((negative >= -1.5) & (negative <= 0.0))


@pytest.mark.smoke
def test_z_controls_positive_branch_probability():
    """Match the positive-shift frequency to (Z+1)/2."""
    torch.manual_seed(3107)
    mutation_z = 0.4
    shifts = sample_z_mutation_shifts(
        1.0, 0.0, mutation_z, 100000, torch.float32, torch.device("cpu"),
    )

    positive_fraction = (shifts > 0.0).float().mean().item()
    assert positive_fraction == pytest.approx((mutation_z + 1.0) / 2.0, abs=0.005)


@pytest.mark.smoke
@pytest.mark.parametrize("mutation_s", [-0.6, 0.0, 0.7])
def test_z_equal_s_matches_parent_uniform_distribution_moments(mutation_s):
    """Match the parent uniform interval mean and variance when Z equals S."""
    torch.manual_seed(90210)
    mutation_x = 1.8
    shifts = sample_z_mutation_shifts(
        mutation_x,
        mutation_s,
        mutation_s,
        200000,
        torch.float64,
        torch.device("cpu"),
    )

    assert shifts.min().item() >= mutation_x * (mutation_s - 1.0)
    assert shifts.max().item() <= mutation_x * (mutation_s + 1.0)
    assert shifts.mean().item() == pytest.approx(mutation_x * mutation_s, abs=0.012)
    assert shifts.var(unbiased=False).item() == pytest.approx(
        mutation_x ** 2 / 3.0,
        abs=0.018,
    )


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("mutation_x", "mutation_s", "mutation_z"),
    [
        (0.0, 0.4, -0.7),
        (2.0, -1.0, 1.0),
        (2.0, 1.0, -1.0),
    ],
)
def test_z_sampler_handles_degenerate_intervals(mutation_x, mutation_s, mutation_z):
    """Return zero shifts when the selected mutation interval collapses."""
    shifts = sample_z_mutation_shifts(
        mutation_x,
        mutation_s,
        mutation_z,
        1000,
        torch.float32,
        torch.device("cpu"),
    )

    torch.testing.assert_close(shifts, torch.zeros_like(shifts))


@pytest.mark.smoke
def test_model_base_z_preserves_fixed_fecundity_and_tensor_properties():
    """Create one child per two parent slots with Z-biased batched mutations."""
    settings = make_z_settings()
    settings.update({
        "max_population": 20,
        "mature_age": 2,
        "fecundity": 1.0,
        "mutation_probability": 1.0,
        "mutation_x": 1.0,
        "mutation_s": 0.0,
        "mutation_z": 1.0,
    })
    model = Model_base_z(settings, torch.device("cpu"))
    model._set_population(torch.tensor([
        [2.0, 0.10],
        [2.0, 0.20],
        [2.0, 0.30],
        [2.0, 0.40],
    ], dtype=torch.float32))

    births = model.apply_reproduction()

    assert births == 2
    assert model.population.shape == (6, 2)
    assert model.population.dtype == torch.float32
    assert model.population.device.type == "cpu"
    torch.testing.assert_close(model.population[-2:, 0], torch.zeros(2))
    assert torch.all(model.population[-2:, 1] >= 0.10)
