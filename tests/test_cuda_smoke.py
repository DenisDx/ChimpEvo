import pytest
import torch

from model_base import Model_base
from model_base_z import Model_base_z
from settings import DEFAULT_SETTINGS


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_model_initializes_and_calculates_mortality_on_cuda():
    """Run applicable model contracts on an available CUDA device."""
    device = torch.device("cuda")
    settings = {
        **DEFAULT_SETTINGS,
        "initial_population": 8,
        "initial_age_max": 5,
        "beta_initial": 0.11,
    }
    model = Model_base(settings, device)

    model.initialize_population()
    probabilities = model.calculate_mortality_probability(
        model.population[:, 0],
        model.population[:, 1],
    )
    beta_bins = model.bin_values("beta", 4)

    assert model.population.device.type == "cuda"
    assert probabilities.device.type == "cuda"
    assert probabilities.shape == (8,)
    assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert sum(beta_bins["data"]) + beta_bins["below_min"] + beta_bins["above_max"] == 8
    assert all(isinstance(count, int) for count in beta_bins["data"])

    z_settings = {
        **settings,
        "max_population": 8,
        "mature_age": 2,
        "fecundity": 1.0,
        "mutation_probability": 1.0,
        "mutation_x": 1.0,
        "mutation_s": 0.0,
        "mutation_z": 1.0,
    }
    z_model = Model_base_z(z_settings, device)
    z_model._set_population(torch.tensor([
        [2.0, 0.10],
        [2.0, 0.20],
        [2.0, 0.30],
        [2.0, 0.40],
    ], dtype=torch.float32, device=device))

    assert z_model.apply_reproduction() == 2
    assert z_model.population.device.type == "cuda"
    assert z_model.population.shape == (6, 2)