import pytest
import torch

from model_base import Model_base
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