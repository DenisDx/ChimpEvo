import pytest
import torch

from model import Model
from settings import DEFAULT_SETTINGS


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_model_initializes_and_calculates_mortality_on_cuda():
    """Run applicable model contracts on an available CUDA device."""
    device = torch.device("cuda")
    model = Model(DEFAULT_SETTINGS.copy(), device)

    model.initialize_population(8, 5, 0.11)
    probabilities = model.calculate_mortality_probability(
        model.population[:, 0],
        model.population[:, 1],
    )

    assert model.population.device.type == "cuda"
    assert probabilities.device.type == "cuda"
    assert probabilities.shape == (8,)
    assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))