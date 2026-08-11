import random

import numpy as np
import pytest
import torch

from main import PopulationSimulation
from settings import DEFAULT_SETTINGS


def make_settings(tag):
    """Return settings with the same deterministic seed."""
    return {
        **DEFAULT_SETTINGS,
        "tag": tag,
        "device": "cpu",
        "initial_population": 16,
        "initial_age_max": 100,
        "seed": 24680,
    }


def capture_seeded_state(simulation):
    """Capture population and values from all seeded generators."""
    return (
        simulation.model.population.clone(),
        random.random(),
        np.random.random(),
        torch.rand(4),
    )


@pytest.mark.smoke
def test_same_seed_reproduces_initial_state_and_random_generators(tmp_path, monkeypatch):
    """Reset Python, NumPy, and Torch for every independent simulation."""
    monkeypatch.chdir(tmp_path)

    first = PopulationSimulation(make_settings("seed_first"))
    first_state = capture_seeded_state(first)
    second = PopulationSimulation(make_settings("seed_second"))
    second_state = capture_seeded_state(second)

    torch.testing.assert_close(first_state[0], second_state[0])
    assert first_state[1] == second_state[1]
    assert first_state[2] == second_state[2]
    torch.testing.assert_close(first_state[3], second_state[3])


@pytest.mark.smoke
def test_zero_seed_keeps_random_generator_state(tmp_path, monkeypatch):
    """Treat zero as random mode without resetting application generators."""
    monkeypatch.chdir(tmp_path)
    settings = {**make_settings("seed_zero"), "seed": 0}
    random.seed(123)
    expected = random.Random(123)

    PopulationSimulation(settings)

    assert random.random() == expected.random()