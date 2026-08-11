from unittest.mock import Mock

import pytest

from main import PopulationSimulation
from settings import DEFAULT_SETTINGS


def make_settings(tag="smoke", **overrides):
    """Return fast deterministic settings for a simulation smoke test."""
    settings = {
        **DEFAULT_SETTINGS,
        "tag": tag,
        "device": "cpu",
        "max_population": 100,
        "initial_population": 4,
        "initial_age_max": 0,
        "alpha": 0.0,
        "lambda": 0.0,
        "mature_age": 12,
        "stat_generation_period": 1,
        "graph_generation_period": 1000,
        "max_iterations": 100,
    }
    settings.update(overrides)
    return settings


@pytest.mark.smoke
def test_step_collects_current_v1_statistics(tmp_path, monkeypatch):
    """Execute one year and collect the current v1 result contract."""
    monkeypatch.chdir(tmp_path)
    simulation = PopulationSimulation(make_settings())
    simulation._generate_year_graphs = Mock()

    has_next = simulation.step()

    assert has_next is True
    assert simulation.year == 1
    assert len(simulation.results) == 1
    assert simulation.results[0] == {
        "year": 0,
        "count": 4,
        "avg_age": 1.0,
        "born": 0,
        "dead": 0,
        "prop_aging": 0.0,
        "avg_beta": pytest.approx(0.11),
        "avg_beta_ema": pytest.approx(0.11),
    }
    simulation._generate_year_graphs.assert_called_once_with(0)


@pytest.mark.smoke
def test_population_below_two_stops_after_completed_year(tmp_path, monkeypatch):
    """Stop the current v1 simulation when fewer than two animals remain."""
    monkeypatch.chdir(tmp_path)
    simulation = PopulationSimulation(make_settings(initial_population=1))
    simulation._generate_year_graphs = Mock()

    has_next = simulation.step()

    assert has_next is False
    assert simulation.year == 1
    assert simulation.results[-1]["count"] == 1