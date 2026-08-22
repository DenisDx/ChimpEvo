from unittest.mock import Mock

import pytest

import main as main_module
from main import PopulationSimulation, run_simulation
from model import Model
from model_base import Model_base
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
def test_runtime_config_rejects_missing_model_setting_before_output(tmp_path, monkeypatch):
    """Reject incomplete model configuration before creating result files."""
    monkeypatch.chdir(tmp_path)
    settings = make_settings()
    settings.pop("beta_initial")

    with pytest.raises(ValueError, match="beta_initial"):
        PopulationSimulation(settings)

    assert not (tmp_path / "result").exists()


@pytest.mark.smoke
def test_runtime_config_rejects_out_of_range_setting_before_output(tmp_path, monkeypatch):
    """Reject invalid ranges instead of clamping runtime configuration."""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="mutation_probability"):
        PopulationSimulation(make_settings(mutation_probability=2.0))

    assert not (tmp_path / "result").exists()


@pytest.mark.smoke
@pytest.mark.parametrize("tag", ["../escape", "nested/tag", "_models"])
def test_runtime_config_rejects_unsafe_result_tags(tmp_path, monkeypatch, tag):
    """Prevent run tags from escaping or colliding with the result namespace."""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="tag"):
        PopulationSimulation(make_settings(tag=tag))

    assert not (tmp_path / "result").exists()


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
        "beta_min": pytest.approx(0.11),
        "beta_max": pytest.approx(0.11),
        "beta_median": pytest.approx(0.11),
        "avg_oldest_death_age": None,
        "avg_years_not_lived": None,
    }
    simulation._generate_year_graphs.assert_called_once_with(0)


@pytest.mark.smoke
def test_run_reports_each_generated_graph_frame(tmp_path, monkeypatch):
    """Report the year whenever a simulation generates a graph frame."""
    monkeypatch.chdir(tmp_path)
    simulation = PopulationSimulation(make_settings())
    simulation._generate_year_graphs = Mock()
    generated_frames = []

    simulation.run(
        should_cancel=lambda: simulation.year >= 1,
        graph_callback=lambda output_dir, year: generated_frames.append((output_dir, year)),
    )

    assert generated_frames == [(simulation.output_dir, 0)]


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


@pytest.mark.smoke
def test_simulation_loads_configured_dynamic_model(tmp_path, monkeypatch):
    """Construct the model class returned for the configured module name."""

    class Model_custom(Model_base):
        """Identify the selected model class without changing behavior."""

    loader = Mock(return_value=Model_custom)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main_module, "load_model_class", loader)

    simulation = PopulationSimulation(make_settings(model="model_custom"))

    loader.assert_called_once_with("model_custom")
    assert isinstance(simulation.model, Model_custom)


@pytest.mark.smoke
def test_cancelled_run_omits_success_final_csv(tmp_path, monkeypatch):
    """Keep a cancelled run out of successful final-output contracts."""
    monkeypatch.chdir(tmp_path)

    _, completed = run_simulation(
        make_settings(tag="cancelled"),
        should_cancel=lambda: True,
        return_completion=True,
    )

    assert completed is False
    assert not (tmp_path / "result" / "cancelled" / "final.csv").exists()


@pytest.mark.smoke
def test_finalize_request_uses_successful_completion_contract(tmp_path, monkeypatch):
    """Export final results when an external request finalizes the current year."""
    monkeypatch.chdir(tmp_path)

    results, completed = run_simulation(
        make_settings(tag="finalized", stat_generation_period=100),
        should_finalize=lambda: True,
        return_completion=True,
    )

    assert completed is True
    assert len(results) == 1
    assert (tmp_path / "result" / "finalized" / "final.csv").exists()


@pytest.mark.smoke
def test_yearly_statistics_use_model_scalar_values(tmp_path, monkeypatch):
    """Build legacy result columns from the model scalar namespace."""
    monkeypatch.chdir(tmp_path)
    simulation = PopulationSimulation(make_settings())
    simulation.model.get_values = Mock(return_value={
        "count": 7,
        "avg_age": 12.5,
        "born": 3,
        "dead": 2,
        "prop_aging": 0.25,
        "avg_beta": 0.42,
        "avg_beta_ema": 0.41,
        "beta_min": 0.20,
        "beta_max": 0.60,
        "beta_median": 0.40,
        "avg_oldest_death_age": None,
        "avg_years_not_lived": None,
    })

    stats = simulation._calculate_yearly_stats()

    simulation.model.get_values.assert_called_once_with()
    assert stats == {
        "year": 0,
        "count": 7,
        "avg_age": 12.5,
        "born": 3,
        "dead": 2,
        "prop_aging": 0.25,
        "avg_beta": 0.42,
        "avg_beta_ema": 0.41,
        "beta_min": 0.20,
        "beta_max": 0.60,
        "beta_median": 0.40,
        "avg_oldest_death_age": None,
        "avg_years_not_lived": None,
    }


@pytest.mark.smoke
def test_model_stop_reason_is_checked_between_statistics_periods(tmp_path, monkeypatch):
    """Collect the exact final state when a model stops on an unsampled year."""
    monkeypatch.chdir(tmp_path)
    simulation = PopulationSimulation(make_settings(stat_generation_period=10))
    simulation.year = 1
    simulation._generate_year_graphs = Mock()
    simulation.model.should_stop = Mock(return_value="custom stop")

    has_next = simulation.step()

    assert has_next is False
    assert simulation.year == 2
    assert len(simulation.results) == 1
    assert simulation.results[0]["year"] == 1
    simulation.model.should_stop.assert_called_once_with()


@pytest.mark.smoke
def test_empty_model_subclass_runs_without_overrides(tmp_path, monkeypatch):
    """Run one core-managed year with an otherwise empty dynamic model."""

    class Model_empty(Model):
        """Use every inherited dynamic-model default."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main_module, "load_model_class", Mock(return_value=Model_empty))
    simulation = PopulationSimulation(make_settings(model="model_empty"))
    simulation._generate_year_graphs = Mock()

    has_next = simulation.step()

    assert has_next is True
    assert simulation.model.population_fields == {"age": 0}
    assert simulation.model.population.shape == (4, 1)
    assert simulation.results[0]["count"] == 4


@pytest.mark.smoke
def test_age_only_external_model_completes_without_beta_outputs(tmp_path, monkeypatch):
    """Run a trusted age-only model through the full simulation lifecycle."""

    class Model_age_only(Model):
        """Use the inherited no-beta population lifecycle."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main_module, "load_model_class", Mock(return_value=Model_age_only))

    _, completed = run_simulation(
        make_settings(model="model_age_only", tag="age_only", initial_population=1),
        return_completion=True,
    )

    result_dir = tmp_path / "result" / "age_only"
    assert completed is True
    assert (result_dir / "result.csv").is_file()
    assert (result_dir / "final.csv").is_file()
    assert (result_dir / "age_distribution.png").is_file()
    assert not (result_dir / "betaoccurrence0.png").exists()