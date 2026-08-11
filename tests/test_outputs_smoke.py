import csv

import pytest
import torch
from PIL import Image

from main import PopulationSimulation
from settings import DEFAULT_SETTINGS


def make_settings(tag="output_smoke"):
    """Return small CPU settings for output smoke tests."""
    return {
        **DEFAULT_SETTINGS,
        "tag": tag,
        "device": "cpu",
        "max_population": 100,
        "initial_population": 2,
        "initial_age_max": 0,
    }


def set_single_result(simulation):
    """Set one valid result row for export tests."""
    simulation.year = 1
    simulation.results = [{
        "year": 0,
        "count": 2,
        "avg_age": 1.0,
        "born": 0,
        "dead": 0,
        "prop_aging": 0.0,
        "avg_beta": 0.11,
        "avg_beta_ema": 0.11,
    }]


@pytest.mark.smoke
def test_export_writes_readable_csv_graphs_and_gifs(tmp_path, monkeypatch):
    """Export current v1 CSV, PNG, and GIF artifacts in isolation."""
    monkeypatch.chdir(tmp_path)
    simulation = PopulationSimulation(make_settings())
    simulation.model.population = torch.tensor(
        [[1.0, 0.11], [3.0, 0.12]],
        dtype=torch.float32,
    )
    set_single_result(simulation)
    simulation._generate_year_graphs(0)

    output_dir = simulation.export_results()

    result_dir = tmp_path / output_dir
    with (result_dir / "result.csv").open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 1
    assert rows[0]["year"] == "0"
    assert rows[0]["count"] == "2"

    expected_images = [
        "distribution0.png",
        "survivorship0.png",
        "betaoccurrence0.png",
        "results_summary.png",
        "distribution.gif",
        "survivorship.gif",
        "betaoccurrence.gif",
    ]
    for file_name in expected_images:
        image_path = result_dir / file_name
        assert image_path.stat().st_size > 0
        with Image.open(image_path) as image:
            image.verify()


@pytest.mark.smoke
@pytest.mark.xfail(
    strict=True,
    reason="v1 does not create the mandatory successful-run final.csv required by v2",
)
def test_successful_export_always_writes_final_csv(tmp_path, monkeypatch):
    """Require the metadata-only final file introduced by v2."""
    monkeypatch.chdir(tmp_path)
    simulation = PopulationSimulation(make_settings(tag="final_smoke"))
    set_single_result(simulation)

    output_dir = simulation.export_results()

    assert (tmp_path / output_dir / "final.csv").is_file()