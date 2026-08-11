import csv
import json

import pytest

from batch import run_batch
from settings import DEFAULT_SETTINGS


def make_batch_config():
    """Return settings that finish each batch row after one year."""
    return {
        **DEFAULT_SETTINGS,
        "device": "cpu",
        "seed": 97531,
        "max_population": 100,
        "initial_population": 1,
        "initial_age_max": 0,
        "alpha": 0.0,
        "lambda": 0.0,
        "mature_age": 12,
        "stat_generation_period": 1,
        "graph_generation_period": 1000,
        "max_iterations": 100,
    }


@pytest.mark.smoke
def test_batch_runs_each_csv_tag_and_can_repeat(tmp_path, monkeypatch):
    """Execute and repeat a short v1 batch for both tagged result folders."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text(
        json.dumps(make_batch_config()),
        encoding="utf-8",
    )
    with (tmp_path / "multi.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["tag", "beta_initial"])
        writer.writeheader()
        writer.writerows([
            {"tag": "batch_a", "beta_initial": "0.10"},
            {"tag": "batch_b", "beta_initial": "0.20"},
        ])

    result_dirs = run_batch("multi.csv", "config.json")
    repeated_result_dirs = run_batch("multi.csv", "config.json")

    assert result_dirs == ["result/batch_a", "result/batch_b"]
    assert repeated_result_dirs == result_dirs
    for tag in ("batch_a", "batch_b"):
        assert (tmp_path / "result" / tag / "result.csv").is_file()
        assert (tmp_path / "result" / tag / "results_summary.png").is_file()