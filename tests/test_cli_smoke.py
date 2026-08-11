import json
from pathlib import Path
import subprocess
import sys

import pytest

from settings import DEFAULT_SETTINGS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_cli_config():
    """Return a one-year CLI smoke configuration."""
    return {
        **DEFAULT_SETTINGS,
        "tag": "cli_smoke",
        "device": "cpu",
        "seed": 13579,
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
def test_main_cli_completes_with_project_interpreter(tmp_path):
    """Run the CLI with the pytest virtual-environment interpreter."""
    (tmp_path / "config.json").write_text(
        json.dumps(make_cli_config()),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "main.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Simulation complete" in completed.stdout
    assert (tmp_path / "result" / "cli_smoke" / "result.csv").is_file()
    assert (tmp_path / "result" / "cli_smoke" / "results_summary.png").is_file()