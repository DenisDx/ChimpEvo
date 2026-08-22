from pathlib import Path
import json
import subprocess
import sys

import pytest

from experiment_manager import (
    ExperimentManager,
    ExperimentNotSelectedError,
    archive_path,
    resolve_experiment_paths,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_experiment_manager_writes_and_reads_default_conf(tmp_path):
    manager = ExperimentManager(tmp_path)
    manager.set_active_experiment("exp_alpha")

    assert manager.get_active_experiment_name() == "exp_alpha"
    assert manager.get_active_experiment_dir() == tmp_path / "data" / "exp_alpha"
    assert (tmp_path / "default.conf").read_text(encoding="utf-8").strip() == "exp_alpha"


def test_experiment_manager_lists_only_experiment_directories(tmp_path):
    """List experiment directories alphabetically and ignore regular files."""
    (tmp_path / "data" / "exp_beta").mkdir(parents=True)
    (tmp_path / "data" / "exp_alpha").mkdir()
    (tmp_path / "data" / "notes.txt").write_text("ignore", encoding="utf-8")

    manager = ExperimentManager(tmp_path)

    assert manager.list_experiments() == ["exp_alpha", "exp_beta"]


@pytest.mark.parametrize("name", ["", "..", "bad/name", "NUL", "trailing."])
def test_experiment_manager_rejects_nonportable_names(tmp_path, name):
    """Reject experiment names that are unsafe on supported platforms."""
    with pytest.raises(ValueError):
        ExperimentManager(tmp_path).create_experiment(name, {"model": "model_base"})


def test_experiment_manager_creates_complete_experiment_before_selection(tmp_path):
    """Write config and batch data before selecting the new experiment."""
    manager = ExperimentManager(tmp_path)

    experiment_dir = manager.create_experiment(
        "exp_alpha",
        {"model": "model_base", "tag": "default"},
        "tag\nrun_a\n",
    )

    assert json.loads((experiment_dir / "config.json").read_text(encoding="utf-8"))["model"] == "model_base"
    assert (experiment_dir / "multi.csv").read_text(encoding="utf-8") == "tag\nrun_a\n"
    assert manager.get_active_experiment_name() == "exp_alpha"


def test_experiment_manager_deletes_experiment_and_clears_active_selector(tmp_path):
    """Remove the selected experiment and its stale default.conf reference."""
    manager = ExperimentManager(tmp_path)
    experiment_dir = manager.create_experiment("exp_alpha", {"model": "model_base"})
    (experiment_dir / "result").mkdir()

    manager.delete_experiment("exp_alpha")

    assert not experiment_dir.exists()
    assert manager.get_active_experiment_name() is None


def test_archive_path_preserves_existing_directory(tmp_path):
    """Rename prior data to a timestamped backup without deleting its contents."""
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    (result_dir / "result.csv").write_text("saved", encoding="utf-8")

    archived_dir = archive_path(result_dir)

    assert archived_dir is not None
    assert archived_dir.name.startswith("result_") and archived_dir.name.endswith(".bak")
    assert (archived_dir / "result.csv").read_text(encoding="utf-8") == "saved"
    assert not result_dir.exists()


def test_resolve_experiment_paths_rejects_root_config_without_default_conf(tmp_path):
    """Require an active experiment even when a legacy root config exists."""
    config_path = tmp_path / "config.json"
    config_path.write_text('{"model": "model_base"}', encoding="utf-8")

    with pytest.raises(ExperimentNotSelectedError):
        resolve_experiment_paths(tmp_path)


def test_main_cli_rejects_missing_default_conf(tmp_path):
    """Exit with an actionable error when no active experiment is selected."""
    (tmp_path / "config.json").write_text(
        '{"model": "model_base", "tag": "legacy_cli", "device": "cpu", '
        '"max_population": 100, "initial_population": 1, "initial_age_max": 0, '
        '"lambda": 0.0, "alpha": 0.0, "max_iterations": 100}',
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

    assert completed.returncode != 0
    assert "Active experiment is not selected" in completed.stderr
    assert not (tmp_path / "result").exists()
