"""Experiment state management for project-local run configuration."""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path


class ExperimentNotSelectedError(RuntimeError):
    """Raised when no active experiment is selected for this project."""


def validate_path_component(name, label="Name"):
    """Reject unsafe or non-portable single path components."""
    reserved_names = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{number}" for number in range(1, 10)),
        *(f"LPT{number}" for number in range(1, 10)),
    }
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or name[-1] in {" ", "."}
        or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
        or name.split(".", 1)[0].upper() in reserved_names
    ):
        raise ValueError(f"{label} is not a valid portable path name")
    return name


def atomic_write_text(path, text):
    """Replace one text file atomically after a complete adjacent write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary_path.write_text(text, encoding="utf-8")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


class ExperimentManager:
    """Manage the active experiment and the local data directory layout."""

    def __init__(self, project_root=".", data_dir_name="data"):
        """Store the project root and experiment data directory name."""
        self.project_root = Path(project_root).resolve()
        self.data_dir_name = data_dir_name

    def _default_conf_path(self):
        """Return the active-experiment selector path."""
        return self.project_root / "default.conf"

    def get_active_experiment_name(self):
        """Return current experiment name from default.conf or None."""
        default_conf = self._default_conf_path()
        if not default_conf.exists():
            return None
        name = default_conf.read_text(encoding="utf-8").strip()
        return name or None

    def get_active_experiment_dir(self):
        """Return the directory for the active experiment or None."""
        name = self.get_active_experiment_name()
        if not name:
            return None
        return self.project_root / self.data_dir_name / name

    def list_experiments(self):
        """Return available experiment directory names in stable order."""
        data_dir = self.project_root / self.data_dir_name
        if not data_dir.exists():
            return []
        return sorted(child.name for child in data_dir.iterdir() if child.is_dir())

    def set_active_experiment(self, experiment_name):
        """Persist the active experiment name to default.conf."""
        self.validate_experiment_name(experiment_name)

        experiment_dir = self.project_root / self.data_dir_name / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self._default_conf_path(), experiment_name)
        return experiment_dir

    def validate_experiment_name(self, experiment_name):
        """Reject unsafe or non-portable experiment directory names."""
        validate_path_component(experiment_name, "Experiment name")

    def create_experiment(self, experiment_name, config, batch_text=None, activate=True):
        """Create one complete experiment and optionally select it after all writes succeed."""
        self.validate_experiment_name(experiment_name)
        experiment_dir = self.project_root / self.data_dir_name / experiment_name
        if experiment_dir.exists():
            raise ValueError(f"Experiment already exists: {experiment_name}")
        try:
            experiment_dir.mkdir(parents=True)
            atomic_write_text(experiment_dir / "config.json", json.dumps(config, indent=2))
            if batch_text:
                atomic_write_text(experiment_dir / "multi.csv", batch_text.rstrip() + "\n")
            if activate:
                self.set_active_experiment(experiment_name)
        except Exception:
            if experiment_dir.exists():
                shutil.rmtree(experiment_dir)
            raise
        return experiment_dir

    def clone_experiment(
        self,
        source_name,
        target_name,
        copy_results=False,
        activate=True,
    ):
        """Clone saved experiment files and optionally current results byte for byte."""
        self.validate_experiment_name(source_name)
        self.validate_experiment_name(target_name)
        data_dir = self.project_root / self.data_dir_name
        source_dir = data_dir / source_name
        target_dir = data_dir / target_name
        source_config = source_dir / "config.json"
        if not source_dir.is_dir() or not source_config.is_file():
            raise ValueError(f"Source experiment is invalid: {source_name}")
        if target_dir.exists():
            raise ValueError(f"Experiment already exists: {target_name}")

        try:
            target_dir.mkdir(parents=True)
            shutil.copy2(source_config, target_dir / "config.json")
            source_batch = source_dir / "multi.csv"
            if source_batch.is_file():
                shutil.copy2(source_batch, target_dir / "multi.csv")
            source_results = source_dir / "result"
            if copy_results and source_results.is_dir():
                shutil.copytree(source_results, target_dir / "result")
            if activate:
                self.set_active_experiment(target_name)
        except Exception:
            if target_dir.exists():
                shutil.rmtree(target_dir)
            raise
        return target_dir

    def delete_experiment(self, experiment_name):
        """Delete one existing experiment and clear its active selector when needed."""
        self.validate_experiment_name(experiment_name)
        experiment_dir = self.project_root / self.data_dir_name / experiment_name
        if not experiment_dir.is_dir():
            raise ValueError(f"Experiment does not exist: {experiment_name}")
        shutil.rmtree(experiment_dir)
        if self.get_active_experiment_name() == experiment_name:
            self._default_conf_path().unlink(missing_ok=True)

    def ensure_default_experiment(self):
        """Pick the first available experiment directory and make it active."""
        data_dir = self.project_root / self.data_dir_name
        if not data_dir.exists():
            return None

        for experiment_name in self.list_experiments():
            self.set_active_experiment(experiment_name)
            return experiment_name
        return None


def resolve_experiment_paths(project_root="."):
    """Resolve config, batch, and result paths for the active experiment."""
    root = Path(project_root).resolve()
    manager = ExperimentManager(root)
    experiment_name = manager.get_active_experiment_name()

    if experiment_name:
        manager.validate_experiment_name(experiment_name)
        experiment_dir = root / "data" / experiment_name
        config_path = experiment_dir / "config.json"
        if not experiment_dir.is_dir() or not config_path.is_file():
            raise ExperimentNotSelectedError(
                f"Active experiment is invalid or missing config.json: {experiment_name}"
            )
        return {
            "project_root": root,
            "experiment_name": experiment_name,
            "experiment_dir": experiment_dir,
            "config_path": config_path,
            "batch_path": experiment_dir / "multi.csv",
            "result_dir": experiment_dir / "result",
        }

    raise ExperimentNotSelectedError(
        "Active experiment is not selected. Create default.conf or start the GUI to create a new experiment."
    )


def archive_path(path):
    """Rename an existing path to a unique timestamped backup beside it."""
    path = Path(path)
    if not path.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archive = path.with_name(f"{path.name}_{timestamp}.bak")
    suffix = 1
    while archive.exists():
        archive = path.with_name(f"{path.name}_{timestamp}_{suffix}.bak")
        suffix += 1
    path.rename(archive)
    return archive
