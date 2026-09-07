"""Discover, validate, and execute trusted dynamic report modules."""

import json
from pathlib import Path
import re

from load_model import _execute_fresh_module
from report import Report, ReportConfigError, normalize_report_config, normalize_report_item


REPORT_NAME_PATTERN = re.compile(r"report_[A-Za-z0-9_]+\Z")
DEFAULT_REPORT_DIRECTORY = Path(__file__).resolve().parent


class ReportLoadError(RuntimeError):
    """Report an invalid dynamic report identifier, module, or class."""


def _resolve_report_directory(report_directory):
    """Return the selected trusted report directory as an absolute path."""
    directory = DEFAULT_REPORT_DIRECTORY if report_directory is None else report_directory
    return Path(directory).resolve()


def discover_reports(report_directory=None):
    """Return sorted valid dynamic report module names from one directory."""
    directory = _resolve_report_directory(report_directory)
    if not directory.is_dir():
        return []
    return sorted(
        path.stem
        for path in directory.glob("report_*.py")
        if path.is_file() and REPORT_NAME_PATTERN.fullmatch(path.stem)
    )


def load_report_class(module_name, report_directory=None):
    """Fresh-load and return one fixed-name Report subclass."""
    if not isinstance(module_name, str) or not REPORT_NAME_PATTERN.fullmatch(module_name):
        raise ReportLoadError(f"Invalid report identifier: {module_name!r}")
    directory = _resolve_report_directory(report_directory)
    module_path = directory / f"{module_name}.py"
    if not module_path.is_file():
        raise ReportLoadError(f"Report file not found: {module_path}")
    try:
        module = _execute_fresh_module(module_name, module_path, directory)
    except Exception as error:
        raise ReportLoadError(f"Failed to load report {module_name}: {error}") from error
    class_name = module_name[0].upper() + module_name[1:]
    report_class = getattr(module, class_name, None)
    if not isinstance(report_class, type):
        raise ReportLoadError(f"Report module must define class {class_name}")
    if not issubclass(report_class, Report):
        raise ReportLoadError(f"{class_name} must be a subclass of Report")
    try:
        default_item = report_class.add_config()
        normalize_report_item(default_item, expected_report=module_name)
    except (ReportConfigError, TypeError, ValueError) as error:
        raise ReportLoadError(f"Invalid default config for {module_name}: {error}") from error
    if not isinstance(report_class.description(), str):
        raise ReportLoadError(f"Report {module_name} description() must return a string")
    descriptions = report_class.config_descriptions()
    if not isinstance(descriptions, dict) or not all(
        isinstance(name, str) and isinstance(text, str)
        for name, text in descriptions.items()
    ):
        raise ReportLoadError(f"Report {module_name} config_descriptions() must return string pairs")
    return report_class


def load_report_config(config_path):
    """Load and validate report_config.json, treating an absent file as disabled."""
    path = Path(config_path)
    if not path.is_file():
        return {"items": []}
    try:
        with path.open(encoding="utf-8") as config_file:
            return normalize_report_config(json.load(config_file))
    except (OSError, json.JSONDecodeError, ReportConfigError) as error:
        raise ReportLoadError(f"Invalid report configuration {path}: {error}") from error


def execute_reports(
    config_path,
    trigger,
    experiment_dir,
    result,
    tag=None,
    output_dir=None,
    year=None,
    selected_index=None,
    logger=print,
):
    """Execute configured reports enabled for one annual, final, or meta trigger."""
    if trigger not in {"annual", "final", "meta"}:
        raise ValueError("Report trigger must be annual, final, or meta")
    config = load_report_config(config_path)
    output_dir = Path(output_dir) if output_dir is not None else Path(experiment_dir) / "result"
    generated_paths = []
    for index, item in enumerate(config["items"]):
        if selected_index is not None and index != selected_index:
            continue
        if not item[trigger]:
            continue
        try:
            report_class = load_report_class(item["report"])
            validated_item = normalize_report_item(item, expected_report=item["report"])
            params = {
                "experiment_dir": Path(experiment_dir),
                "tag": tag,
                "trigger": trigger,
                "config": validated_item,
                "result": result,
                "output_dir": output_dir,
                "year": year,
            }
            produced = report_class.execute(params)
            if produced is not None:
                generated_paths.append(str(produced))
        except Exception as error:
            logger(f"Report {item.get('report', '')} failed: {error}")
    return generated_paths