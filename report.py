"""Base contract and validation helpers for dynamic simulation reports."""

from copy import deepcopy


REPORT_BASE_DEFAULTS = {
    "report": "",
    "filename": "",
    "annual": False,
    "final": False,
    "meta": True,
}
REPORT_BASE_DESCRIPTIONS = {
    "report": "Dynamic report module name.",
    "filename": "Output filename without an extension.",
    "annual": "Generate after each graph-generation interval for one tag.",
    "final": "Generate after a successful simulation for one tag.",
    "meta": "Generate after a completed batch using aggregate results.",
}


class ReportConfigError(ValueError):
    """Report an invalid report configuration or plugin declaration."""


class Report:
    """Provide the lifecycle contract for dynamically loaded reports."""

    @staticmethod
    def description():
        """Return a short Markdown description of the report's purpose."""
        return "# Custom report\n\nOverride this description in the report module."

    @staticmethod
    def add_config():
        """Return default configuration fields for one report instance."""
        return {}

    @staticmethod
    def config_descriptions():
        """Return descriptions keyed by editable report configuration field."""
        return dict(REPORT_BASE_DESCRIPTIONS)

    @staticmethod
    def execute(params):
        """Generate report output from validated execution parameters."""
        raise NotImplementedError("Reports must implement execute(params)")


def normalize_report_item(item, expected_report=None):
    """Return one validated report item with common defaults applied."""
    if not isinstance(item, dict):
        raise ReportConfigError("Report item must be an object")
    persisted_item = {
        name: value for name, value in item.items()
        if not name.endswith("_description")
    }
    normalized = {**REPORT_BASE_DEFAULTS, **deepcopy(persisted_item)}
    report_name = normalized["report"]
    if not isinstance(report_name, str):
        raise ReportConfigError("Report item report must be a string")
    if expected_report is not None and report_name != expected_report:
        raise ReportConfigError(
            f"Report item must declare report {expected_report!r}"
        )
    if not isinstance(normalized["filename"], str) or not normalized["filename"].strip():
        raise ReportConfigError("Report item filename must be a non-empty string")
    if any(not isinstance(normalized[name], bool) for name in ("annual", "final", "meta")):
        raise ReportConfigError("Report item annual, final, and meta must be booleans")
    return normalized


def normalize_report_config(config):
    """Return a validated report-config object with a normalized items list."""
    if not isinstance(config, dict):
        raise ReportConfigError("Report configuration must be an object")
    unknown_keys = set(config) - {"items"}
    if unknown_keys:
        raise ReportConfigError(
            f"Report configuration contains unknown keys: {sorted(unknown_keys)}"
        )
    items = config.get("items", [])
    if not isinstance(items, list):
        raise ReportConfigError("Report configuration items must be a list")
    return {"items": [normalize_report_item(item) for item in items]}