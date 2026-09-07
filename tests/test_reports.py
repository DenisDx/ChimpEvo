"""Focused tests for dynamic report discovery and generation."""

from load_report import discover_reports, execute_reports, load_report_class
from report import normalize_report_config
from report_heatmap import _normalize_numeric


def _aggregate_rows():
    """Return a complete two-by-two-by-two-by-two aggregate result grid."""
    rows = []
    for mutation_probability in ("0.1", "0.2"):
        for mortality_lambda in ("0.01", "0.02"):
            for mutation_s in ("-0.5", "0.5"):
                for mutation_x in ("0.5", "1.0"):
                    rows.append({
                        "mutation_probability": mutation_probability,
                        "lambda": mortality_lambda,
                        "mutation_s": mutation_s,
                        "mutation_x": mutation_x,
                        "avg_beta": str(float(mutation_probability) + float(mortality_lambda)),
                    })
    return rows


def test_heatmap_report_is_discoverable_and_has_valid_defaults():
    """Load the bundled heatmap plugin through the dynamic report contract."""
    assert "report_heatmap" in discover_reports()
    report_class = load_report_class("report_heatmap")
    assert report_class.add_config()["report"] == "report_heatmap"


def test_report_config_normalization_removes_legacy_description_fields():
    """Keep GUI-only legacy descriptions out of the persisted report config."""
    config = normalize_report_config({
        "items": [{
            "report": "report_heatmap",
            "filename": "heatmap",
            "value_description": "Old stored description",
        }],
    })
    assert "value_description" not in config["items"][0]


def test_heatmap_numeric_tolerance_groups_equivalent_dimension_values():
    """Treat equivalent CSV number spellings and rounding noise as one value."""
    assert _normalize_numeric("1") == _normalize_numeric("1.0")
    assert _normalize_numeric("0.99999999999999") == _normalize_numeric("1.0")


def test_heatmap_ignores_incomplete_rows_and_averages_duplicate_coordinates(tmp_path):
    """Generate a usable grid when a batch has incomplete or repeated result rows."""
    config_path = tmp_path / "report_config.json"
    config_path.write_text(
        """{"items": [{"report": "report_heatmap", "filename": "heatmap", "meta": true,
        "value": "avg_beta", "value1": "mutation_probability", "value2": "lambda",
        "value3": "mutation_s", "value4": "mutation_x", "title": "Beta",
        "title1": "P", "title2": "L", "title3": "S", "title4": "X", "filter": ""}]}""",
        encoding="utf-8",
    )
    rows = _aggregate_rows()
    rows.extend([dict(rows[0]), {**rows[1], "avg_beta": ""}])
    paths = execute_reports(config_path, "meta", tmp_path, rows, output_dir=tmp_path / "result")
    assert paths == [str(tmp_path / "result" / "heatmap.pdf")]


def test_execute_meta_heatmap_writes_pdf(tmp_path):
    """Generate the configured heatmap PDF from complete aggregate rows."""
    config_path = tmp_path / "report_config.json"
    config_path.write_text(
        """{
  "items": [{
    "report": "report_heatmap",
    "filename": "heatmap",
    "meta": true,
    "value": "avg_beta",
    "value1": "mutation_probability",
    "value2": "lambda",
    "value3": "mutation_s",
    "value4": "mutation_x",
    "title": "Beta",
    "title1": "Mutation probability",
    "title2": "Lambda",
    "title3": "S",
    "title4": "X",
    "filter": ""
  }]
}""",
        encoding="utf-8",
    )
    paths = execute_reports(
        config_path,
        "meta",
        tmp_path,
        _aggregate_rows(),
        output_dir=tmp_path / "result",
    )
    assert paths == [str(tmp_path / "result" / "heatmap.pdf")]
    assert (tmp_path / "result" / "heatmap.pdf").is_file()