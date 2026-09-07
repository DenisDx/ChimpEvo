"""Render aggregate batch results as a four-dimensional PDF heatmap."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np

from report import Report


NUMERIC_TOLERANCE = 1e-9


def _sort_values(values):
    """Return values ordered numerically where every value is numeric."""
    try:
        return sorted(values, key=float)
    except (TypeError, ValueError):
        return sorted(values, key=str)


def _normalize_numeric(value):
    """Return one numeric value rounded to the shared comparison tolerance."""
    return round(float(value) / NUMERIC_TOLERANCE) * NUMERIC_TOLERANCE


def _format_numeric(value):
    """Return a compact stable label for one normalized numeric value."""
    return f"{value:.12g}"


def _parse_filter(filter_text):
    """Parse an empty or JSON range-filter string into field bounds."""
    if not isinstance(filter_text, str):
        raise ValueError("filter must be an empty string or a JSON object")
    filter_text = filter_text.strip()
    if not filter_text:
        return {}
    try:
        filter_value = json.loads(filter_text)
    except json.JSONDecodeError as error:
        raise ValueError("filter must be a JSON object of [min, max] ranges") from error
    if not isinstance(filter_value, dict):
        raise ValueError("filter must be a JSON object of [min, max] ranges")
    normalized = {}
    for name, bounds in filter_value.items():
        if (
            not isinstance(name, str)
            or not isinstance(bounds, list)
            or len(bounds) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bounds)
            or bounds[0] > bounds[1]
        ):
            raise ValueError("filter values must be [min, max] numeric ranges")
        normalized[name] = bounds
    return normalized


def _parse_color_boundary(value, name):
    """Return an optional finite color-range boundary from a string setting."""
    if value == "":
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a number or an empty string")
    try:
        boundary = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number or an empty string") from error
    if not np.isfinite(boundary):
        raise ValueError(f"{name} must be finite")
    return boundary


def _get_figure_width(values1, values3, title1):
    """Return a PDF width that fits the outer horizontal group labels."""
    column_count = len(values1) * len(values3)
    label_width = max(
        len(f"{title1}: {_format_numeric(value1)}") * 0.08 + 0.40
        for value1 in values1
    )
    return max(8.0, column_count * 0.9 + len(values1) * 0.12 + 2.3, len(values1) * label_width / 0.73)


class Report_heatmap(Report):
    """Render one aggregate metric over four result-table dimensions."""

    @staticmethod
    def description():
        """Return the heatmap report description as lightweight Markdown."""
        return "# Four-dimensional heatmap\n\nRenders aggregate batch results as colored tables in one PDF."

    @staticmethod
    def add_config():
        """Return the default heatmap configuration item."""
        return {
            "report": "report_heatmap",
            "filename": "report_heatmap",
            "annual": False,
            "final": False,
            "meta": True,
            "value": "avg_beta",
            "value1": "mutation_probability",
            "value2": "lambda",
            "value3": "mutation_s",
            "value4": "mutation_x",
            "title": "Average beta",
            "title1": "Mutation probability",
            "title2": "Lambda",
            "title3": "Mutation asymmetry (S)",
            "title4": "Mutation effect size (X)",
            "filter": "",
            "color_range_min": "",
            "color_range_max": "",
        }

    @staticmethod
    def config_descriptions():
        """Return descriptions for heatmap configuration fields."""
        return {
            **Report.config_descriptions(),
            "value": "Numeric result column shown in each colored cell.",
            "value1": "Outer horizontal dimension.",
            "value2": "Outer vertical dimension.",
            "value3": "Inner-table horizontal dimension.",
            "value4": "Inner-table vertical dimension.",
            "title": "PDF title.",
            "title1": "Label for the outer horizontal dimension.",
            "title2": "Label for the outer vertical dimension.",
            "title3": "Label for the inner horizontal dimension.",
            "title4": "Label for the inner vertical dimension.",
            "filter": "JSON numeric ranges, for example {\"lambda\": [0.01, 0.05]}.",
            "color_range_min": "Minimum color-scale value; empty uses the smallest cell value.",
            "color_range_max": "Maximum color-scale value; empty uses the largest cell value.",
        }

    @staticmethod
    def execute(params):
        """Write one PDF heatmap from aggregate result rows and return its path."""
        if params["trigger"] != "meta":
            return None
        config = params["config"]
        required_fields = [config[name] for name in ("value", "value1", "value2", "value3", "value4")]
        if any(not isinstance(name, str) or not name for name in required_fields):
            raise ValueError("value and value1 through value4 must be non-empty column names")
        filter_ranges = _parse_filter(config.get("filter", ""))
        rows = []
        for row in params["result"]:
            try:
                if not all(
                    bounds[0] - NUMERIC_TOLERANCE <= float(row[field]) <= bounds[1] + NUMERIC_TOLERANCE
                    for field, bounds in filter_ranges.items()
                ):
                    continue
                for field in required_fields:
                    if row[field] == "":
                        raise ValueError
                    float(row[field])
                rows.append(row)
            except (KeyError, TypeError, ValueError):
                continue
        if not rows:
            raise ValueError("No complete numeric aggregate rows remain after applying the filter")

        value_name, value1_name, value2_name, value3_name, value4_name = required_fields
        cell_samples = {}
        for row in rows:
            key = tuple(
                _normalize_numeric(row[name])
                for name in (value1_name, value2_name, value3_name, value4_name)
            )
            cell_samples.setdefault(key, []).append(float(row[value_name]))
        values1 = _sort_values({key[0] for key in cell_samples})
        values2 = _sort_values({key[1] for key in cell_samples})
        values3 = _sort_values({key[2] for key in cell_samples})
        values4 = _sort_values({key[3] for key in cell_samples})
        cells = {
            key: float(np.mean(samples))
            for key, samples in cell_samples.items()
        }
        color_values = np.array(list(cells.values()), dtype=float)
        if not np.isfinite(color_values).all():
            raise ValueError("Heatmap value column must contain finite numbers")
        minimum = _parse_color_boundary(config.get("color_range_min", ""), "color_range_min")
        maximum = _parse_color_boundary(config.get("color_range_max", ""), "color_range_max")
        if minimum is None:
            minimum = float(color_values.min())
        if maximum is None:
            maximum = float(color_values.max())
        if minimum > maximum:
            raise ValueError("color_range_min must not exceed color_range_max")
        if minimum == maximum:
            maximum = minimum + 1.0
        normalization = colors.Normalize(vmin=minimum, vmax=maximum, clip=True)
        colormap = LinearSegmentedColormap.from_list(
            "muted_blue_orange",
            ["#2166ac", "#f7f7f7", "#b35806"],
        )

        output_dir = Path(params["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{params['year']}" if params["trigger"] == "annual" and params["year"] is not None else ""
        output_path = output_dir / f"{config['filename']}{suffix}.pdf"
        column_count = len(values1) * len(values3)
        row_count = len(values2) * len(values4)
        row_group_gap = 0.35
        table_height = row_count + (len(values2) - 1) * row_group_gap
        figure_width = _get_figure_width(values1, values3, config["title1"])
        figure_height = max(6.0, table_height * 0.55 + len(values2) * 0.22 + 1.6)
        with PdfPages(output_path) as pdf:
            figure = plt.figure(figsize=(figure_width, figure_height))
            table_axis = figure.add_axes((0.10, 0.14, 0.73, 0.70))
            header_axis = figure.add_axes((0.10, 0.86, 0.73, 0.05))
            group_axis = figure.add_axes((0.84, 0.14, 0.035, 0.70))
            color_axis = figure.add_axes((0.92, 0.15, 0.025, 0.62))
            table_axis.set_xlim(0, column_count)
            table_axis.set_ylim(table_height, 0)
            table_axis.set_xticks([
                index + 0.5
                for index in range(column_count)
            ], [
                _format_numeric(value3)
                for _ in values1
                for value3 in values3
            ], fontsize=8)
            table_axis.set_yticks([])
            table_axis.set_xlabel(config["title3"], fontsize=10)
            table_axis.tick_params(length=0, pad=7)
            for spine in table_axis.spines.values():
                spine.set_visible(False)
            for row_index, dimension2_value in enumerate(values2):
                group_start = row_index * (len(values4) + row_group_gap)
                for column_index, dimension1_value in enumerate(values1):
                    for inner_row, dimension4_value in enumerate(values4):
                        for inner_column, dimension3_value in enumerate(values3):
                            key = (dimension1_value, dimension2_value, dimension3_value, dimension4_value)
                            x = column_index * len(values3) + inner_column
                            y = group_start + inner_row
                            cell_value = cells.get(key)
                            color = "#f7f7f7" if cell_value is None else colormap(normalization(cell_value))
                            table_axis.add_patch(Rectangle(
                                (x, y), 1, 1,
                                facecolor=color,
                                edgecolor="white",
                                linewidth=0.8,
                            ))
                            if cell_value is not None:
                                sample_count = len(cell_samples[key])
                                table_axis.text(
                                    x + 0.5,
                                    y + (0.42 if sample_count > 1 else 0.5),
                                    f"{cell_value:.3g}",
                                    ha="center",
                                    va="center",
                                    fontsize=8,
                                )
                                if sample_count > 1:
                                    table_axis.text(
                                        x + 0.5,
                                        y + 0.68,
                                        f"(n={sample_count})",
                                        ha="center",
                                        va="center",
                                        fontsize=5.5,
                                        color="#444444",
                                    )
            for group_index in range(len(values2)):
                group_start = group_index * (len(values4) + row_group_gap)
                for value_index, value4 in enumerate(values4):
                    table_axis.text(
                        -0.04,
                        group_start + value_index + 0.5,
                        _format_numeric(value4),
                        ha="right",
                        va="center",
                        fontsize=8,
                        clip_on=False,
                    )
            for group_index in range(1, len(values1)):
                table_axis.axvline(group_index * len(values3), color="white", linewidth=9)

            header_axis.set_xlim(0, column_count)
            header_axis.set_ylim(0, 1)
            header_axis.axis("off")
            for index, value1 in enumerate(values1):
                start = index * len(values3)
                header_axis.add_patch(Rectangle(
                    (start + 0.05, 0.08), len(values3) - 0.1, 0.84,
                    facecolor="#e5f0fb",
                    edgecolor="#c4cbd2",
                    linewidth=0.8,
                ))
                header_axis.text(
                    start + len(values3) / 2,
                    0.5,
                    f"{config['title1']}: {_format_numeric(value1)}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    color="#00468b",
                )

            group_axis.set_xlim(0, 1)
            group_axis.set_ylim(table_height, 0)
            group_axis.axis("off")
            for index, value2 in enumerate(values2):
                start = index * (len(values4) + row_group_gap)
                group_axis.add_patch(Rectangle(
                    (0.08, start + 0.05), 0.84, len(values4) - 0.1,
                    facecolor="#fde5e5",
                    edgecolor="#d6c2c2",
                    linewidth=0.8,
                ))
                group_axis.text(
                    0.5,
                    start + len(values4) / 2,
                    f"{config['title2']}: {_format_numeric(value2)}",
                    ha="center",
                    va="center",
                    rotation=270,
                    fontsize=7,
                    fontweight="bold",
                    color="#9d0000",
                )
            figure.suptitle(config["title"], fontsize=13, y=0.99)
            figure.text(
                0.025,
                0.49,
                config["title4"],
                ha="center",
                va="center",
                rotation=90,
                fontsize=10,
            )
            figure.colorbar(
                plt.cm.ScalarMappable(norm=normalization, cmap=colormap),
                cax=color_axis,
                label=value_name,
            )
            pdf.savefig(figure)
            plt.close(figure)
        return output_path