"""Focused tests for the shared lines/points/bars graph renderer."""

import matplotlib.pyplot as plt
import pytest

from graph_style import _scale_point_sizes, render_series


@pytest.mark.smoke
def test_render_series_defaults_to_lines():
    """Draw a plain line when style is "lines", matching prior behavior."""
    fig, ax = plt.subplots()

    render_series(ax, [1, 2, 3], [0.1, 0.2, 0.3], "Average", "lines")

    assert len(ax.lines) == 1
    assert ax.lines[0].get_label() == "Average"
    assert ax.lines[0].get_marker() == "None"
    plt.close(fig)


@pytest.mark.smoke
def test_render_series_lines_passes_through_marker():
    """Use the caller-provided marker for lines, preserving each call site's look."""
    fig, ax = plt.subplots()

    render_series(ax, [1, 2], [0.1, 0.2], "Average", "lines", marker="o")

    assert ax.lines[0].get_marker() == "o"
    plt.close(fig)


@pytest.mark.smoke
def test_render_series_bars_draws_bar_patches():
    """Draw one bar per x value when style is "bars"."""
    fig, ax = plt.subplots()

    render_series(ax, [1, 2, 3], [4, 5, 6], "Count", "bars")

    assert len(ax.patches) == 3
    assert len(ax.lines) == 0
    plt.close(fig)


@pytest.mark.smoke
def test_render_series_points_draws_scatter_without_size_or_color():
    """Draw a plain scatter when style is "points" without values2/values3."""
    fig, ax = plt.subplots()

    render_series(ax, [1, 2, 3], [0.1, 0.2, 0.3], "Average", "points")

    assert len(ax.collections) == 1
    plt.close(fig)


@pytest.mark.smoke
def test_render_series_points_scales_sizes_and_colors():
    """Scale marker sizes between 2 and max_point_size and forward color values."""
    fig, ax = plt.subplots()

    render_series(
        ax,
        [1, 2, 3],
        [0.1, 0.2, 0.3],
        "Average",
        "points",
        size_values=[0.0, 5.0, 10.0],
        color_values=[1.0, 2.0, 3.0],
        max_point_size=100.0,
    )

    scatter = ax.collections[0]
    sizes = scatter.get_sizes()
    assert sizes[0] == pytest.approx(2.0)
    assert sizes[-1] == pytest.approx(100.0)
    assert scatter.get_cmap().name == "rainbow_r"
    plt.close(fig)


@pytest.mark.smoke
def test_render_series_skips_empty_series():
    """Add no artists when there is no data to plot."""
    fig, ax = plt.subplots()

    render_series(ax, [], [], "Average", "lines")

    assert len(ax.lines) == 0
    assert len(ax.patches) == 0
    assert len(ax.collections) == 0
    plt.close(fig)


@pytest.mark.smoke
def test_scale_point_sizes_linear_between_min_and_max():
    """Map the smallest and largest values to 2 and max_point_size respectively."""
    sizes = _scale_point_sizes([0.0, 5.0, 10.0], 100.0)

    assert sizes[0] == pytest.approx(2.0)
    assert sizes[1] == pytest.approx(51.0)
    assert sizes[2] == pytest.approx(100.0)


@pytest.mark.smoke
def test_scale_point_sizes_constant_values_use_midpoint():
    """Use the midpoint size when every value is identical."""
    sizes = _scale_point_sizes([3.0, 3.0], 200.0)

    assert sizes == [pytest.approx(101.0), pytest.approx(101.0)]


@pytest.mark.smoke
def test_scale_point_sizes_none_when_no_size_values():
    """Return None so matplotlib uses its default marker size."""
    assert _scale_point_sizes(None, 200.0) is None
