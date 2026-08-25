"""Shared lines/points/bars rendering for declared graphs and metagraphs."""

GRAPH_STYLES = {"lines", "points", "bars"}


def render_series(
    ax,
    x_values,
    y_values,
    label,
    style,
    size_values=None,
    color_values=None,
    max_point_size=200.0,
    marker=None,
):
    """Plot one named series as lines (default), points, or bars.

    size_values/color_values (points style only) must already be the same
    length as x_values/y_values with no missing entries.
    """
    if not x_values:
        return
    if style == "bars":
        ax.bar(x_values, y_values, label=label, alpha=0.8)
        return
    if style == "points":
        ax.scatter(
            x_values,
            y_values,
            s=_scale_point_sizes(size_values, max_point_size),
            c=color_values,
            cmap="rainbow_r" if color_values is not None else None,
            label=label,
        )
        return
    ax.plot(x_values, y_values, marker=marker, linewidth=2, label=label)


def _scale_point_sizes(size_values, max_point_size):
    """Return point sizes linearly scaled from 2 to max_point_size."""
    if size_values is None:
        return None
    minimum, maximum = min(size_values), max(size_values)
    if maximum == minimum:
        return [(2.0 + max_point_size) / 2.0] * len(size_values)
    span = maximum - minimum
    return [2.0 + (value - minimum) / span * (max_point_size - 2.0) for value in size_values]
