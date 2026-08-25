"""Validate and normalize dynamic model metadata declarations."""

from copy import deepcopy

from graph_style import GRAPH_STYLES


SETTING_KEYS = {"description", "default", "type", "min", "max"}
VALUE_KEYS = {"title", "description", "annual", "final", "format"}
GRAPH_KEYS = {
    "filename", "values", "type", "title", "description", "annual", "final",
    "xlabel", "labels", "animated", "min", "max", "padding_min", "padding_max",
    "scale", "bin_count", "style", "values2", "values3", "max_point_size",
}
METAGRAPH_KEYS = {
    "filename", "values", "title", "description", "xlabel", "xvalue", "labels",
    "animated", "data_label", "style", "values2", "values3", "max_point_size",
}
SETTING_TYPES = {"int": int, "float": float, "str": str, "bool": bool}


class ModelMetadataError(ValueError):
    """Report invalid model declaration metadata."""


def _reject_unknown_keys(metadata, allowed_keys, path):
    """Reject metadata keys outside the strict declaration schema."""
    unknown_keys = set(metadata) - allowed_keys
    if unknown_keys:
        raise ModelMetadataError(f"{path} contains unknown keys: {sorted(unknown_keys)}")


def _normalize_style_fields(metadata, graph_values, source_names, path):
    """Validate and normalize optional style/values2/values3/max_point_size keys."""
    style = metadata.get("style", "lines")
    if style not in GRAPH_STYLES:
        raise ModelMetadataError(f"{path}.style must be one of {sorted(GRAPH_STYLES)}")
    result = {"style": style}
    for key in ("values2", "values3"):
        series = metadata.get(key)
        if series is None:
            continue
        if not isinstance(series, list) or len(series) != len(graph_values) or not all(
            isinstance(name, str) and name for name in series
        ):
            raise ModelMetadataError(f"{path}.{key} must match values")
        missing_names = [name for name in series if name not in source_names]
        if missing_names:
            raise ModelMetadataError(f"{path}.{key} references missing names: {missing_names}")
        result[key] = list(series)
    max_point_size = metadata.get("max_point_size", 200)
    if (
        not isinstance(max_point_size, (int, float))
        or isinstance(max_point_size, bool)
        or max_point_size <= 2
    ):
        raise ModelMetadataError(f"{path}.max_point_size must be a number greater than 2")
    result["max_point_size"] = float(max_point_size)
    return result


def _require_named_dict(value, method_name):
    """Require a dictionary keyed by non-empty string names."""
    if not isinstance(value, dict):
        raise ModelMetadataError(f"{method_name}() must return a dict")
    for name in value:
        if not isinstance(name, str) or not name:
            raise ModelMetadataError(f"{method_name}() names must be non-empty strings")
    return value


def _normalize_settings(model_class):
    """Normalize and validate model setting declarations."""
    declarations = _require_named_dict(model_class.add_settings(), "add_settings")
    normalized = {}
    for name, raw_metadata in declarations.items():
        path = f"settings.{name}"
        if not isinstance(raw_metadata, dict):
            raise ModelMetadataError(f"{path} must be a dict")
        metadata = deepcopy(raw_metadata)
        _reject_unknown_keys(metadata, SETTING_KEYS, path)
        if "description" not in metadata or not isinstance(metadata["description"], str):
            raise ModelMetadataError(f"{path}.description is required and must be a string")
        if "default" not in metadata:
            raise ModelMetadataError(f"{path}.default is required")

        type_name = metadata.get("type", "float")
        if type_name not in SETTING_TYPES:
            raise ModelMetadataError(f"{path}.type must be one of {sorted(SETTING_TYPES)}")
        expected_type = SETTING_TYPES[type_name]
        default = metadata["default"]
        if type_name in {"int", "float"}:
            if isinstance(default, bool) or not isinstance(default, (int, float)):
                raise ModelMetadataError(f"{path}.default must match type {type_name}")
            default = expected_type(default)
        elif not isinstance(default, expected_type):
            raise ModelMetadataError(f"{path}.default must match type {type_name}")

        minimum = metadata.get("min")
        maximum = metadata.get("max")
        if minimum is not None and (isinstance(minimum, bool) or not isinstance(minimum, (int, float))):
            raise ModelMetadataError(f"{path}.min must be numeric")
        if maximum is not None and (isinstance(maximum, bool) or not isinstance(maximum, (int, float))):
            raise ModelMetadataError(f"{path}.max must be numeric")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ModelMetadataError(f"{path}.min must not exceed max")
        if minimum is not None and default < minimum:
            raise ModelMetadataError(f"{path}.default is below min")
        if maximum is not None and default > maximum:
            raise ModelMetadataError(f"{path}.default is above max")

        normalized_metadata = {
            "description": metadata["description"],
            "default": default,
            "type": type_name,
        }
        if minimum is not None:
            normalized_metadata["min"] = minimum
        if maximum is not None:
            normalized_metadata["max"] = maximum
        normalized[name] = normalized_metadata
    return normalized


def _normalize_values(model_class):
    """Normalize and validate scalar value declarations."""
    declarations = _require_named_dict(model_class.add_values(), "add_values")
    normalized = {}
    for name, raw_metadata in declarations.items():
        path = f"values.{name}"
        if not isinstance(raw_metadata, dict):
            raise ModelMetadataError(f"{path} must be a dict")
        metadata = deepcopy(raw_metadata)
        _reject_unknown_keys(metadata, VALUE_KEYS, path)
        result = {
            "title": metadata.get("title", name),
            "description": metadata.get("description", ""),
            "annual": metadata.get("annual", True),
            "final": metadata.get("final", False),
            "format": metadata.get("format", ".4f"),
        }
        if not isinstance(result["title"], str) or not isinstance(result["description"], str):
            raise ModelMetadataError(f"{path} title and description must be strings")
        if not isinstance(result["annual"], bool) or not isinstance(result["final"], bool):
            raise ModelMetadataError(f"{path} annual and final must be bool")
        if not isinstance(result["format"], str) or not result["format"]:
            raise ModelMetadataError(f"{path}.format must be a non-empty string")
        normalized[name] = result
    return normalized


def _normalize_graphs(model_class, values):
    """Normalize graph declarations and validate their data references."""
    declarations = model_class.add_graphs()
    if not isinstance(declarations, list):
        raise ModelMetadataError("add_graphs() must return a list")
    public_fields = {
        name for name, metadata in model_class.add_population_fields().items()
        if isinstance(metadata, dict) and metadata.get("public", False)
    }
    normalized = []
    for index, raw_metadata in enumerate(declarations):
        path = f"graphs[{index}]"
        if not isinstance(raw_metadata, dict):
            raise ModelMetadataError(f"{path} must be a dict")
        metadata = deepcopy(raw_metadata)
        _reject_unknown_keys(metadata, GRAPH_KEYS, path)
        filename = metadata.get("filename")
        graph_values = metadata.get("values")
        if not isinstance(filename, str) or not filename:
            raise ModelMetadataError(f"{path}.filename is required and must be a non-empty string")
        if not isinstance(graph_values, list) or not graph_values or not all(
            isinstance(name, str) and name for name in graph_values
        ):
            raise ModelMetadataError(f"{path}.values must be a non-empty list of names")

        graph_type = metadata.get("type", "distr")
        if graph_type not in {"time", "distr"}:
            raise ModelMetadataError(f"{path}.type must be time or distr")
        source_names = set(values) if graph_type == "time" else public_fields
        missing_names = [name for name in graph_values if name not in source_names]
        if missing_names:
            raise ModelMetadataError(f"{path}.values references missing names: {missing_names}")
        style_keys = ("style", "values2", "values3", "max_point_size")
        if graph_type == "distr" and any(key in metadata for key in style_keys):
            raise ModelMetadataError(f"{path} style options are only supported for time graphs")
        style_fields = _normalize_style_fields(metadata, graph_values, source_names, path)

        annual = metadata.get("annual", True)
        final = metadata.get("final", False)
        if not isinstance(annual, bool) or not isinstance(final, bool):
            raise ModelMetadataError(f"{path} annual and final must be bool")
        if not annual and not final:
            raise ModelMetadataError(f"{path} must enable annual or final output")
        labels = metadata.get("labels", list(graph_values))
        if not isinstance(labels, list) or len(labels) != len(graph_values) or not all(
            isinstance(label, str) for label in labels
        ):
            raise ModelMetadataError(f"{path}.labels must match values")

        bin_count = metadata.get("bin_count", 50)
        scale = metadata.get("scale", 1.0)
        padding_min = metadata.get("padding_min", 0.0)
        padding_max = metadata.get("padding_max", 0.0)
        if not isinstance(bin_count, int) or isinstance(bin_count, bool) or bin_count < 1:
            raise ModelMetadataError(f"{path}.bin_count must be a positive integer")
        if not isinstance(scale, (int, float)) or isinstance(scale, bool) or scale <= 0:
            raise ModelMetadataError(f"{path}.scale must be positive")
        if not 0 <= padding_min < 1 or not 0 <= padding_max < 1 or padding_min + padding_max >= 1:
            raise ModelMetadataError(f"{path} padding values are invalid")
        minimum = metadata.get("min")
        maximum = metadata.get("max")
        if minimum is not None and maximum is not None and minimum >= maximum:
            raise ModelMetadataError(f"{path}.min must be less than max")

        result = {
            "filename": filename,
            "values": list(graph_values),
            "type": graph_type,
            "title": metadata.get("title", filename),
            "description": metadata.get("description", ""),
            "annual": annual,
            "final": final,
            "xlabel": metadata.get("xlabel", ""),
            "labels": list(labels),
            "animated": metadata.get("animated", True),
            "bin_count": bin_count,
            "scale": float(scale),
            "padding_min": float(padding_min),
            "padding_max": float(padding_max),
            **style_fields,
        }
        if minimum is not None:
            result["min"] = minimum
        if maximum is not None:
            result["max"] = maximum
        for text_key in ("title", "description", "xlabel"):
            if not isinstance(result[text_key], str):
                raise ModelMetadataError(f"{path}.{text_key} must be a string")
        if not isinstance(result["animated"], bool):
            raise ModelMetadataError(f"{path}.animated must be bool")
        normalized.append(result)
    return normalized


def _normalize_metagraphs(model_class, settings, values):
    """Normalize aggregate graph declarations and validate report references."""
    declarations = model_class.add_metagraphs()
    if not isinstance(declarations, list):
        raise ModelMetadataError("add_metagraphs() must return a list")
    normalized = []
    for index, raw_metadata in enumerate(declarations):
        path = f"metagraphs[{index}]"
        if not isinstance(raw_metadata, dict):
            raise ModelMetadataError(f"{path} must be a dict")
        metadata = deepcopy(raw_metadata)
        _reject_unknown_keys(metadata, METAGRAPH_KEYS, path)
        filename = metadata.get("filename")
        graph_values = metadata.get("values")
        if not isinstance(filename, str) or not filename:
            raise ModelMetadataError(f"{path}.filename is required and must be a non-empty string")
        if not isinstance(graph_values, list) or not graph_values or not all(
            isinstance(name, str) and name for name in graph_values
        ):
            raise ModelMetadataError(f"{path}.values must be a non-empty list of names")
        missing_names = [name for name in graph_values if name not in values]
        if missing_names:
            raise ModelMetadataError(f"{path}.values references missing names: {missing_names}")
        xvalue = metadata.get("xvalue")
        if xvalue is not None and (not isinstance(xvalue, str) or not xvalue):
            raise ModelMetadataError(f"{path}.xvalue must be a non-empty string")
        if xvalue is not None and xvalue not in settings and xvalue not in values:
            raise ModelMetadataError(f"{path}.xvalue references missing name: {xvalue}")
        data_label = metadata.get("data_label")
        if data_label is not None and data_label != "tag":
            raise ModelMetadataError(f"{path}.data_label currently only supports \"tag\"")
        labels = metadata.get("labels", list(graph_values))
        if not isinstance(labels, list) or len(labels) != len(graph_values) or not all(
            isinstance(label, str) for label in labels
        ):
            raise ModelMetadataError(f"{path}.labels must match values")
        style_fields = _normalize_style_fields(metadata, graph_values, values, path)
        result = {
            "filename": filename,
            "values": list(graph_values),
            "title": metadata.get("title", filename),
            "description": metadata.get("description", ""),
            "xlabel": metadata.get("xlabel", ""),
            "labels": list(labels),
            "animated": metadata.get("animated", True),
            **style_fields,
        }
        if xvalue is not None:
            result["xvalue"] = xvalue
        if data_label is not None:
            result["data_label"] = data_label
        for text_key in ("title", "description", "xlabel"):
            if not isinstance(result[text_key], str):
                raise ModelMetadataError(f"{path}.{text_key} must be a string")
        if not isinstance(result["animated"], bool):
            raise ModelMetadataError(f"{path}.animated must be bool")
        normalized.append(result)
    return normalized


def validate_model_metadata(model_class):
    """Return normalized settings, scalar values, graph, and metagraph declarations."""
    settings = _normalize_settings(model_class)
    values = _normalize_values(model_class)
    graphs = _normalize_graphs(model_class, values)
    metagraphs = _normalize_metagraphs(model_class, settings, values)
    return {
        "settings": settings,
        "values": values,
        "graphs": graphs,
        "metagraphs": metagraphs,
    }