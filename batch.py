"""
Batch simulation runner: execute multiple simulations with parameter sweeps
Loads parameter variations from multi.csv and runs sequentially
"""

import csv
import json
import shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from main import run_simulation, log
from model_loader import load_model_class
from model_metadata import validate_model_metadata
from settings import DEFAULT_SETTINGS, PARAMETER_RANGES


class BatchValidationError(ValueError):
    """Report invalid batch rows or incompatible persisted batch results."""


def _load_variants(csv_path, base_config):
    """Return validated tag-first batch rows and their CSV field order."""
    path = Path(csv_path)
    if not path.exists():
        return ["tag"], [{"tag": base_config.get("tag", "default")}]
    with path.open(newline="", encoding="utf-8") as batch_file:
        reader = csv.DictReader(batch_file)
        fieldnames = reader.fieldnames
        if not fieldnames or fieldnames[0] != "tag":
            raise BatchValidationError("Batch CSV must have tag as its first column")
        variants = list(reader)

    tags = [row.get("tag", "").strip() for row in variants]
    if any(not tag for tag in tags):
        raise BatchValidationError("Batch tags must be non-empty")
    if len(tags) != len(set(tags)):
        raise BatchValidationError("Batch tags must be unique")
    configurations = [
        tuple((name, row.get(name, "")) for name in fieldnames if name != "tag")
        for row in variants
    ]
    if len(configurations) != len(set(configurations)):
        raise BatchValidationError("Batch parameter rows must be unique")
    return fieldnames, variants


def _parse_value(value_text, type_name):
    """Parse one CSV value according to a declared setting type."""
    if type_name == "int":
        return int(value_text)
    if type_name == "float":
        return float(value_text)
    if type_name == "str":
        return value_text
    if value_text.lower() in {"true", "1", "yes"}:
        return True
    if value_text.lower() in {"false", "0", "no"}:
        return False
    raise ValueError("must be a boolean")


def _resolve_row_config(base_config, variant, model_settings):
    """Merge one batch row into base config using core and model metadata."""
    config = dict(base_config)
    for name, value_text in variant.items():
        if name == "tag":
            config["tag"] = value_text
            continue
        if value_text is None or value_text == "":
            continue
        if name in model_settings:
            metadata = model_settings[name]
            try:
                value = _parse_value(value_text, metadata["type"])
            except ValueError as error:
                raise BatchValidationError(f"Invalid {name}={value_text}: {error}") from error
            if "min" in metadata:
                value = max(value, metadata["min"])
            if "max" in metadata:
                value = min(value, metadata["max"])
            config[name] = value
            continue
        if name not in DEFAULT_SETTINGS:
            raise BatchValidationError(f"Unknown batch setting: {name}")
        default = DEFAULT_SETTINGS[name]
        type_name = "bool" if isinstance(default, bool) else "int" if isinstance(default, int) else "float" if isinstance(default, float) else "str"
        try:
            value = _parse_value(value_text, type_name)
        except ValueError as error:
            raise BatchValidationError(f"Invalid {name}={value_text}: {error}") from error
        if name in PARAMETER_RANGES:
            minimum, maximum = PARAMETER_RANGES[name]
            value = max(minimum, min(value, maximum))
        config[name] = value
    return config


def _read_final_row(tag, model_name):
    """Read and validate one completed row's successful final CSV."""
    final_path = Path("result") / tag / "final.csv"
    if not final_path.is_file():
        raise BatchValidationError(f"Completed tag {tag} is missing final.csv")
    with final_path.open(newline="", encoding="utf-8") as final_file:
        rows = list(csv.DictReader(final_file))
    if len(rows) != 1 or rows[0].get("tag") != tag or rows[0].get("model") != model_name:
        raise BatchValidationError(f"Completed tag {tag} has incompatible final.csv")
    return rows[0]


def _load_aggregate_rows(report_path, model_name, active_tags):
    """Read and validate prior aggregate rows for one selected model."""
    if not report_path.exists():
        return [], []
    with report_path.open(newline="", encoding="utf-8") as report_file:
        reader = csv.DictReader(report_file)
        fieldnames = reader.fieldnames or []
        if "model" not in fieldnames or "tag" not in fieldnames:
            raise BatchValidationError("Aggregate result.csv must contain model and tag columns")
        rows = list(reader)
    tags = [row.get("tag", "") for row in rows]
    if any(not tag for tag in tags) or len(tags) != len(set(tags)):
        raise BatchValidationError("Aggregate result.csv contains duplicate or empty tags")
    for row in rows:
        if row["model"] != model_name:
            raise BatchValidationError("Aggregate result.csv belongs to a different model")
        _read_final_row(row["tag"], model_name)
    return fieldnames, rows


def _write_aggregate_rows(report_path, fieldnames, rows):
    """Atomically write the aggregate report after successful row completion."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = report_path.with_name(f"{report_path.name}.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(report_path)


def _select_active_rows(rows, variants):
    """Return completed aggregate rows in the current batch CSV's tag order."""
    rows_by_tag = {row["tag"]: row for row in rows}
    return [rows_by_tag[variant["tag"]] for variant in variants if variant["tag"] in rows_by_tag]


def _save_gif(image_paths, output_path):
    """Save compatible image files as one GIF when at least one frame exists."""
    if not image_paths:
        output_path.unlink(missing_ok=True)
        return
    frames = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            frames.append(image.convert("P"))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=500,
        loop=0,
    )


def _render_final_graph_movies(metadata, rows, result_dir):
    """Collect declared per-tag final graph PNGs into root-level GIF movies."""
    for graph in metadata["graphs"]:
        if not graph["final"]:
            continue
        image_paths = [
            result_dir / row["tag"] / f"{graph['filename']}.png"
            for row in rows
            if (result_dir / row["tag"] / f"{graph['filename']}.png").is_file()
        ]
        _save_gif(image_paths, result_dir / f"{graph['filename']}.gif")


def _render_metagraphs(metadata, rows, result_dir):
    """Render cumulative aggregate plots and optional GIFs for declared metagraphs."""
    for graph in metadata["metagraphs"]:
        filename = graph["filename"]
        for stale_frame in result_dir.glob(f"{filename}_*.png"):
            stale_frame.unlink()
        gif_path = result_dir / f"{filename}.gif"
        gif_path.unlink(missing_ok=True)

        xvalue = graph.get("xvalue")
        if xvalue is not None and any(xvalue not in row or row[xvalue] == "" for row in rows):
            log(f"Skipping metagraph {filename}: aggregate report has no {xvalue} values")
            continue
        ordered_rows = list(rows)
        if xvalue is not None:
            try:
                ordered_rows.sort(key=lambda row: float(row[xvalue]))
            except ValueError:
                log(f"Skipping metagraph {filename}: {xvalue} values must be numeric")
                continue

        frame_paths = []
        for index in range(1, len(ordered_rows) + 1):
            frame_rows = ordered_rows[:index]
            x_values = list(range(1, index + 1)) if xvalue is None else [
                float(row[xvalue]) for row in frame_rows
            ]
            figure, axis = plt.subplots(figsize=(10, 6))
            for value_name, label in zip(graph["values"], graph["labels"]):
                try:
                    y_values = [float(row[value_name]) for row in frame_rows]
                except (KeyError, TypeError, ValueError):
                    continue
                axis.plot(x_values, y_values, marker="o", linewidth=2, label=label)
            axis.set_title(graph["title"])
            axis.set_xlabel(graph["xlabel"] or xvalue or "Batch run")
            axis.grid(True, alpha=0.3)
            if len(graph["values"]) > 1:
                axis.legend()
            frame_path = result_dir / f"{filename}_{index:07d}.png"
            figure.tight_layout()
            figure.savefig(frame_path, dpi=100, bbox_inches="tight")
            plt.close(figure)
            frame_paths.append(frame_path)
        if graph["animated"]:
            _save_gif(frame_paths, gif_path)


def _rebuild_batch_artifacts(metadata, rows):
    """Rebuild root-level batch movies and metagraphs from aggregate rows."""
    result_dir = Path("result")
    result_dir.mkdir(parents=True, exist_ok=True)
    _render_final_graph_movies(metadata, rows, result_dir)
    _render_metagraphs(metadata, rows, result_dir)


def run_batch(csv_path="multi.csv", config_path="config.json", should_cancel=None, progress_callback=None):
    """Run multiple simulations with parameter variants
    
    Args:
        csv_path: multi.csv file with parameter variants (tag in first column)
        config_path: base config.json to merge with variants
        should_cancel: optional callback checked before and during each row
        progress_callback: callback(completed, total, tag, status) for row updates
        
    Returns:
        list of result directories
    """
    config_path = Path(config_path)
    if config_path.exists():
        with config_path.open(encoding="utf-8") as config_file:
            persisted_config = json.load(config_file)
    else:
        persisted_config = {}
    model_name = persisted_config.get("model", DEFAULT_SETTINGS["model"])
    model_metadata = validate_model_metadata(load_model_class(model_name))
    base_config = {
        **DEFAULT_SETTINGS,
        **{name: metadata["default"] for name, metadata in model_metadata["settings"].items()},
        **persisted_config,
    }
    fieldnames, variants = _load_variants(csv_path, base_config)
    active_tags = {row["tag"] for row in variants}
    report_path = Path("result") / "result.csv"
    existing_fields, aggregate_rows = _load_aggregate_rows(report_path, model_name, active_tags)
    final_fields = [
        "year",
        "duration_seconds",
        *[name for name, metadata in model_metadata["values"].items() if metadata["final"]],
    ]
    model_setting_names = list(model_metadata["settings"])
    aggregate_fields = list(dict.fromkeys([
        *existing_fields,
        "model",
        *fieldnames,
        *model_setting_names,
        *final_fields,
    ]))
    completed_tags = {row["tag"] for row in aggregate_rows}
    results_dirs = []
    completed_count = 0
    total_count = len(variants)
    should_cancel = should_cancel or (lambda: False)
    progress_callback = progress_callback or (lambda completed, total, tag, status: None)
    
    for i, variant in enumerate(variants):
        tag = variant["tag"]
        if should_cancel():
            progress_callback(completed_count, total_count, tag, "cancelled")
            break
        if tag in completed_tags:
            results_dirs.append(f"result/{tag}")
            completed_count += 1
            progress_callback(completed_count, total_count, tag, "skipped")
            continue
        progress_callback(completed_count, total_count, tag, "started")
        log(f"\n{'='*60}")
        log(f"Batch run {i+1}/{len(variants)}")
        log(f"{'='*60}")
        
        config = _resolve_row_config(base_config, variant, model_metadata["settings"])
        
        log(f"Running with tag '{tag or 'default'}'")
        log(f"Config: {json.dumps(config, indent=2)[:200]}...")
        
        # Run simulation
        try:
            _, completed = run_simulation(
                config,
                should_cancel=should_cancel,
                return_completion=True,
            )
            if not completed:
                partial_output_dir = Path("result") / config.get("tag", "default")
                shutil.rmtree(partial_output_dir, ignore_errors=True)
                progress_callback(completed_count, total_count, config.get("tag", "default"), "cancelled")
                break
            results_dirs.append(f"result/{config.get('tag', 'default')}")
            final_row = _read_final_row(config["tag"], model_name)
            aggregate_rows.append({
                "model": model_name,
                **{name: config[name] for name in model_setting_names},
                **variant,
                **final_row,
            })
            _write_aggregate_rows(report_path, aggregate_fields, aggregate_rows)
            completed_tags.add(config["tag"])
            _rebuild_batch_artifacts(
                model_metadata,
                _select_active_rows(aggregate_rows, variants),
            )
            completed_count += 1
            progress_callback(completed_count, total_count, config.get("tag", "default"), "completed")
        except Exception as e:
            log(f"Error in batch run {i+1}: {e}")
            progress_callback(completed_count, total_count, config.get("tag", "default"), "failed")
    
    _rebuild_batch_artifacts(
        model_metadata,
        _select_active_rows(aggregate_rows, variants),
    )
    log(f"\n{'='*60}")
    log(f"Batch complete: {len(results_dirs)} simulations finished")
    log(f"Results in: {results_dirs}")
    
    return results_dirs


if __name__ == "__main__":
    import sys
    
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "multi.csv"
    config_file = sys.argv[2] if len(sys.argv) > 2 else "config.json"
    
    run_batch(csv_file, config_file)
