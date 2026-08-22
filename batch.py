"""
Batch simulation runner: execute multiple simulations with parameter sweeps
Loads parameter variations from multi.csv and runs sequentially
"""

import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from main import run_simulation, log, validate_runtime_config
from experiment_manager import ExperimentNotSelectedError, archive_path, resolve_experiment_paths
from load_model import load_model_class
from metadata import validate_model_metadata
from settings import DEFAULT_SETTINGS


class BatchValidationError(ValueError):
    """Report invalid batch rows or incompatible persisted batch results."""


class BatchExecutionError(RuntimeError):
    """Report one or more batch rows that failed during execution."""


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
        config[name] = value
    return config


def _read_final_row(tag, model_name, result_dir):
    """Read and validate one completed row's successful final CSV."""
    final_path = result_dir / tag / "final.csv"
    if not final_path.is_file():
        raise BatchValidationError(f"Completed tag {tag} is missing final.csv")
    with final_path.open(newline="", encoding="utf-8") as final_file:
        rows = list(csv.DictReader(final_file))
    if len(rows) != 1 or rows[0].get("tag") != tag or rows[0].get("model") != model_name:
        raise BatchValidationError(f"Completed tag {tag} has incompatible final.csv")
    return rows[0]


def _load_aggregate_rows(report_path, expected_signatures, active_tags, result_dir):
    """Read and validate prior aggregate rows against current tag models."""
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
        _read_final_row(row["tag"], row["model"], result_dir)
        expected_signature = expected_signatures.get(row["tag"])
        if row["tag"] in active_tags and row.get("config_signature") != expected_signature:
            raise BatchValidationError(f"Completed tag {row['tag']} has a different resolved configuration")
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


def _render_final_graph_movies(metadata, rows, result_dir, output_dir):
    """Collect declared per-tag final graph PNGs into root-level GIF movies."""
    for graph in metadata["graphs"]:
        if not graph["final"]:
            continue
        image_paths = [
            result_dir / row["tag"] / f"{graph['filename']}.png"
            for row in rows
            if (result_dir / row["tag"] / f"{graph['filename']}.png").is_file()
        ]
        _save_gif(image_paths, output_dir / f"{graph['filename']}.gif")


def _render_metagraphs(metadata, rows, output_dir):
    """Render cumulative aggregate plots and optional GIFs for declared metagraphs."""
    for graph in metadata["metagraphs"]:
        filename = graph["filename"]
        for stale_frame in output_dir.glob(f"{filename}_*.png"):
            stale_frame.unlink()
        gif_path = output_dir / f"{filename}.gif"
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
            frame_path = output_dir / f"{filename}_{index:07d}.png"
            figure.tight_layout()
            figure.savefig(frame_path, dpi=100, bbox_inches="tight")
            plt.close(figure)
            frame_paths.append(frame_path)
        if graph["animated"]:
            _save_gif(frame_paths, gif_path)


def _rebuild_batch_artifacts(metadata, rows, result_dir, output_dir=None):
    """Rebuild root-level batch movies and metagraphs from aggregate rows."""
    result_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir or result_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _render_final_graph_movies(metadata, rows, result_dir, output_dir)
    _render_metagraphs(metadata, rows, output_dir)


def _rebuild_all_model_artifacts(metadata_by_model, rows, result_dir):
    """Rebuild batch artifacts separately for each model in a mixed batch."""
    mixed_models = len(metadata_by_model) > 1
    for model_name, metadata in metadata_by_model.items():
        model_rows = [row for row in rows if row.get("model") == model_name]
        output_dir = result_dir / "_models" / model_name if mixed_models else result_dir
        _rebuild_batch_artifacts(metadata, model_rows, result_dir, output_dir)


def _config_signature(config):
    """Return a stable serialized identity for one fully resolved configuration."""
    return json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def run_batch(
    csv_path=None,
    config_path=None,
    should_cancel=None,
    should_finalize=None,
    progress_callback=None,
    graph_callback=None,
    result_dir=None,
    selected_tag=None,
):
    """Run multiple simulations with parameter variants
    
    Args:
        csv_path: multi.csv file with parameter variants (tag in first column)
        config_path: base config.json to merge with variants
        should_cancel: optional callback checked before and during each row
        should_finalize: optional callback that finalizes the current row and batch
        progress_callback: callback(completed, total, tag, status) for row updates
        result_dir: directory containing aggregate and tag-specific results
        selected_tag: optional single validated batch tag to execute
        
    Returns:
        list of result directories
    """
    if csv_path is None or config_path is None:
        paths = resolve_experiment_paths(Path.cwd())
        csv_path = paths["batch_path"] if csv_path is None else csv_path
        config_path = paths["config_path"] if config_path is None else config_path
        result_dir = paths["result_dir"] if result_dir is None else result_dir
    config_path = Path(config_path)
    result_dir = Path(result_dir) if result_dir is not None else config_path.parent / "result"
    if config_path.exists():
        with config_path.open(encoding="utf-8") as config_file:
            persisted_config = json.load(config_file)
    else:
        persisted_config = {}
    base_config = dict(persisted_config)
    fieldnames, variants = _load_variants(csv_path, base_config)
    resolved_rows = []
    metadata_by_model = {}
    for variant in variants:
        model_name = variant.get("model") or base_config.get("model")
        if not model_name:
            raise BatchValidationError(f"Batch tag {variant.get('tag', '')} has no model")
        model_metadata = metadata_by_model.setdefault(
            model_name,
            validate_model_metadata(load_model_class(model_name)),
        )
        config = _resolve_row_config(base_config, variant, model_metadata["settings"])
        validate_runtime_config(config)
        resolved_rows.append((variant, config, model_metadata))
    execution_variants = variants
    if selected_tag is not None:
        execution_variants = [variant for variant in variants if variant["tag"] == selected_tag]
        if not execution_variants:
            raise BatchValidationError(f"Unknown selected batch tag: {selected_tag}")
    active_tags = {row["tag"] for row in variants}
    expected_signatures = {
        config["tag"]: _config_signature(config)
        for _, config, _ in resolved_rows
    }
    report_path = result_dir / "result.csv"
    existing_fields, aggregate_rows = _load_aggregate_rows(
        report_path,
        expected_signatures,
        active_tags,
        result_dir,
    )
    final_fields = ["year", "duration_seconds"]
    for metadata in metadata_by_model.values():
        final_fields.extend(name for name, details in metadata["values"].items() if details["final"])
    resolved_setting_names = list(dict.fromkeys(
        name for _, config, _ in resolved_rows for name in config
    ))
    aggregate_fields = list(dict.fromkeys([
        *existing_fields,
        "model",
        "config_signature",
        *fieldnames,
        *[f"input_{name}" for name in fieldnames],
        *resolved_setting_names,
        *final_fields,
    ]))
    completed_tags = {row["tag"] for row in aggregate_rows}
    results_dirs = []
    completed_count = 0
    total_count = len(execution_variants)
    should_cancel = should_cancel or (lambda: False)
    should_finalize = should_finalize or (lambda: False)
    progress_callback = progress_callback or (lambda completed, total, tag, status: None)
    graph_callback = graph_callback or (lambda output_dir, year: None)
    failed_rows = []
    
    resolved_by_tag = {variant["tag"]: (config, metadata) for variant, config, metadata in resolved_rows}
    for i, variant in enumerate(execution_variants):
        tag = variant["tag"]
        if should_cancel():
            progress_callback(completed_count, total_count, tag, "cancelled")
            break
        if tag in completed_tags:
            results_dirs.append((result_dir / tag).as_posix())
            completed_count += 1
            progress_callback(completed_count, total_count, tag, "skipped")
            continue
        progress_callback(completed_count, total_count, tag, "started")
        log(f"\n{'='*60}")
        log(f"Batch run {i+1}/{len(execution_variants)}")
        log(f"{'='*60}")
        
        config, model_metadata = resolved_by_tag[tag]
        model_name = config["model"]
        tag_result_dir = result_dir / config.get("tag", "default")
        if tag_result_dir.exists():
            archive_path(tag_result_dir)
        
        log(f"Running with tag '{tag or 'default'}'")
        log(f"Config: {json.dumps(config, indent=2)[:200]}...")
        
        # Run simulation
        try:
            row_finalized = False

            def finalize_current_row():
                """Record and return a successful-finalization request for this row."""
                nonlocal row_finalized
                row_finalized = should_finalize()
                return row_finalized

            _, completed = run_simulation(
                config,
                should_cancel=should_cancel,
                should_finalize=finalize_current_row,
                return_completion=True,
                graph_callback=graph_callback,
                result_root=result_dir,
            )
            if not completed:
                partial_output_dir = result_dir / config.get("tag", "default")
                try:
                    archive_path(partial_output_dir)
                except OSError as archive_error:
                    log(
                        f"Could not archive cancelled output {partial_output_dir}: "
                        f"{archive_error}. It will be archived before the next run."
                    )
                progress_callback(completed_count, total_count, config.get("tag", "default"), "cancelled")
                break
            final_row = _read_final_row(config["tag"], model_name, result_dir)
            aggregate_row = {
                "model": model_name,
                "config_signature": _config_signature(config),
                **config,
                **{f"input_{name}": value for name, value in variant.items()},
                **final_row,
            }
            pending_rows = [*aggregate_rows, aggregate_row]
            _rebuild_all_model_artifacts(
                metadata_by_model,
                _select_active_rows(pending_rows, variants),
                result_dir,
            )
            _write_aggregate_rows(report_path, aggregate_fields, pending_rows)
            aggregate_rows = pending_rows
            completed_tags.add(config["tag"])
            results_dirs.append((result_dir / config.get("tag", "default")).as_posix())
            completed_count += 1
            row_finalized = row_finalized or should_finalize()
            status = "finalized" if row_finalized else "completed"
            progress_callback(completed_count, total_count, config.get("tag", "default"), status)
            if row_finalized:
                break
        except Exception as e:
            failed_output_dir = result_dir / config.get("tag", "default")
            try:
                archive_path(failed_output_dir)
            except OSError as archive_error:
                log(f"Could not archive failed output {failed_output_dir}: {archive_error}")
            log(f"Error in batch run {i+1}: {e}")
            progress_callback(completed_count, total_count, config.get("tag", "default"), "failed")
            failed_rows.append((config.get("tag", "default"), str(e)))
    
    if not failed_rows:
        _rebuild_all_model_artifacts(
            metadata_by_model,
            _select_active_rows(aggregate_rows, variants),
            result_dir,
        )
    log(f"\n{'='*60}")
    log(f"Batch complete: {len(results_dirs)} simulations finished")
    log(f"Results in: {results_dirs}")

    if failed_rows:
        failure_text = "; ".join(f"{tag}: {message}" for tag, message in failed_rows)
        raise BatchExecutionError(f"Batch rows failed: {failure_text}")
    
    return results_dirs


if __name__ == "__main__":
    import sys
    
    csv_file = sys.argv[1] if len(sys.argv) > 1 else None
    config_file = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        run_batch(csv_file, config_file)
    except ExperimentNotSelectedError as error:
        raise SystemExit(str(error)) from error
