import csv
import json
from pathlib import Path

import pytest
from PIL import Image

import batch as batch_module
from batch import run_batch
from settings import DEFAULT_SETTINGS


def make_batch_config():
    """Return settings that finish each batch row after one year."""
    return {
        **DEFAULT_SETTINGS,
        "device": "cpu",
        "seed": 97531,
        "max_population": 100,
        "initial_population": 1,
        "initial_age_max": 0,
        "alpha": 0.0,
        "lambda": 0.0,
        "mature_age": 12,
        "stat_generation_period": 1,
        "graph_generation_period": 1000,
        "max_iterations": 100,
    }


@pytest.mark.smoke
def test_metagraph_filter_last_and_range_options(tmp_path, monkeypatch):
    """Filter aggregate rows, render one final frame, and clamp configured axes."""
    captured = []
    monkeypatch.setattr(
        batch_module,
        "render_series",
        lambda axis, x_values, y_values, *args, **kwargs: captured.append((x_values, y_values)),
    )
    metadata = {
        "metagraphs": [{
            "filename": "filtered",
            "values": ["avg_beta"],
            "labels": ["Average beta"],
            "title": "Filtered",
            "xlabel": "mutation_x",
            "xvalue": "mutation_x",
            "animated": True,
            "style": "lines",
            "max_point_size": 200.0,
            "last": True,
            "filter": {"lambda": [0.0, 0.1]},
            "range": {"mutation_x": [0.0, 1.0], "avg_beta": [0.0, 1.0]},
        }],
    }
    rows = [
        {"tag": "kept_a", "mutation_x": "-1", "lambda": "0.05", "avg_beta": "-2"},
        {"tag": "dropped", "mutation_x": "0.5", "lambda": "0.2", "avg_beta": "0.5"},
        {"tag": "kept_b", "mutation_x": "2", "lambda": "0.05", "avg_beta": "2"},
    ]

    batch_module._render_metagraphs(metadata, rows, tmp_path)

    assert captured == [([0.0, 1.0], [0.0, 1.0])]
    assert (tmp_path / "filtered.png").is_file()
    assert not list(tmp_path.glob("filtered_*.png"))



@pytest.mark.smoke
def test_annotate_metagraph_points_labels_every_point():
    """Label every metagraph point with its data_label value."""
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots()
    graph = {"values": ["avg_beta"]}
    frame_rows = [{"tag": "first", "avg_beta": "0.1"}, {"tag": "second", "avg_beta": "0.2"}]

    batch_module._annotate_metagraph_points(axis, graph, frame_rows, [1, 2], "tag")

    assert [text.get_text() for text in axis.texts] == ["first", "second"]
    plt.close(fig)


@pytest.mark.smoke
def test_annotate_metagraph_points_skips_blank_or_unparseable_values():
    """Add no annotation for a point whose label or plotted value is unusable."""
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots()
    graph = {"values": ["avg_beta"]}
    frame_rows = [
        {"tag": "", "avg_beta": "0.1"},
        {"tag": "second", "avg_beta": "not-a-number"},
    ]

    batch_module._annotate_metagraph_points(axis, graph, frame_rows, [1, 2], "tag")

    assert len(axis.texts) == 0
    plt.close(fig)


@pytest.mark.smoke
def test_batch_runs_each_csv_tag_and_can_repeat(tmp_path, monkeypatch):
    """Execute and repeat a short v1 batch for both tagged result folders."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text(
        json.dumps(make_batch_config()),
        encoding="utf-8",
    )
    with (tmp_path / "multi.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["tag", "beta_initial"])
        writer.writeheader()
        writer.writerows([
            {"tag": "batch_a", "beta_initial": "0.10"},
            {"tag": "batch_b", "beta_initial": "0.20"},
        ])

    result_dirs = run_batch("multi.csv", "config.json")
    repeated_result_dirs = run_batch("multi.csv", "config.json")

    assert result_dirs == ["result/batch_a", "result/batch_b"]
    assert repeated_result_dirs == result_dirs
    for tag in ("batch_a", "batch_b"):
        assert (tmp_path / "result" / tag / "result.csv").is_file()
        assert (tmp_path / "result" / tag / "final.csv").is_file()
        assert (tmp_path / "result" / tag / "results_summary.png").is_file()


@pytest.mark.smoke
def test_batch_stops_after_current_row_when_cancel_requested(tmp_path, monkeypatch):
    """Report progress and stop before launching the next requested batch row."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(make_batch_config()), encoding="utf-8")
    (tmp_path / "multi.csv").write_text(
        "tag,beta_initial\nbatch_a,0.10\nbatch_b,0.20\n",
        encoding="utf-8",
    )
    events = []
    cancel_requested = False

    def on_progress(completed, total, tag, status):
        """Request cancellation after the first completed row."""
        nonlocal cancel_requested
        events.append((completed, total, tag, status))
        if status == "completed":
            cancel_requested = True

    result_dirs = run_batch(
        "multi.csv",
        "config.json",
        should_cancel=lambda: cancel_requested,
        progress_callback=on_progress,
    )

    assert result_dirs == ["result/batch_a"]
    assert (tmp_path / "result" / "batch_a" / "final.csv").is_file()
    assert not (tmp_path / "result" / "batch_b").exists()
    assert events[-1] == (1, 2, "batch_b", "cancelled")


@pytest.mark.smoke
def test_batch_finalize_completes_current_row_and_stops_batch(tmp_path, monkeypatch):
    """Persist the finalized current row without launching the next batch row."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(make_batch_config()), encoding="utf-8")
    (tmp_path / "multi.csv").write_text(
        "tag,beta_initial\nbatch_a,0.10\nbatch_b,0.20\n",
        encoding="utf-8",
    )
    events = []

    result_dirs = run_batch(
        "multi.csv",
        "config.json",
        should_finalize=lambda: True,
        progress_callback=lambda *event: events.append(event),
    )

    assert result_dirs == ["result/batch_a"]
    assert (tmp_path / "result" / "batch_a" / "final.csv").is_file()
    assert not (tmp_path / "result" / "batch_b").exists()
    assert events[-1] == (1, 2, "batch_a", "finalized")


@pytest.mark.smoke
def test_batch_archives_partial_current_row_after_cancellation(tmp_path, monkeypatch):
    """Archive partial files when cancellation interrupts the current row."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(make_batch_config()), encoding="utf-8")
    (tmp_path / "multi.csv").write_text("tag\nbatch_a\n", encoding="utf-8")
    cancel_checks = 0

    def should_cancel():
        """Allow batch row setup then cancel at the first simulation year."""
        nonlocal cancel_checks
        cancel_checks += 1
        return cancel_checks >= 2

    result_dirs = run_batch("multi.csv", "config.json", should_cancel=should_cancel)

    assert result_dirs == []
    assert not (tmp_path / "result" / "batch_a").exists()
    assert len(list((tmp_path / "result").glob("batch_a_*.bak"))) == 1


@pytest.mark.smoke
def test_batch_cancellation_tolerates_temporarily_locked_partial_output(tmp_path, monkeypatch):
    """Do not report a batch failure when Windows temporarily blocks partial archival."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(make_batch_config()), encoding="utf-8")
    (tmp_path / "multi.csv").write_text("tag\nbatch_a\n", encoding="utf-8")
    events = []

    def cancelled_simulation(config, **kwargs):
        """Create partial output and report cooperative cancellation."""
        output_dir = Path(kwargs["result_root"]) / config["tag"]
        output_dir.mkdir(parents=True)
        (output_dir / "result.csv").write_text("partial", encoding="utf-8")
        return [], False

    monkeypatch.setattr(batch_module, "run_simulation", cancelled_simulation)
    monkeypatch.setattr(
        batch_module,
        "archive_path",
        lambda path: (_ for _ in ()).throw(PermissionError(5, "Access is denied", str(path))),
    )

    result_dirs = run_batch(
        "multi.csv",
        "config.json",
        progress_callback=lambda *event: events.append(event),
    )

    assert result_dirs == []
    assert events[-1] == (0, 1, "batch_a", "cancelled")
    assert (tmp_path / "result" / "batch_a" / "result.csv").is_file()


@pytest.mark.smoke
def test_batch_writes_aggregate_report_and_skips_completed_tags(tmp_path, monkeypatch):
    """Append final values to the aggregate report and resume by matching tags."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(make_batch_config()), encoding="utf-8")
    (tmp_path / "multi.csv").write_text(
        "tag,beta_initial\nbatch_a,0.10\nbatch_b,0.20\n",
        encoding="utf-8",
    )

    first_result_dirs = run_batch("multi.csv", "config.json")
    with (tmp_path / "result" / "result.csv").open(newline="") as report_file:
        report_rows = list(csv.DictReader(report_file))
    monkeypatch.setattr(
        batch_module,
        "run_simulation",
        lambda *args, **kwargs: pytest.fail("completed tag should be skipped"),
    )

    resumed_result_dirs = run_batch("multi.csv", "config.json")

    assert first_result_dirs == ["result/batch_a", "result/batch_b"]
    assert resumed_result_dirs == first_result_dirs
    assert [row["tag"] for row in report_rows] == ["batch_a", "batch_b"]
    assert all(row["model"] == "model_base" for row in report_rows)
    assert [row["beta_initial"] for row in report_rows] == ["0.1", "0.2"]
    assert [row["input_beta_initial"] for row in report_rows] == ["0.10", "0.20"]
    assert all(row["avg_beta"] for row in report_rows)
    assert all(row["max_population"] == "100" for row in report_rows)
    assert all(row["mutation_probability"] == "0.1" for row in report_rows)


@pytest.mark.smoke
def test_batch_builds_final_graph_movie_and_cumulative_metagraph(tmp_path, monkeypatch):
    """Render root-level final graph GIFs and cumulative declared metagraph frames."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(make_batch_config()), encoding="utf-8")
    (tmp_path / "multi.csv").write_text(
        "tag,mutation_x\nbatch_a,0.10\nbatch_b,0.20\n",
        encoding="utf-8",
    )

    run_batch("multi.csv", "config.json")

    result_dir = tmp_path / "result"
    assert (result_dir / "beta_evolution.gif").is_file()
    assert (result_dir / "beta_by_mutation.png").is_file()
    assert not list(result_dir.glob("beta_by_mutation_*.png"))
    assert not (result_dir / "beta_by_mutation.gif").exists()


@pytest.mark.smoke
def test_batch_rebuilds_root_artifacts_for_current_csv_tags_only(tmp_path, monkeypatch):
    """Exclude stale aggregate rows when the active batch CSV removes a tag."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(make_batch_config()), encoding="utf-8")
    batch_path = tmp_path / "multi.csv"
    batch_path.write_text(
        "tag,mutation_x\nbatch_a,0.10\nbatch_b,0.20\n",
        encoding="utf-8",
    )
    run_batch("multi.csv", "config.json")
    batch_path.write_text("tag,mutation_x\nbatch_b,0.20\n", encoding="utf-8")
    monkeypatch.setattr(
        batch_module,
        "run_simulation",
        lambda *args, **kwargs: pytest.fail("completed tag should be skipped"),
    )

    run_batch("multi.csv", "config.json")

    result_dir = tmp_path / "result"
    assert (result_dir / "beta_by_mutation.png").is_file()
    assert not list(result_dir.glob("beta_by_mutation_*.png"))
    assert not (result_dir / "beta_by_mutation.gif").exists()


@pytest.mark.smoke
def test_batch_rejects_duplicate_tags(tmp_path, monkeypatch):
    """Reject duplicate batch identities before starting any calculation."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(make_batch_config()), encoding="utf-8")
    (tmp_path / "multi.csv").write_text(
        "tag,mutation_x\nbatch_a,0.1\nbatch_a,0.2\n",
        encoding="utf-8",
    )

    with pytest.raises(batch_module.BatchValidationError):
        run_batch("multi.csv", "config.json")

    assert not (tmp_path / "result").exists()


@pytest.mark.smoke
def test_batch_rejects_aggregate_tag_without_matching_final_file(tmp_path, monkeypatch):
    """Do not treat an aggregate row as completed when final.csv is absent."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(make_batch_config()), encoding="utf-8")
    (tmp_path / "multi.csv").write_text("tag\nbatch_a\n", encoding="utf-8")
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    (result_dir / "result.csv").write_text("model,tag\nmodel_base,batch_a\n", encoding="utf-8")

    with pytest.raises(batch_module.BatchValidationError, match="final.csv"):
        run_batch("multi.csv", "config.json")


@pytest.mark.smoke
def test_batch_uses_active_experiment_paths_when_called_without_arguments(tmp_path, monkeypatch):
    """Resolve config, batch, and results inside the active experiment."""
    monkeypatch.chdir(tmp_path)
    experiment_dir = tmp_path / "data" / "experiment_alpha"
    experiment_dir.mkdir(parents=True)
    (tmp_path / "default.conf").write_text("experiment_alpha", encoding="utf-8")
    (experiment_dir / "config.json").write_text(
        json.dumps(make_batch_config()),
        encoding="utf-8",
    )
    (experiment_dir / "multi.csv").write_text("tag\nbatch_a\n", encoding="utf-8")

    result_dirs = run_batch()

    expected_result_dir = experiment_dir / "result" / "batch_a"
    assert result_dirs == [expected_result_dir.as_posix()]
    assert (expected_result_dir / "final.csv").is_file()
    assert (experiment_dir / "result" / "result.csv").is_file()
    assert not (tmp_path / "result").exists()


@pytest.mark.smoke
def test_batch_supports_per_row_models_and_preserves_input_values(tmp_path, monkeypatch):
    """Resolve and report each batch row using its selected model metadata."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(make_batch_config()), encoding="utf-8")
    (tmp_path / "multi.csv").write_text(
        "tag,model,mutation_x\nbase,model_base,0.10\nfast,model_base_fast,0.20\n",
        encoding="utf-8",
    )

    run_batch("multi.csv", "config.json")

    with (tmp_path / "result" / "result.csv").open(newline="", encoding="utf-8") as report_file:
        rows = list(csv.DictReader(report_file))
    assert [row["model"] for row in rows] == ["model_base", "model_base_fast"]
    assert [row["input_model"] for row in rows] == ["model_base", "model_base_fast"]
    assert [row["input_mutation_x"] for row in rows] == ["0.10", "0.20"]
    assert (tmp_path / "result" / "_models" / "model_base").is_dir()
    assert (tmp_path / "result" / "_models" / "model_base_fast").is_dir()


@pytest.mark.smoke
def test_batch_warns_and_recalculates_changed_completed_tag(tmp_path, monkeypatch):
    """Archive and replace a completed tag whose resolved configuration changed."""
    monkeypatch.chdir(tmp_path)
    messages = []
    (tmp_path / "config.json").write_text(json.dumps(make_batch_config()), encoding="utf-8")
    batch_path = tmp_path / "multi.csv"
    batch_path.write_text("tag,mutation_x\nbatch_a,0.10\n", encoding="utf-8")
    run_batch("multi.csv", "config.json")
    batch_path.write_text("tag,mutation_x\nbatch_a,0.20\n", encoding="utf-8")
    monkeypatch.setattr(batch_module, "log", lambda message: messages.append(message))

    result_dirs = run_batch("multi.csv", "config.json")

    with (tmp_path / "result" / "result.csv").open(newline="", encoding="utf-8") as report_file:
        rows = list(csv.DictReader(report_file))
    assert result_dirs == ["result/batch_a"]
    assert len(rows) == 1
    assert rows[0]["tag"] == "batch_a"
    assert rows[0]["mutation_x"] == "0.2"
    assert any("mutation_x: 0.1 -> 0.2" in message for message in messages)
    assert list((tmp_path / "result").glob("batch_a_*.bak"))


@pytest.mark.smoke
def test_batch_recalculates_only_changed_completed_tags(tmp_path, monkeypatch):
    """Skip matching completed tags while replacing only changed configurations."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(make_batch_config()), encoding="utf-8")
    batch_path = tmp_path / "multi.csv"
    batch_path.write_text(
        "tag,mutation_x\nunchanged,0.10\nchanged,0.20\n",
        encoding="utf-8",
    )
    run_batch("multi.csv", "config.json")
    batch_path.write_text(
        "tag,mutation_x\nunchanged,0.10\nchanged,0.30\n",
        encoding="utf-8",
    )
    executed_tags = []
    original_run_simulation = batch_module.run_simulation

    def record_run(config, *args, **kwargs):
        """Record recalculated tags and delegate to the simulation runner."""
        executed_tags.append(config["tag"])
        return original_run_simulation(config, *args, **kwargs)

    monkeypatch.setattr(batch_module, "run_simulation", record_run)

    run_batch("multi.csv", "config.json")

    assert executed_tags == ["changed"]
    with (tmp_path / "result" / "result.csv").open(newline="", encoding="utf-8") as report_file:
        rows = list(csv.DictReader(report_file))
    assert [(row["tag"], row["mutation_x"]) for row in rows] == [
        ("unchanged", "0.1"),
        ("changed", "0.3"),
    ]


@pytest.mark.smoke
def test_batch_keeps_old_results_for_tags_marked_keep(tmp_path, monkeypatch):
    """Skip recalculation and preserve the stale row for a tag marked to keep."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(make_batch_config()), encoding="utf-8")
    batch_path = tmp_path / "multi.csv"
    batch_path.write_text("tag,mutation_x\nchanged,0.10\n", encoding="utf-8")
    run_batch("multi.csv", "config.json")
    original_result_file = (tmp_path / "result" / "changed" / "result.csv").read_bytes()
    batch_path.write_text("tag,mutation_x\nchanged,0.20\n", encoding="utf-8")
    monkeypatch.setattr(
        batch_module,
        "run_simulation",
        lambda *args, **kwargs: pytest.fail("kept tag should not be recalculated"),
    )

    result_dirs = run_batch("multi.csv", "config.json", keep_changed_tags={"changed"})

    assert result_dirs == ["result/changed"]
    assert (tmp_path / "result" / "changed" / "result.csv").read_bytes() == original_result_file
    with (tmp_path / "result" / "result.csv").open(newline="", encoding="utf-8") as report_file:
        rows = list(csv.DictReader(report_file))
    assert [(row["tag"], row["mutation_x"]) for row in rows] == [("changed", "0.1")]


@pytest.mark.smoke
def test_batch_change_preflight_limits_report_to_selected_tag(tmp_path, monkeypatch):
    """Validate the whole batch but report configuration changes for one selected row."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(make_batch_config()), encoding="utf-8")
    batch_path = tmp_path / "multi.csv"
    batch_path.write_text(
        "tag,mutation_x\nfirst,0.10\nsecond,0.20\n",
        encoding="utf-8",
    )
    run_batch("multi.csv", "config.json")
    batch_path.write_text(
        "tag,mutation_x\nfirst,0.30\nsecond,0.40\n",
        encoding="utf-8",
    )

    changes = batch_module.inspect_batch_configuration_changes(
        "multi.csv",
        "config.json",
        selected_tag="second",
    )

    assert [change["tag"] for change in changes] == ["second"]
    assert changes[0]["changed_values"]["mutation_x"] == (0.2, 0.4)


@pytest.mark.smoke
def test_batch_rejects_out_of_range_csv_without_creating_results(tmp_path, monkeypatch):
    """Reject invalid CSV values instead of silently clamping them."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(make_batch_config()), encoding="utf-8")
    (tmp_path / "multi.csv").write_text(
        "tag,mutation_probability\nbatch_a,2.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mutation_probability"):
        run_batch("multi.csv", "config.json")

    assert not (tmp_path / "result").exists()


@pytest.mark.smoke
def test_batch_raises_when_row_execution_fails(tmp_path, monkeypatch):
    """Expose row failures to CLI and GUI callers instead of returning success."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(make_batch_config()), encoding="utf-8")
    (tmp_path / "multi.csv").write_text("tag\nbatch_a\n", encoding="utf-8")
    monkeypatch.setattr(batch_module, "run_simulation", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(batch_module.BatchExecutionError, match="batch_a: boom"):
        run_batch("multi.csv", "config.json")

    assert not (tmp_path / "result" / "result.csv").exists()


@pytest.mark.smoke
def test_selected_batch_row_still_validates_full_csv(tmp_path, monkeypatch):
    """Reject duplicates outside the selected row before running anything."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(make_batch_config()), encoding="utf-8")
    (tmp_path / "multi.csv").write_text(
        "tag,mutation_x\nselected,0.1\nduplicate,0.2\nduplicate,0.3\n",
        encoding="utf-8",
    )

    with pytest.raises(batch_module.BatchValidationError, match="unique"):
        run_batch("multi.csv", "config.json", selected_tag="selected")

    assert not (tmp_path / "result").exists()


@pytest.mark.smoke
def test_batch_warns_and_runs_identical_parameter_rows(tmp_path, monkeypatch):
    """Continue CLI execution after listing tags with identical raw parameters."""
    monkeypatch.chdir(tmp_path)
    messages = []
    (tmp_path / "config.json").write_text(json.dumps(make_batch_config()), encoding="utf-8")
    (tmp_path / "multi.csv").write_text(
        "tag,mutation_x\nbatch_a,0.1\nbatch_b,0.1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(batch_module, "log", lambda message: messages.append(message))

    result_dirs = run_batch("multi.csv", "config.json")

    assert len(result_dirs) == 2
    assert any("batch_a, batch_b" in message for message in messages)