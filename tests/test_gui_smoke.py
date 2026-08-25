import tkinter as tk
import json
from pathlib import Path

import pytest
from PIL import Image

import gui as gui_module
from gui import SimulationGUI
from main import set_logger
from model import Model
from metadata import validate_model_metadata


def _stub_modal_dialog(gui_app, monkeypatch, interact):
    """Answer the next grab_set/wait_window dialog in-process, never on screen."""
    def immediate_wait_window(window):
        """Interact with the dialog synchronously instead of pumping a live event loop."""
        window.update_idletasks()
        interact(window)

    monkeypatch.setattr(gui_app.root, "wait_window", immediate_wait_window)


def _dialog_descendants(widget):
    """Return every nested Tk child below one widget."""
    children = list(widget.winfo_children())
    return children + [nested for child in children for nested in _dialog_descendants(child)]


@pytest.mark.smoke
@pytest.mark.parametrize("content", ["not json", "[]"])
def test_load_config_rejects_invalid_documents(tmp_path, content):
    """Reject malformed JSON and JSON values that are not objects."""
    config_path = tmp_path / "invalid.json"
    config_path.write_text(content)
    application = object.__new__(SimulationGUI)
    application.config_file = str(config_path)

    with pytest.raises((json.JSONDecodeError, ValueError)):
        application._load_config()


@pytest.mark.smoke
def test_model_selection_normalizes_declared_settings_without_removing_inactive_keys():
    """Add missing defaults and clamp retained model settings on selection."""
    config, metadata, corrected_names = gui_module._prepare_model_configuration(
        {
            "model": "model_base",
            "max_population": 1,
            "inactive_model_setting": "keep",
        },
        "model_base",
    )

    assert config["model"] == "model_base"
    assert config["max_population"] == 100
    assert config["seed"] == 0
    assert config["inactive_model_setting"] == "keep"
    assert metadata["settings"]["max_population"]["min"] == 100
    assert corrected_names == ["max_population"]


@pytest.mark.smoke
def test_lightweight_markdown_view_renders_supported_elements(gui_app):
    """Render headings, bullets, bold, code, and formulas without marker text."""
    view = gui_module.LightweightMarkdownView(gui_app.root)
    view.set_markdown(
        "# Model title\n\n## Purpose\n\n"
        "A **bold** value and `setting`.\n\n"
        "- First item\n\n$$value = exp(beta * age)$$"
    )

    rendered = view.text.get("1.0", "end-1c")
    assert "Model title" in rendered
    assert "First item" in rendered
    assert "value = exp(beta * age)" in rendered
    assert "**" not in rendered
    assert "`" not in rendered
    assert str(view.text.cget("state")) == "disabled"
    for tag_name in ("heading1", "heading2", "bold", "code", "formula", "bullet"):
        assert view.text.tag_ranges(tag_name)
    view.destroy()


@pytest.fixture
def gui_app(tmp_path, monkeypatch):
    """Create and destroy a hidden GUI rooted in a temporary directory."""
    monkeypatch.chdir(tmp_path)
    try:
        root = tk.Tk()
    except tk.TclError as error:
        pytest.skip(f"Tk is unavailable in this environment: {error}")

    root.withdraw()
    application = SimulationGUI(root)
    yield application
    set_logger(None)
    root.destroy()


@pytest.mark.smoke
def test_gui_bootstraps_active_experiment_from_existing_data_dir(tmp_path, monkeypatch):
    """Select the first experiment directory and write default.conf when it is missing."""
    monkeypatch.chdir(tmp_path)
    experiment_dir = tmp_path / "data" / "exp_alpha"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "config.json").write_text(json.dumps({"model": "model_base"}), encoding="utf-8")

    try:
        root = tk.Tk()
    except tk.TclError as error:
        pytest.skip(f"Tk is unavailable in this environment: {error}")

    root.withdraw()
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "gui_bootstrap")
    application = SimulationGUI(root)
    try:
        assert (tmp_path / "default.conf").read_text(encoding="utf-8").strip() == "exp_alpha"
        assert Path(application.config_file).resolve() == (experiment_dir / "config.json").resolve()
    finally:
        set_logger(None)
        root.destroy()


@pytest.mark.smoke
def test_gui_switches_to_valid_experiment_before_updating_default(gui_app, tmp_path):
    """Load target files and then persist the newly active experiment."""
    experiment_dir = tmp_path / "data" / "exp_beta"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "config.json").write_text(
        json.dumps({"model": "model_base", "tag": "beta_tag"}),
        encoding="utf-8",
    )
    (experiment_dir / "multi.csv").write_text("tag\nbeta_batch\n", encoding="utf-8")
    gui_app.experiment_selector.configure(values=["exp_beta"])
    gui_app.experiment_var.set("exp_beta")

    gui_app._on_experiment_selected()

    assert gui_app.experiment_dir == experiment_dir
    assert Path(gui_app.config_file) == experiment_dir / "config.json"
    assert gui_app.batch_path == experiment_dir / "multi.csv"
    assert gui_app.config["tag"] == "beta_tag"
    assert (tmp_path / "default.conf").read_text(encoding="utf-8") == "exp_beta"


@pytest.mark.smoke
def test_gui_cancels_dirty_experiment_switch_without_changing_default(gui_app, tmp_path, monkeypatch):
    """Keep the active experiment unchanged when the dirty-state prompt is cancelled."""
    current_dir = gui_app.experiment_manager.set_active_experiment("exp_alpha")
    gui_app.experiment_dir = current_dir
    gui_app.config_file = str(current_dir / "config.json")
    gui_app.is_config_dirty = True
    gui_app.experiment_var.set("exp_beta")
    monkeypatch.setattr(gui_module.messagebox, "askyesnocancel", lambda *args: None)

    gui_app._on_experiment_selected()

    assert gui_app.experiment_var.get() == "exp_alpha"
    assert gui_app.experiment_manager.get_active_experiment_name() == "exp_alpha"
    assert gui_app.experiment_dir == current_dir


@pytest.mark.smoke
def test_gui_cancels_new_experiment_when_dirty_transition_is_cancelled(gui_app, monkeypatch):
    """Do not prompt for or create an experiment after cancelling dirty-state resolution."""
    prompts = []
    gui_app.is_config_dirty = True
    monkeypatch.setattr(gui_module.messagebox, "askyesnocancel", lambda *args: None)
    monkeypatch.setattr(gui_app, "_prompt_new_experiment", lambda: prompts.append(True))

    gui_app._on_new_experiment()

    assert prompts == []


@pytest.mark.smoke
def test_gui_does_not_prompt_for_clone_when_busy_or_dirty_transition_is_cancelled(
    gui_app,
    monkeypatch,
):
    """Keep clone unavailable during calculations and after cancelling dirty resolution."""
    gui_app._create_experiment("source", "model_base")
    gui_app._activate_experiment("source")
    prompts = []
    gui_app.is_config_dirty = True
    monkeypatch.setattr(gui_module.messagebox, "askyesnocancel", lambda *args: None)
    monkeypatch.setattr(
        gui_app,
        "_prompt_clone_experiment",
        lambda source_name: prompts.append(source_name),
    )

    gui_app._on_clone_experiment()
    gui_app.is_config_dirty = False
    gui_app.is_running = True
    monkeypatch.setattr(gui_module.messagebox, "showwarning", lambda *args: None)
    gui_app._on_clone_experiment()

    assert prompts == []


@pytest.mark.smoke
def test_gui_constructs_and_updates_valid_settings(gui_app):
    """Construct the GUI and transfer valid controls into memory."""
    gui_app.model_setting_rows["max_population"]["value"].set("500")
    gui_app.tag_var.set("gui_smoke")

    updated = gui_app._update_config_from_ui()

    assert updated is True
    assert gui_app.config["max_population"] == 500
    assert gui_app.config["tag"] == "gui_smoke"


@pytest.mark.smoke
def test_gui_update_from_ui_does_not_mark_dirty_without_edits(gui_app):
    """Reading unchanged controls back must never mark the configuration dirty."""
    assert gui_app.is_config_dirty is False

    gui_app._update_config_from_ui()

    assert gui_app.is_config_dirty is False


@pytest.mark.smoke
def test_gui_uses_persistent_non_modal_progress_window(gui_app):
    """Keep Progress outside the main tabs and preserve it across show and hide actions."""
    assert "Progress" not in [gui_app.notebook.tab(tab_id, "text") for tab_id in gui_app.notebook.tabs()]
    assert gui_app.progress_window.state() == "withdrawn"

    gui_app._show_progress_window()
    gui_app.root.update_idletasks()

    assert gui_app.progress_window.state() != "withdrawn"
    gui_app._hide_progress_window()
    assert gui_app.progress_window.state() == "withdrawn"
    assert gui_app.progress_window.winfo_exists() == 1


@pytest.mark.smoke
def test_gui_simulation_launch_shows_progress_and_enables_both_stop_buttons(gui_app, monkeypatch):
    """Open Progress automatically and synchronize its stop control at launch."""
    thread_starts = []

    class FakeThread:
        """Record a requested worker start without running a simulation."""

        def __init__(self, target, daemon):
            """Store the requested worker attributes."""
            self.target = target
            self.daemon = daemon

        def start(self):
            """Record that the GUI requested the worker start."""
            thread_starts.append(True)

    monkeypatch.setattr(gui_module.threading, "Thread", FakeThread)

    gui_app._launch_simulation()
    gui_app.root.update_idletasks()

    assert gui_app.progress_window.state() != "withdrawn"
    assert str(gui_app.stop_btn.cget("state")) == "normal"
    assert str(gui_app.progress_stop_btn.cget("state")) == "normal"
    assert str(gui_app.progress_finalize_btn.cget("state")) == "normal"
    assert thread_starts == [True]

    gui_app._stop_simulation()

    assert str(gui_app.stop_btn.cget("state")) == "disabled"
    assert str(gui_app.progress_stop_btn.cget("state")) == "disabled"
    assert str(gui_app.progress_finalize_btn.cget("state")) == "disabled"


@pytest.mark.smoke
def test_gui_progress_finalize_requests_successful_batch_stop(gui_app, monkeypatch):
    """Request current-row finalization and disable repeated requests."""
    monkeypatch.setattr(gui_module.threading.Thread, "start", lambda self: None)

    gui_app._launch_batch("selected")
    gui_app.progress_finalize_btn.invoke()

    assert gui_app.batch_finalize_requested is True
    assert gui_app.batch_cancel_requested is False
    assert gui_app.batch_status_var.get() == "Finalization requested"
    assert gui_app._consume_finalize_request() is True
    assert gui_app._consume_finalize_request() is False
    assert str(gui_app.progress_finalize_btn.cget("state")) == "disabled"


@pytest.mark.smoke
def test_gui_progress_displays_batch_performance(gui_app, monkeypatch):
    """Display nonzero speed statistics reported by the active batch row."""
    def fake_run_batch(*args, **kwargs):
        """Report one deterministic batch performance snapshot."""
        kwargs["performance_callback"](2.0, 4, 1000)

    monkeypatch.setattr(gui_module, "run_batch", fake_run_batch)
    monkeypatch.setattr(
        gui_app.root,
        "after",
        lambda delay, callback, *args: (
            callback(*args)
            if len(args) == 3
            else None
        ),
    )
    gui_app.is_batch_running = True
    gui_app.batch_selected_tag = None

    gui_app._run_batch_thread()

    assert gui_app.stat_elapsed_time.cget("text") == "2.000 s"
    assert gui_app.stat_avg_iteration.cget("text") == "0.500000 s"
    assert gui_app.stat_avg_element.cget("text") == "2000.000 μs"


@pytest.mark.smoke
def test_gui_model_settings_grid_uses_declared_metadata(gui_app):
    """Create editable model rows from the selected model declarations."""
    max_population_row = gui_app.model_setting_rows["max_population"]

    assert str(gui_app.settings_canvas.cget("yscrollcommand"))
    assert "min_iterations" in gui_app.core_setting_widgets
    assert set(gui_app.model_setting_rows) == set(gui_app.model_metadata["settings"])
    assert max_population_row["supported"] is True
    assert max_population_row["bounds"].get() == "100 <= x <= 100000000"
    assert max_population_row["description"].get() == "Population carrying capacity"


@pytest.mark.smoke
def test_gui_batch_graph_callback_does_not_reopen_progress(gui_app, monkeypatch):
    """Queue batch graphs without raising a Progress window hidden by the user."""
    progress_shows = []

    def fake_run_batch(*args, **kwargs):
        """Report one graph without executing a simulation."""
        kwargs["graph_callback"]("result/row", 7)

    monkeypatch.setattr(gui_module, "run_batch", fake_run_batch)
    monkeypatch.setattr(gui_app, "_show_progress_window", lambda: progress_shows.append(True))
    monkeypatch.setattr(gui_app.root, "after", lambda *args: None)
    gui_app.is_batch_running = True
    gui_app.batch_selected_tag = None

    gui_app._run_batch_thread()

    assert progress_shows == []
    assert gui_app._pending_graph_update == ("result/row", 7)


@pytest.mark.smoke
def test_gui_tooltips_cover_buttons_settings_and_indicators(gui_app):
    """Resolve and render concise hints for the requested GUI control groups."""
    manager = gui_app.tooltips
    core_widgets = gui_app.core_setting_widgets["max_iterations"]
    model_widgets = gui_app.model_setting_rows["max_population"]["widgets"]
    mutation_x_widgets = gui_app.model_setting_rows["mutation_x"]["widgets"]
    mutation_s_widgets = gui_app.model_setting_rows["mutation_s"]["widgets"]

    assert manager.get_text(gui_app.start_btn) == (
        "Start one simulation with the current saved configuration."
    )
    assert manager.get_text(gui_app.config_dirty_label) == (
        "Shows whether the configuration has unsaved changes."
    )
    assert all(
        manager.get_text(widget) == "Maximum simulation years"
        for widget in core_widgets
    )
    assert all(
        manager.get_text(widget) == "Population carrying capacity"
        for widget in model_widgets
    )
    assert all(
        manager.get_text(widget)
        == "Mutation half-width X. The beta shift is sampled from [-X + S*X, X + S*X]."
        for widget in mutation_x_widgets
    )
    assert all(
        manager.get_text(widget)
        == "Mutation asymmetry S. It moves the interval center by S*X: [-X + S*X, X + S*X]."
        for widget in mutation_s_widgets
    )

    manager._widget = gui_app.start_btn
    manager._show(gui_app.start_btn, manager.get_text(gui_app.start_btn))
    assert manager._tooltip is not None
    manager._hide()


@pytest.mark.smoke
def test_gui_graph_viewer_creates_legacy_and_dynamic_tabs(gui_app):
    """Create one legacy viewer tab and one tab for each declared model graph."""
    tab_titles = [gui_app.graph_notebook.tab(tab_id, "text") for tab_id in gui_app.graph_notebook.tabs()]

    assert tab_titles == ["Legacy", "Age Distribution", "Beta Distribution", "Beta Evolution"]
    assert set(gui_app.dynamic_graph_canvases) == {
        "age_distribution", "beta_distribution", "beta_evolution",
    }


@pytest.mark.smoke
def test_gui_dynamic_graph_tabs_load_declared_annual_frame(gui_app, tmp_path):
    """Load a declared seven-digit annual graph frame into its matching tab."""
    frame_path = tmp_path / "age_distribution_0000000.png"
    Image.new("RGB", (10, 10), "white").save(frame_path)

    gui_app._display_dynamic_graphs(tmp_path, 0)

    assert gui_app.dynamic_graph_images["age_distribution"].size == (10, 10)


@pytest.mark.smoke
def test_gui_detects_only_years_with_generated_graphs(gui_app, tmp_path):
    """Queue graph refreshes only for years that generated graph files."""
    assert gui_app._has_year_graphs(tmp_path, 5) is False

    Image.new("RGB", (10, 10), "white").save(tmp_path / "distribution5.png")

    assert gui_app._has_year_graphs(tmp_path, 5) is True


@pytest.mark.smoke
def test_gui_batch_tab_loads_and_saves_csv_only_after_action(gui_app, tmp_path):
    """Keep editable batch data in memory until the user saves it."""
    batch_path = tmp_path / "multi.csv"
    batch_path.write_text("tag,mutation_x\nfirst,0.5\n")

    gui_app._load_batch_csv(batch_path)
    gui_app.batch_rows[0]["mutation_x"] = "0.75"
    gui_app.is_batch_dirty = True

    assert batch_path.read_text() == "tag,mutation_x\nfirst,0.5\n"

    gui_app._save_batch_csv(batch_path)

    assert batch_path.read_text() == "tag,mutation_x\nfirst,0.75\n"
    assert gui_app.is_batch_dirty is False


@pytest.mark.smoke
def test_gui_duplicate_batch_save_can_be_cancelled(gui_app, monkeypatch, tmp_path):
    """List duplicate-row tags and leave the saved CSV unchanged on Cancel."""
    prompts = []
    batch_path = tmp_path / "duplicates.csv"
    gui_app.batch_columns = ["tag", "mutation_x"]
    gui_app.batch_rows = [
        {"tag": "first", "mutation_x": "0.5"},
        {"tag": "second", "mutation_x": "0.5"},
    ]
    gui_app._set_batch_dirty(True)
    monkeypatch.setattr(
        gui_module.messagebox,
        "askokcancel",
        lambda title, text: prompts.append((title, text)) or False,
    )

    assert gui_app._save_batch_csv(batch_path) is False
    assert not batch_path.exists()
    assert "first, second" in prompts[0][1]
    assert gui_app.is_batch_dirty is True


@pytest.mark.smoke
def test_gui_batch_start_confirms_duplicate_rows_once(gui_app, monkeypatch):
    """Confirm duplicate rows once before saving dirty inputs and launching."""
    confirmations = []
    launches = []
    gui_app.batch_columns = ["tag", "mutation_x"]
    gui_app.batch_rows = [
        {"tag": "first", "mutation_x": "0.5"},
        {"tag": "second", "mutation_x": "0.5"},
    ]
    gui_app._set_batch_dirty(True)
    gui_app.is_config_dirty = False
    monkeypatch.setattr(gui_app, "_update_config_from_ui", lambda: True)
    monkeypatch.setattr(
        gui_module,
        "inspect_batch_configuration_changes",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(gui_module.messagebox, "askyesno", lambda *args: True)
    monkeypatch.setattr(
        gui_module.messagebox,
        "askokcancel",
        lambda *args: confirmations.append(args) or True,
    )
    monkeypatch.setattr(gui_app, "_launch_batch", lambda: launches.append(True))

    gui_app._on_start_batch()

    assert len(confirmations) == 1
    assert launches == [True]


@pytest.mark.smoke
def test_gui_changed_batch_configuration_can_cancel_launch(gui_app, monkeypatch):
    """List changed resolved values and leave the batch stopped on Cancel."""
    observed = {}
    launches = []
    gui_app.is_config_dirty = False
    gui_app.is_batch_dirty = False
    monkeypatch.setattr(gui_app, "_update_config_from_ui", lambda: True)
    monkeypatch.setattr(gui_app, "_confirm_duplicate_batch_rows", lambda: True)
    monkeypatch.setattr(gui_app, "_confirm_saved_batch_inputs", lambda: True)
    monkeypatch.setattr(
        gui_module,
        "inspect_batch_configuration_changes",
        lambda *args, **kwargs: [{
            "tag": "set1",
            "changed_values": {"mutation_x": (0.1, 0.2)},
        }],
    )
    monkeypatch.setattr(gui_app, "_launch_batch", lambda: launches.append(True))

    def decline_dialog(dialog):
        """Read the scrollable warning text and click Cancel."""
        widgets = _dialog_descendants(dialog)
        text_widget = next(widget for widget in widgets if widget.winfo_class() == "Text")
        cancel_button = next(
            widget
            for widget in widgets
            if widget.winfo_class() == "TButton" and widget.cget("text") == "Cancel"
        )
        observed["text"] = text_widget.get("1.0", "end-1c")
        cancel_button.invoke()

    _stub_modal_dialog(gui_app, monkeypatch, decline_dialog)
    gui_app._on_start_batch()

    assert launches == []
    assert "set1" in observed["text"]
    assert "mutation_x: 0.1 -> 0.2" in observed["text"]


@pytest.mark.smoke
def test_gui_changed_batch_configuration_can_continue_launch(gui_app, monkeypatch):
    """Launch once after choosing to delete old results, without a resume prompt."""
    launches = []
    generic_prompts = []
    gui_app.is_config_dirty = False
    gui_app.is_batch_dirty = False
    monkeypatch.setattr(gui_app, "_update_config_from_ui", lambda: True)
    monkeypatch.setattr(gui_app, "_confirm_duplicate_batch_rows", lambda: True)
    monkeypatch.setattr(gui_app, "_confirm_saved_batch_inputs", lambda: True)
    monkeypatch.setattr(
        gui_module,
        "inspect_batch_configuration_changes",
        lambda *args, **kwargs: [{
            "tag": "set1",
            "changed_values": {"mutation_x": (0.1, 0.2)},
        }],
    )
    monkeypatch.setattr(
        gui_module.messagebox,
        "askyesno",
        lambda *args: generic_prompts.append(args) or True,
    )
    monkeypatch.setattr(gui_app, "_launch_batch", lambda: launches.append(True))

    def delete_dialog(dialog):
        """Click Delete old results on the scrollable changed-configuration dialog."""
        delete_button = next(
            widget
            for widget in _dialog_descendants(dialog)
            if widget.winfo_class() == "TButton" and widget.cget("text") == "Delete old results"
        )
        delete_button.invoke()

    _stub_modal_dialog(gui_app, monkeypatch, delete_dialog)
    gui_app._on_start_batch()

    assert launches == [True]
    assert generic_prompts == []
    assert gui_app.batch_keep_changed_tags == set()


@pytest.mark.smoke
def test_gui_changed_batch_configuration_can_keep_old_results(gui_app, monkeypatch):
    """Launch while keeping old results for tags where the user chose to keep them."""
    launches = []
    gui_app.is_config_dirty = False
    gui_app.is_batch_dirty = False
    monkeypatch.setattr(gui_app, "_update_config_from_ui", lambda: True)
    monkeypatch.setattr(gui_app, "_confirm_duplicate_batch_rows", lambda: True)
    monkeypatch.setattr(gui_app, "_confirm_saved_batch_inputs", lambda: True)
    monkeypatch.setattr(gui_module.messagebox, "askyesno", lambda *args: True)
    monkeypatch.setattr(
        gui_module,
        "inspect_batch_configuration_changes",
        lambda *args, **kwargs: [{
            "tag": "set1",
            "changed_values": {"mutation_x": (0.1, 0.2)},
        }],
    )
    monkeypatch.setattr(gui_app, "_launch_batch", lambda: launches.append(True))

    def keep_dialog(dialog):
        """Click Keep old results on the scrollable changed-configuration dialog."""
        keep_button = next(
            widget
            for widget in _dialog_descendants(dialog)
            if widget.winfo_class() == "TButton" and widget.cget("text") == "Keep old results"
        )
        keep_button.invoke()

    _stub_modal_dialog(gui_app, monkeypatch, keep_dialog)
    gui_app._on_start_batch()

    assert launches == [True]
    assert gui_app.batch_keep_changed_tags == {"set1"}


@pytest.mark.smoke
def test_gui_changed_batch_configuration_dialog_caps_height_and_scrolls(gui_app, monkeypatch):
    """Cap a long changed-configuration list at 75% of the screen height with a scrollbar."""
    gui_app.root.deiconify()
    gui_app.root.update_idletasks()
    warning_text = "\n".join(f"tag_{index}: mutation_x: 0.1 -> 0.{index}" for index in range(200))
    observed = {}

    def inspect_dialog(dialog):
        """Capture the capped dialog geometry and scroll wiring, then close it."""
        widgets = _dialog_descendants(dialog)
        text_widget = next(widget for widget in widgets if widget.winfo_class() == "Text")
        cancel_button = next(
            widget
            for widget in widgets
            if widget.winfo_class() == "TButton" and widget.cget("text") == "Cancel"
        )
        observed["dialog_height"] = dialog.winfo_height()
        observed["max_height"] = int(dialog.winfo_screenheight() * 0.75)
        observed["scrollcommand"] = str(text_widget.cget("yscrollcommand"))
        observed["text"] = text_widget.get("1.0", "end-1c")
        cancel_button.invoke()

    _stub_modal_dialog(gui_app, monkeypatch, inspect_dialog)

    decision = gui_app._show_configuration_change_dialog(warning_text)

    assert decision == "cancel"
    assert observed["text"] == warning_text
    assert observed["scrollcommand"]
    assert observed["dialog_height"] <= observed["max_height"]


@pytest.mark.smoke
def test_gui_load_model_batch_defaults_updates_only_editor_memory(gui_app, monkeypatch):
    """Load active model batch defaults without writing multi.csv."""
    notices = []
    original_contents = gui_app.batch_path.read_text() if gui_app.batch_path.exists() else None
    monkeypatch.setattr(gui_module.messagebox, "showinfo", lambda *args: notices.append(args))

    gui_app._load_current_model_batch_defaults()

    assert gui_app.batch_columns == ["tag", "mutation_x"]
    assert gui_app.batch_rows[0] == {"tag": "x_0.05", "mutation_x": "0.05"}
    assert gui_app.is_batch_dirty is True
    if original_contents is None:
        assert not gui_app.batch_path.exists()
    else:
        assert gui_app.batch_path.read_text() == original_contents


@pytest.mark.smoke
def test_gui_accepts_model_batch_replacement_without_saving(gui_app, monkeypatch):
    """Replace batch editor rows only after accepting a model default prompt."""
    gui_app.batch_rows = [{"tag": "custom", "mutation_x": "9.0"}]
    gui_app.is_batch_dirty = False
    monkeypatch.setattr(gui_module.messagebox, "askyesno", lambda *args: True)

    gui_app._offer_model_batch_defaults("model_base")

    assert gui_app.batch_rows[0]["tag"] == "x_0.05"
    assert gui_app.is_batch_dirty is True


@pytest.mark.smoke
def test_gui_batch_start_and_stop_manage_worker_state(gui_app, monkeypatch):
    """Launch the in-GUI batch worker and request cooperative cancellation."""
    launches = []
    gui_app.is_config_dirty = False
    gui_app.is_batch_dirty = False
    monkeypatch.setattr(gui_app, "_confirm_duplicate_batch_rows", lambda: True)
    monkeypatch.setattr(gui_app, "_confirm_batch_configuration_changes", lambda selected_tag=None: ([], set()))
    monkeypatch.setattr(gui_module.messagebox, "askyesno", lambda *args: True)
    monkeypatch.setattr(
        gui_app,
        "_launch_batch",
        lambda: (launches.append(True), setattr(gui_app, "is_batch_running", True)),
    )

    gui_app._on_start_batch()
    gui_app._stop_batch()

    assert launches == [True]
    assert gui_app.batch_cancel_requested is True


@pytest.mark.smoke
def test_gui_progress_stop_cancels_active_batch(gui_app, monkeypatch):
    """Enable Progress Stop for batch work and route it to cooperative cancellation."""
    thread_starts = []

    class FakeThread:
        """Record batch worker startup without executing it."""

        def __init__(self, target, daemon):
            """Store worker properties."""
            self.target = target
            self.daemon = daemon

        def start(self):
            """Record the requested worker start."""
            thread_starts.append(True)

    monkeypatch.setattr(gui_module.threading, "Thread", FakeThread)

    gui_app._launch_batch("selected")

    assert gui_app.is_batch_running is True
    assert str(gui_app.progress_stop_btn.cget("state")) == "normal"
    assert thread_starts == [True]

    gui_app.progress_stop_btn.invoke()

    assert gui_app.batch_cancel_requested is True
    assert gui_app.batch_status_var.get() == "Cancellation requested"
    assert str(gui_app.progress_stop_btn.cget("state")) == "disabled"


@pytest.mark.smoke
def test_gui_selected_batch_row_keeps_existing_results_when_cancelled(gui_app, monkeypatch, tmp_path):
    """Require the tag replacement confirmation before launching one selected row."""
    launches = []
    selected_tag = "selected"
    gui_app.batch_columns = ["tag"]
    gui_app.batch_rows = [{"tag": selected_tag}]
    gui_app._render_batch_grid()
    result_file = tmp_path / "result" / selected_tag / "final.csv"
    result_file.parent.mkdir(parents=True)
    result_file.write_text("old results", encoding="utf-8")
    gui_app.batch_tree.selection_set("0")
    monkeypatch.setattr(gui_app, "_update_config_from_ui", lambda: True)
    monkeypatch.setattr(gui_app, "_confirm_duplicate_batch_rows", lambda: True)
    monkeypatch.setattr(gui_app, "_confirm_saved_batch_inputs", lambda: True)
    monkeypatch.setattr(gui_app, "_confirm_batch_configuration_changes", lambda selected_tag=None: ([], set()))
    monkeypatch.setattr(gui_app, "_ask_tag_result_replacement", lambda tag: False)
    monkeypatch.setattr(gui_app, "_launch_batch", lambda tag: launches.append(tag))

    gui_app._on_run_selected_row()

    assert result_file.read_text(encoding="utf-8") == "old results"
    assert launches == []


@pytest.mark.smoke
def test_gui_refresh_reschedules_after_display_error(gui_app, monkeypatch):
    """Keep the periodic GUI refresh alive when one display update fails."""
    schedules = []
    monkeypatch.setattr(gui_app, "_apply_gui_refresh", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(gui_app, "_schedule_gui_refresh", lambda: schedules.append(True))

    gui_app._refresh_gui()

    assert schedules == [True]


@pytest.mark.smoke
def test_gui_counts_and_archives_aggregate_batch_results(gui_app, monkeypatch, tmp_path):
    """Expose aggregate completion count and archive results after confirmation."""
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    (result_dir / "result.csv").write_text("model,tag\nmodel_base,first\nmodel_base,second\n")
    monkeypatch.setattr(gui_module.messagebox, "askyesno", lambda *args: True)

    assert gui_app._aggregate_result_count() == 2

    gui_app._clear_result_directory()

    assert not result_dir.exists()
    assert len(list(tmp_path.glob("result_*.bak"))) == 1


@pytest.mark.smoke
def test_gui_dirty_start_saves_before_launching(gui_app, monkeypatch):
    """Save canonical config before starting an unsaved GUI configuration."""
    saves = []
    starts = []
    gui_app.is_config_dirty = True
    monkeypatch.setattr(gui_module.messagebox, "askyesnocancel", lambda *args: True)
    monkeypatch.setattr(gui_app, "_save_config", lambda: saves.append(True))
    monkeypatch.setattr(gui_app, "_launch_simulation", lambda: starts.append(True))

    gui_app._start_simulation()

    assert saves == [True]
    assert starts == [True]


@pytest.mark.smoke
def test_gui_dirty_start_discards_changes_and_launches_saved_config(gui_app, monkeypatch):
    """Restore the canonical configuration before launching after choosing No."""
    starts = []
    gui_app.is_config_dirty = True
    monkeypatch.setattr(gui_module.messagebox, "askyesnocancel", lambda *args: False)
    monkeypatch.setattr(gui_app, "_restore_canonical_config", lambda: True)
    monkeypatch.setattr(gui_app, "_launch_simulation", lambda: starts.append(True))

    gui_app._start_simulation()

    assert starts == [True]


@pytest.mark.smoke
def test_gui_dirty_start_cancels_without_launching(gui_app, monkeypatch):
    """Leave unsaved configuration intact when the start dialog is cancelled."""
    starts = []
    gui_app.is_config_dirty = True
    monkeypatch.setattr(gui_module.messagebox, "askyesnocancel", lambda *args: None)
    monkeypatch.setattr(gui_app, "_launch_simulation", lambda: starts.append(True))

    gui_app._start_simulation()

    assert starts == []


@pytest.mark.smoke
def test_gui_single_start_archives_existing_tag_results(gui_app, monkeypatch, tmp_path):
    """Archive only the active tag directory before starting after confirmation."""
    starts = []
    result_dir = tmp_path / "result" / gui_app.config["tag"]
    result_dir.mkdir(parents=True)
    (result_dir / "result.csv").write_text("old results")
    monkeypatch.setattr(gui_app, "_ask_tag_result_replacement", lambda tag: True)
    monkeypatch.setattr(gui_app, "_launch_simulation", lambda: starts.append(True))

    gui_app._start_simulation()

    assert not result_dir.exists()
    assert len(list((tmp_path / "result").glob(f"{gui_app.config['tag']}_*.bak"))) == 1
    assert starts == [True]


@pytest.mark.smoke
def test_gui_single_start_keeps_existing_tag_results_when_cancelled(gui_app, monkeypatch, tmp_path):
    """Keep existing tag results and cancel the start when replacement is declined."""
    starts = []
    result_dir = tmp_path / "result" / gui_app.config["tag"]
    result_dir.mkdir(parents=True)
    result_file = result_dir / "result.csv"
    result_file.write_text("old results")
    monkeypatch.setattr(gui_app, "_ask_tag_result_replacement", lambda tag: False)
    monkeypatch.setattr(gui_app, "_launch_simulation", lambda: starts.append(True))

    gui_app._start_simulation()

    assert result_file.read_text() == "old results"
    assert starts == []


@pytest.mark.smoke
def test_gui_save_handler_writes_config_only_after_action(gui_app, monkeypatch):
    """Write config through the save handler without displaying a dialog."""
    monkeypatch.setattr(gui_module.messagebox, "showinfo", lambda *args, **kwargs: None)

    gui_app._on_save_config()

    assert gui_app.config_file == "config.json"
    assert gui_module.Path("config.json").is_file()


@pytest.mark.smoke
def test_gui_load_handler_keeps_canonical_path_and_marks_unsaved(gui_app, monkeypatch, tmp_path):
    """Load selected JSON into controls without replacing canonical config.json."""
    config_path = tmp_path / "loaded.json"
    config_path.write_text(json.dumps({"max_population": 500, "tag": "loaded"}))
    monkeypatch.setattr(gui_module.filedialog, "askopenfilename", lambda **kwargs: str(config_path))
    monkeypatch.setattr(gui_module.messagebox, "showinfo", lambda *args, **kwargs: None)

    gui_app._on_load_config()

    assert gui_app.config_file == "config.json"
    assert gui_app.config["max_population"] == 500
    assert gui_app.config["model"] == "model_base"
    assert gui_app.tag_var.get() == "loaded"
    assert gui_app.is_config_dirty is True


@pytest.mark.smoke
def test_gui_load_handler_prepares_and_renders_selected_model(gui_app, monkeypatch, tmp_path):
    """Load a different model configuration into matching in-memory controls."""
    config_path = tmp_path / "age_only.json"
    config_path.write_text(json.dumps({"model": "model_age_only", "seed": 7}))
    prepared_metadata = validate_model_metadata(Model)
    preparations = []
    monkeypatch.setattr(gui_module.filedialog, "askopenfilename", lambda **kwargs: str(config_path))
    monkeypatch.setattr(gui_module.messagebox, "showinfo", lambda *args, **kwargs: None)
    monkeypatch.setattr(gui_module.messagebox, "showwarning", lambda *args, **kwargs: None)

    def prepare(config, model_name):
        """Return the metadata for the selected age-only model."""
        preparations.append(model_name)
        return {**config, "model": model_name}, prepared_metadata, []

    monkeypatch.setattr(gui_module, "_prepare_model_configuration", prepare)

    gui_app._on_load_config()

    assert preparations == ["model_age_only"]
    assert gui_app.config["model"] == "model_age_only"
    assert gui_app.model_var.get() == "model_age_only"
    assert gui_app.model_setting_rows["seed"]["supported"] is True
    assert gui_app.model_setting_rows["max_population"]["supported"] is False
    assert [gui_app.graph_notebook.tab(tab, "text") for tab in gui_app.graph_notebook.tabs()] == [
        "Legacy", "Age Distribution",
    ]


@pytest.mark.smoke
def test_gui_loads_all_current_model_defaults_without_writing_files(gui_app, monkeypatch):
    """Reset active settings and batch editor to model defaults in memory only."""
    notices = []
    gui_app.config["max_population"] = 100
    gui_app.model_setting_rows["max_population"]["value"].set("100")
    original_batch = gui_app.batch_path.read_text() if gui_app.batch_path.exists() else None
    monkeypatch.setattr(gui_module.messagebox, "showinfo", lambda *args: notices.append(args))

    gui_app._load_current_model_defaults()

    assert gui_app.config["max_population"] == 2000
    assert gui_app.model_setting_rows["max_population"]["value"].get() == "2000"
    assert gui_app.batch_rows[0]["tag"] == "x_0.05"
    assert gui_app.is_config_dirty is True
    assert gui_app.is_batch_dirty is True
    assert notices
    if original_batch is None:
        assert not gui_app.batch_path.exists()
    else:
        assert gui_app.batch_path.read_text() == original_batch


@pytest.mark.smoke
def test_gui_save_as_exports_copy_without_changing_canonical_path(gui_app, monkeypatch, tmp_path):
    """Export a config copy while retaining config.json as the Save target."""
    export_path = tmp_path / "exported.json"
    monkeypatch.setattr(gui_module.filedialog, "asksaveasfilename", lambda **kwargs: str(export_path))
    monkeypatch.setattr(gui_module.messagebox, "showinfo", lambda *args, **kwargs: None)
    gui_app.tag_var.set("exported")

    gui_app._on_save_config_as()

    assert gui_app.config_file == "config.json"
    assert json.loads(export_path.read_text())["tag"] == "exported"


@pytest.mark.smoke
def test_gui_model_selection_applies_corrections_and_marks_config_dirty(gui_app, monkeypatch):
    """Apply selected model declarations to the unsaved GUI configuration."""
    warnings = []
    gui_app.model_setting_rows["max_population"]["value"].set("1")
    gui_app.model_var.set("model_base")
    monkeypatch.setattr(gui_module.messagebox, "showwarning", lambda *args: warnings.append(args))

    gui_app._on_model_selected()

    assert gui_app.config["model"] == "model_base"
    assert gui_app.config["max_population"] == 100
    assert gui_app.is_config_dirty is True
    assert warnings


@pytest.mark.smoke
def test_gui_load_handler_keeps_current_config_when_json_is_invalid(gui_app, monkeypatch, tmp_path):
    """Preserve active configuration when the selected JSON cannot be parsed."""
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("not json")
    errors = []
    monkeypatch.setattr(gui_module.filedialog, "askopenfilename", lambda **kwargs: str(invalid_path))
    monkeypatch.setattr(gui_module.messagebox, "showerror", lambda *args: errors.append(args))
    gui_app.config["tag"] = "current"
    gui_app._load_config_to_ui()

    gui_app._on_load_config()

    assert gui_app.config_file == "config.json"
    assert gui_app.config["tag"] == "current"
    assert gui_app.tag_var.get() == "current"
    assert errors


@pytest.mark.smoke
def test_gui_stop_handler_clears_running_state(gui_app):
    """Stop handler clears the running flag without a simulation."""
    gui_app.is_running = True

    gui_app._stop_simulation()

    assert gui_app.is_running is False


@pytest.mark.smoke
def test_gui_dirty_indicators_are_independent_and_blue_when_modified(gui_app):
    """Show separate synchronized config and batch dirty indicators in the top panel."""
    gui_app._set_config_dirty(True)

    assert gui_app.config_dirty_var.get() == "Config modified"
    assert gui_app.config_dirty_label.cget("foreground") == "#0067c0"
    assert gui_app.batch_dirty_var.get() == "Batch saved"

    gui_app._set_batch_dirty(True)
    gui_app._set_config_dirty(False)

    assert gui_app.config_dirty_var.get() == "Config saved"
    assert gui_app.batch_dirty_var.get() == "Batch modified"
    assert gui_app.batch_dirty_label.cget("foreground") == "#0067c0"


@pytest.mark.smoke
def test_gui_progress_shows_single_and_batch_calculation_sources(gui_app):
    """Display a tag plus default-config or full batch-row source in Progress."""
    gui_app._set_progress_calculation("single")

    assert gui_app.progress_tag_var.get() == "single"
    assert gui_app.progress_source_var.get() == "Default config"

    gui_app.batch_rows = [{"tag": "batch_a", "model": "model_base", "mutation_x": "0.2"}]
    gui_app._update_batch_status(0, 1, "batch_a", "running")

    assert gui_app.progress_tag_var.get() == "batch_a"
    assert gui_app.progress_source_var.get() == "Batch row: tag=batch_a, model=model_base, mutation_x=0.2"


@pytest.mark.smoke
def test_gui_log_autoscroll_can_be_disabled(gui_app, monkeypatch):
    """Scroll new log messages only while the Progress checkbox is enabled."""
    scroll_calls = []
    monkeypatch.setattr(gui_app.log_text, "see", lambda position: scroll_calls.append(position))
    gui_app._gui_log_messages.put("first")

    gui_app._apply_gui_refresh()

    assert gui_app.log_autoscroll_var.get() is True
    assert scroll_calls == [tk.END]

    gui_app.log_autoscroll_var.set(False)
    gui_app._gui_log_messages.put("second")
    gui_app._apply_gui_refresh()

    assert scroll_calls == [tk.END]
    assert "second" in gui_app.log_text.get("1.0", tk.END)


@pytest.mark.smoke
def test_gui_batch_model_option_and_delete_column(gui_app):
    """Offer model as a batch column and delete optional columns from every row."""
    assert "model" in tuple(gui_app.batch_column_selector.cget("values"))
    gui_app.batch_columns = ["tag", "model"]
    gui_app.batch_rows = [{"tag": "run", "model": "model_base"}]
    gui_app.batch_column_var.set("model")

    gui_app._delete_batch_column()

    assert gui_app.batch_columns == ["tag"]
    assert gui_app.batch_rows == [{"tag": "run"}]
    assert gui_app.is_batch_dirty is True


@pytest.mark.smoke
def test_gui_dirty_transition_names_current_experiment(gui_app, monkeypatch):
    """Identify the experiment whose unsaved changes are being resolved."""
    prompts = []
    gui_app.experiment_manager.set_active_experiment("exp_alpha")
    gui_app._set_config_dirty(True)
    monkeypatch.setattr(
        gui_module.messagebox,
        "askyesnocancel",
        lambda title, message: prompts.append((title, message)),
    )

    assert gui_app._confirm_experiment_transition() is False
    assert "exp_alpha" in prompts[0][1]


@pytest.mark.smoke
def test_gui_deletes_active_experiment_and_selects_next(gui_app, tmp_path, monkeypatch):
    """Delete the confirmed active experiment and activate another available one."""
    manager = gui_app.experiment_manager
    alpha_dir = manager.create_experiment("exp_alpha", {"model": "model_base"})
    manager.create_experiment("exp_beta", {"model": "model_base"}, activate=False)
    gui_app.experiment_dir = alpha_dir
    gui_app.config_file = str(alpha_dir / "config.json")
    gui_app._set_config_dirty(False)
    gui_app._set_batch_dirty(False)
    monkeypatch.setattr(gui_module.messagebox, "askyesno", lambda *args: True)

    gui_app._on_delete_experiment()

    assert not alpha_dir.exists()
    assert manager.get_active_experiment_name() == "exp_beta"
    assert gui_app.experiment_dir == tmp_path / "data" / "exp_beta"


@pytest.mark.smoke
def test_gui_opens_selected_experiment_in_platform_file_manager(gui_app, monkeypatch):
    """Open the selected experiment directory with each supported platform command."""
    selected_dir = gui_app.experiment_manager.create_experiment(
        "exp_beta",
        {"model": "model_base"},
        activate=False,
    )
    gui_app.experiment_var.set("exp_beta")
    startfile_calls = []
    process_calls = []
    monkeypatch.setattr(gui_module.os, "startfile", startfile_calls.append)
    monkeypatch.setattr(gui_module.subprocess, "Popen", process_calls.append)

    monkeypatch.setattr(gui_module.sys, "platform", "win32")
    gui_app._on_open_experiment_dir()
    monkeypatch.setattr(gui_module.sys, "platform", "darwin")
    gui_app._on_open_experiment_dir()
    monkeypatch.setattr(gui_module.sys, "platform", "linux")
    gui_app._on_open_experiment_dir()

    assert startfile_calls == [selected_dir]
    assert process_calls == [
        ["open", str(selected_dir)],
        ["xdg-open", str(selected_dir)],
    ]
    siblings = gui_app.open_experiment_dir_button.master.pack_slaves()
    button_index = siblings.index(gui_app.open_experiment_dir_button)
    assert siblings[button_index - 1].cget("text") == "Delete Experiment"
    assert gui_app.tooltips.get_text(gui_app.open_experiment_dir_button) == (
        "Open the selected experiment directory in the system file manager."
    )


@pytest.mark.smoke
def test_gui_clone_dialog_copies_results_and_activates_clone(gui_app, monkeypatch):
    """Clone exact saved files and selected results, then activate the completed copy."""
    gui_app._create_experiment("source", "model_base")
    gui_app._activate_experiment("source")
    gui_app.root.deiconify()
    gui_app.root.update_idletasks()
    source_name = gui_app.experiment_manager.get_active_experiment_name()
    source_dir = gui_app.experiment_dir
    config_bytes = Path(gui_app.config_file).read_bytes()
    batch_bytes = gui_app.batch_path.read_bytes() if gui_app.batch_path.exists() else None
    result_file = source_dir / "result" / "run_a" / "result.csv"
    result_file.parent.mkdir(parents=True)
    result_file.write_bytes(b"year,count\r\n1,100\r\n")
    archive_dir = source_dir / "result_20260823_120000_000000.bak"
    archive_dir.mkdir()
    (archive_dir / "old.csv").write_bytes(b"old")
    observed = {}

    def descendants(widget):
        """Return every nested Tk child below one widget."""
        children = list(widget.winfo_children())
        return children + [nested for child in children for nested in descendants(child)]

    def complete_dialog(dialog):
        """Select result copying and submit the live clone form."""
        widgets = descendants(dialog)
        name_entry = next(widget for widget in widgets if widget.winfo_class() == "TEntry")
        copy_check = next(
            widget
            for widget in widgets
            if widget.winfo_class() == "TCheckbutton" and widget.cget("text") == "Copy results"
        )
        ok_button = next(
            widget
            for widget in widgets
            if widget.winfo_class() == "TButton" and widget.cget("text") == "OK"
        )
        observed["state"] = str(copy_check.cget("state"))
        observed["selected_before"] = copy_check.instate(["selected"])
        observed["grab"] = dialog.grab_current()
        name_entry.insert(0, "cloned_experiment")
        copy_check.invoke()
        ok_button.invoke()

    _stub_modal_dialog(gui_app, monkeypatch, complete_dialog)
    gui_app._on_clone_experiment()

    clone_dir = source_dir.parent / "cloned_experiment"
    assert observed["state"] == "normal"
    assert observed["selected_before"] is False
    assert observed["grab"] is not None
    assert (clone_dir / "config.json").read_bytes() == config_bytes
    if batch_bytes is not None:
        assert (clone_dir / "multi.csv").read_bytes() == batch_bytes
    assert (clone_dir / "result" / "run_a" / "result.csv").read_bytes() == result_file.read_bytes()
    assert not (clone_dir / archive_dir.name).exists()
    assert gui_app.experiment_manager.get_active_experiment_name() == "cloned_experiment"
    assert gui_app.experiment_var.get() == "cloned_experiment"
    assert gui_app.experiment_dir == clone_dir
    assert Path(gui_app.config_file) == clone_dir / "config.json"
    assert source_name != "cloned_experiment"

    siblings = gui_app.clone_experiment_button.master.pack_slaves()
    clone_index = siblings.index(gui_app.clone_experiment_button)
    assert siblings[clone_index - 1].cget("text") == "New Experiment"
    assert siblings[clone_index + 1].cget("text") == "Delete Experiment"
    assert gui_app.tooltips.get_text(gui_app.clone_experiment_button) == (
        "Clone the active experiment's saved configuration and optional results."
    )


@pytest.mark.smoke
def test_gui_clone_dialog_disables_result_copy_without_results(gui_app, monkeypatch):
    """Keep Copy results unchecked and disabled for an empty result directory."""
    gui_app._create_experiment("source", "model_base")
    gui_app._activate_experiment("source")
    gui_app.root.deiconify()
    result_dir = gui_app.experiment_dir / "result"
    result_dir.mkdir(exist_ok=True)
    observed = {}

    def descendants(widget):
        """Return every nested Tk child below one widget."""
        children = list(widget.winfo_children())
        return children + [nested for child in children for nested in descendants(child)]

    def cancel_dialog(dialog):
        """Inspect and cancel the live clone form."""
        widgets = descendants(dialog)
        copy_check = next(
            widget
            for widget in widgets
            if widget.winfo_class() == "TCheckbutton" and widget.cget("text") == "Copy results"
        )
        cancel_button = next(
            widget
            for widget in widgets
            if widget.winfo_class() == "TButton" and widget.cget("text") == "Cancel"
        )
        observed["state"] = str(copy_check.cget("state"))
        observed["selected"] = copy_check.instate(["selected"])
        cancel_button.invoke()

    _stub_modal_dialog(gui_app, monkeypatch, cancel_dialog)
    clone_dir = gui_app._prompt_clone_experiment(
        gui_app.experiment_manager.get_active_experiment_name(),
    )

    assert clone_dir is None
    assert observed == {"state": "disabled", "selected": False}


@pytest.mark.smoke
def test_gui_new_experiment_dialog_selects_model_and_creates_its_defaults(gui_app, monkeypatch):
    """Offer discovered models in a usable dialog and create files from the selection."""
    gui_app.root.deiconify()
    gui_app.root.update_idletasks()
    selected_model = next(model for model in gui_app.available_models if model != "model_base")
    observed = {}

    def descendants(widget):
        """Return every nested Tk child below one widget."""
        children = list(widget.winfo_children())
        return children + [nested for child in children for nested in descendants(child)]

    def complete_dialog(dialog):
        """Inspect and submit the live modal new-experiment form."""
        gui_app.root.update_idletasks()
        widgets = descendants(dialog)
        name_entry = next(widget for widget in widgets if widget.winfo_class() == "TEntry")
        model_selector = next(widget for widget in widgets if widget.winfo_class() == "TCombobox")
        description_text = next(widget for widget in widgets if widget.winfo_class() == "Text")
        create_button = next(
            widget
            for widget in widgets
            if widget.winfo_class() == "TButton" and widget.cget("text") == "Create"
        )
        observed["minsize"] = tuple(dialog.minsize())
        observed["state"] = str(model_selector.cget("state"))
        observed["models"] = tuple(model_selector.cget("values"))
        observed["explanation"] = next(
            widget.cget("text")
            for widget in widgets
            if widget.winfo_class() == "Message"
        )
        observed["position"] = (dialog.winfo_x(), dialog.winfo_y())
        observed["expected_position"] = (
            max(0, gui_app.root.winfo_rootx() + (gui_app.root.winfo_width() - dialog.winfo_width()) // 2),
            max(0, gui_app.root.winfo_rooty() + (gui_app.root.winfo_height() - dialog.winfo_height()) // 2),
        )
        name_entry.insert(0, "selected_model")
        model_selector.set(selected_model)
        model_selector.event_generate("<<ComboboxSelected>>")
        gui_app.root.update_idletasks()
        observed["description"] = description_text.get("1.0", "end-1c")
        create_button.invoke()

    _stub_modal_dialog(gui_app, monkeypatch, complete_dialog)
    experiment_dir = gui_app._prompt_new_experiment()

    config = json.loads((experiment_dir / "config.json").read_text(encoding="utf-8"))
    selected_metadata = validate_model_metadata(gui_module.load_model_class(selected_model))
    assert observed["minsize"] == (620, 480)
    assert observed["state"] == "readonly"
    assert observed["models"] == tuple(gui_app.available_models)
    assert "can be switched at any time" in observed["explanation"]
    assert "data/<name>/" in observed["explanation"]
    assert "default.conf" in observed["explanation"]
    assert "active experiment" in observed["explanation"]
    expected_heading = gui_module._load_model_description(selected_model).splitlines()[0][2:]
    assert expected_heading in observed["description"]
    assert "Purpose" in observed["description"]
    assert "Inheritance" in observed["description"]
    assert observed["position"] == observed["expected_position"]
    assert config["model"] == selected_model
    assert all(
        config[name] == details["default"]
        for name, details in selected_metadata["settings"].items()
    )
    assert (experiment_dir / "multi.csv").is_file()


@pytest.mark.smoke
def test_gui_about_model_button_opens_modal_selected_description(gui_app, monkeypatch):
    """Show the current selector model in a modal read-only Markdown window."""
    gui_app.root.deiconify()
    selected_model = "model_base_fast_fixed_fecundity"
    gui_app.model_var.set(selected_model)
    observed = {}

    def inspect_dialog(dialog):
        """Capture the live About dialog and close it through its button."""
        descendants = list(dialog.winfo_children())
        markdown_view = next(
            widget for widget in descendants if isinstance(widget, gui_module.LightweightMarkdownView)
        )
        close_button = next(
            widget
            for widget in descendants
            if widget.winfo_class() == "TButton" and widget.cget("text") == "Close"
        )
        observed["title"] = dialog.title()
        observed["grab"] = dialog.grab_current()
        observed["description"] = markdown_view.text.get("1.0", "end-1c")
        close_button.invoke()

    _stub_modal_dialog(gui_app, monkeypatch, inspect_dialog)
    gui_app._on_about_model()

    assert observed["title"] == f"About Model: {selected_model}"
    assert observed["grab"] is not None
    assert "Fast beta model with fixed parent fecundity" in observed["description"]
    assert gui_app.tooltips.get_text(gui_app.about_model_button) == (
        "Show the selected model's purpose, inheritance, rules, and differences."
    )