import tkinter as tk
import json
from pathlib import Path

import pytest
from PIL import Image

import gui as gui_module
from gui import SimulationGUI
from main import set_logger
from model import Model
from model_metadata import validate_model_metadata


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
def test_gui_constructs_and_updates_valid_settings(gui_app):
    """Construct the GUI and transfer valid controls into memory."""
    gui_app.model_setting_rows["max_population"]["value"].set("500")
    gui_app.tag_var.set("gui_smoke")

    updated = gui_app._update_config_from_ui()

    assert updated is True
    assert gui_app.config["max_population"] == 500
    assert gui_app.config["tag"] == "gui_smoke"


@pytest.mark.smoke
def test_gui_model_settings_grid_uses_declared_metadata(gui_app):
    """Create editable model rows from the selected model declarations."""
    max_population_row = gui_app.model_setting_rows["max_population"]

    assert set(gui_app.model_setting_rows) == set(gui_app.model_metadata["settings"])
    assert max_population_row["configured"].get() == "X"
    assert max_population_row["bounds"].get() == "100 <= x <= 100000000"
    assert max_population_row["description"].get() == "Population carrying capacity"


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
def test_gui_counts_and_clears_aggregate_batch_results(gui_app, monkeypatch, tmp_path):
    """Expose aggregate completion count and remove results after confirmation."""
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    (result_dir / "result.csv").write_text("model,tag\nmodel_base,first\nmodel_base,second\n")
    monkeypatch.setattr(gui_module.messagebox, "askyesno", lambda *args: True)

    assert gui_app._aggregate_result_count() == 2

    gui_app._clear_result_directory()

    assert not result_dir.exists()


@pytest.mark.smoke
def test_gui_dirty_start_saves_before_launching(gui_app, monkeypatch):
    """Save canonical config before starting an unsaved GUI configuration."""
    saves = []
    starts = []
    gui_app.is_config_dirty = True
    monkeypatch.setattr(gui_module.messagebox, "askyesno", lambda *args: True)
    monkeypatch.setattr(gui_app, "_save_config", lambda: saves.append(True))
    monkeypatch.setattr(gui_app, "_launch_simulation", lambda: starts.append(True))

    gui_app._start_simulation()

    assert saves == [True]
    assert starts == [True]


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
    assert set(gui_app.model_setting_rows) == {"seed"}
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