import tkinter as tk

import pytest

import gui as gui_module
from gui import SimulationGUI
from main import set_logger


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
    gui_app.setting_vars["max_population"].set("500")
    gui_app.tag_var.set("gui_smoke")

    updated = gui_app._update_config_from_ui()

    assert updated is True
    assert gui_app.config["max_population"] == 500
    assert gui_app.config["tag"] == "gui_smoke"


@pytest.mark.smoke
def test_gui_save_handler_writes_config_only_after_action(gui_app, monkeypatch):
    """Write config through the save handler without displaying a dialog."""
    monkeypatch.setattr(gui_module.messagebox, "showinfo", lambda *args, **kwargs: None)

    gui_app._on_save_config()

    assert gui_app.config_file == "config.json"
    assert gui_module.Path("config.json").is_file()


@pytest.mark.smoke
def test_gui_stop_handler_clears_running_state(gui_app):
    """Stop handler clears the running flag without a simulation."""
    gui_app.is_running = True

    gui_app._stop_simulation()

    assert gui_app.is_running is False