"""
Tkinter GUI for chimp evolution simulation
Allows parameter configuration, execution control, and result visualization
"""

import json
import csv
import io
import os
import re
import subprocess
import sys
import threading
import time
import queue
from pathlib import Path
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, messagebox, filedialog
import torch
from PIL import Image, ImageTk

from main import PopulationSimulation, set_logger, log
from batch import (
    find_duplicate_variant_tags,
    format_configuration_change_warning,
    format_duplicate_variant_warning,
    inspect_batch_configuration_changes,
    run_batch,
)
from experiment_manager import ExperimentManager, archive_path, atomic_write_text
from load_model import ModelLoadError, discover_models, load_model_class
from metadata import validate_model_metadata
from settings import DEFAULT_SETTINGS, PARAMETER_DESCRIPTIONS, PARAMETER_RANGES


CORE_SETTING_NAMES = (
    "stat_generation_period",
    "graph_generation_period",
    "min_iterations",
    "max_iterations",
)

BUTTON_TOOLTIPS = {
    "New Experiment": "Create an experiment from the selected model defaults.",
    "Clone...": "Clone the active experiment's saved configuration and optional results.",
    "Delete Experiment": "Delete the active experiment and all of its stored data.",
    "Open In File Explorer": "Open the selected experiment directory in the system file manager.",
    "Load Model": "Load the selected model and its declared settings.",
    "About Model...": "Show the selected model's purpose, inheritance, rules, and differences.",
    "Load All Model Defaults": "Reset configuration and batch values to the active model defaults.",
    "Start Simulation": "Start one simulation with the current saved configuration.",
    "Stop": "Cancel the active single simulation and keep partial results.",
    "Save Config": "Save the current configuration to the active experiment.",
    "Save Config As...": "Save the current configuration to another JSON file.",
    "Load Config ...": "Load configuration values from a JSON file.",
    "Re-read Config": "Discard unsaved configuration edits and reload config.json.",
    "Show Progress Window": "Open the non-modal graphs, statistics, and log window.",
    "Create": "Create the experiment with the entered name and selected model.",
    "Cancel": "Close this dialog without applying the requested action.",
    "Stop Simulation": "Cancel the active simulation or batch and keep partial results.",
    "Finalize Simulation": "Finish the current simulation successfully at the end of this year.",
    "Add Row": "Append an editable row to the batch table.",
    "Load Model Defaults": "Replace batch rows with defaults declared by the active model.",
    "Clear Results": "Archive all current experiment results.",
    "Add Column": "Add the selected configuration field to every batch row.",
    "Delete Column": "Delete the selected optional field from every batch row.",
    "Save Batch": "Save the batch table to multi.csv.",
    "Re-read Batch": "Discard unsaved batch edits and reload multi.csv.",
    "Start Batch": "Run all incomplete rows from the saved batch table.",
    "Run Selected Row": "Run only the selected saved batch row.",
    "Stop Batch": "Cancel the active batch and keep partial current-row results.",
    "Yes": "Confirm the requested action.",
    "Close": "Close this window.",
}

INLINE_MARKDOWN_PATTERN = re.compile(r"(\*\*[^*\n]+\*\*|`[^`\n]+`)")


class TooltipManager:
    """Display delayed contextual hints for registered widgets and buttons."""

    def __init__(self, root, delay_ms=500):
        """Bind tooltip discovery to all widgets owned by one Tk root."""
        self.root = root
        self.delay_ms = delay_ms
        self._texts = {}
        self._after_id = None
        self._tooltip = None
        self._widget = None
        root.bind_all("<Enter>", self._on_enter, add="+")
        root.bind_all("<Leave>", self._on_leave, add="+")
        root.bind_all("<ButtonPress>", self._on_leave, add="+")

    def register(self, widget, text):
        """Associate one widget with its concise hint text."""
        if text:
            self._texts[str(widget)] = text

    def get_text(self, widget):
        """Return the explicit or button-text hint for one widget."""
        explicit_text = self._texts.get(str(widget))
        if explicit_text:
            return explicit_text
        if not isinstance(widget, tk.Misc):
            return None
        try:
            if widget.winfo_class() == "TButton":
                return BUTTON_TOOLTIPS.get(widget.cget("text"))
        except tk.TclError:
            return None
        return None

    def _on_enter(self, event):
        """Schedule a hint when the entered widget has one."""
        self._hide()
        text = self.get_text(event.widget)
        if not text:
            return
        self._widget = event.widget
        self._after_id = self.root.after(
            self.delay_ms,
            lambda: self._show(event.widget, text),
        )

    def _on_leave(self, event=None):
        """Cancel pending display and hide the current hint."""
        self._hide()

    def _show(self, widget, text):
        """Show one borderless hint near the pointer."""
        self._after_id = None
        try:
            if self._widget is not widget or not widget.winfo_exists():
                return
            x = widget.winfo_pointerx() + 14
            y = widget.winfo_pointery() + 18
        except tk.TclError:
            return
        self._tooltip = tk.Toplevel(self.root)
        self._tooltip.wm_overrideredirect(True)
        self._tooltip.wm_geometry(f"+{x}+{y}")
        tk.Label(
            self._tooltip,
            text=text,
            background="#fffbd6",
            foreground="#222222",
            relief=tk.SOLID,
            borderwidth=1,
            padx=7,
            pady=4,
            justify=tk.LEFT,
            wraplength=360,
        ).pack()

    def _hide(self):
        """Remove pending and visible tooltip state."""
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None
        if self._tooltip is not None:
            try:
                self._tooltip.destroy()
            except tk.TclError:
                pass
            self._tooltip = None
        self._widget = None


class LightweightMarkdownView(ttk.Frame):
    """Render a small documented Markdown subset in a read-only Tk text view."""

    def __init__(self, parent, height=16):
        """Create a scrollable Markdown text view."""
        super().__init__(parent)
        self.text = tk.Text(
            self,
            height=height,
            wrap=tk.WORD,
            state=tk.DISABLED,
            padx=14,
            pady=12,
            relief=tk.FLAT,
            borderwidth=0,
            cursor="arrow",
        )
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.text.yview)
        self.text.configure(yscrollcommand=scrollbar.set)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._configure_tags()

    def _configure_tags(self):
        """Configure visual styles for the supported Markdown elements."""
        default_font = tkfont.nametofont("TkDefaultFont")
        fixed_font = tkfont.nametofont("TkFixedFont")
        family = default_font.actual("family")
        size = int(default_font.actual("size"))
        self._fonts = {
            "heading1": tkfont.Font(family=family, size=size + 5, weight="bold"),
            "heading2": tkfont.Font(family=family, size=size + 2, weight="bold"),
            "bold": tkfont.Font(family=family, size=size, weight="bold"),
            "formula": tkfont.Font(
                family=fixed_font.actual("family"),
                size=int(fixed_font.actual("size")) + 1,
            ),
        }
        self.text.tag_configure(
            "heading1",
            font=self._fonts["heading1"],
            foreground="#18324a",
            spacing1=4,
            spacing3=10,
        )
        self.text.tag_configure(
            "heading2",
            font=self._fonts["heading2"],
            foreground="#28536f",
            spacing1=8,
            spacing3=5,
        )
        self.text.tag_configure(
            "bold",
            font=self._fonts["bold"],
        )
        self.text.tag_configure(
            "code",
            font=fixed_font,
            background="#eef2f4",
            foreground="#243746",
        )
        self.text.tag_configure(
            "formula",
            font=self._fonts["formula"],
            background="#f2f6f8",
            foreground="#17384d",
            lmargin1=18,
            lmargin2=18,
            rmargin=18,
            spacing1=6,
            spacing3=8,
        )
        self.text.tag_configure(
            "bullet",
            lmargin1=16,
            lmargin2=32,
            spacing1=2,
            spacing3=2,
        )
        self.text.tag_configure("paragraph", spacing3=7)

    def _insert_inline(self, text, block_tag):
        """Insert bold and code spans while preserving the block style."""
        position = 0
        for match in INLINE_MARKDOWN_PATTERN.finditer(text):
            if match.start() > position:
                self.text.insert(tk.END, text[position:match.start()], (block_tag,))
            token = match.group(0)
            inline_tag = "bold" if token.startswith("**") else "code"
            marker_width = 2 if inline_tag == "bold" else 1
            self.text.insert(
                tk.END,
                token[marker_width:-marker_width],
                (block_tag, inline_tag),
            )
            position = match.end()
        if position < len(text):
            self.text.insert(tk.END, text[position:], (block_tag,))

    def set_markdown(self, markdown_text):
        """Replace the view with rendered lightweight Markdown text."""
        self.text.configure(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        lines = markdown_text.strip().splitlines()
        index = 0
        while index < len(lines):
            stripped = lines[index].strip()
            if not stripped:
                index += 1
                continue
            if stripped.startswith("# "):
                self._insert_inline(stripped[2:], "heading1")
                self.text.insert(tk.END, "\n", ("heading1",))
                index += 1
                continue
            if stripped.startswith("## "):
                self._insert_inline(stripped[3:], "heading2")
                self.text.insert(tk.END, "\n", ("heading2",))
                index += 1
                continue
            if stripped.startswith("$$") and stripped.endswith("$$"):
                self.text.insert(tk.END, stripped[2:-2].strip() + "\n", ("formula",))
                index += 1
                continue
            if stripped.startswith("- "):
                bullet_parts = [stripped[2:]]
                index += 1
                while index < len(lines) and lines[index].startswith("  "):
                    bullet_parts.append(lines[index].strip())
                    index += 1
                self.text.insert(tk.END, "\N{BULLET} ", ("bullet",))
                self._insert_inline(" ".join(bullet_parts), "bullet")
                self.text.insert(tk.END, "\n", ("bullet",))
                continue

            paragraph_parts = [stripped]
            index += 1
            while index < len(lines):
                next_line = lines[index].strip()
                if (
                    not next_line
                    or next_line.startswith("# ")
                    or next_line.startswith("## ")
                    or next_line.startswith("- ")
                    or (next_line.startswith("$$") and next_line.endswith("$$"))
                ):
                    break
                paragraph_parts.append(next_line)
                index += 1
            self._insert_inline(" ".join(paragraph_parts), "paragraph")
            self.text.insert(tk.END, "\n", ("paragraph",))
        self.text.configure(state=tk.DISABLED)
        self.text.yview_moveto(0.0)


def _load_model_description(model_name):
    """Load and validate one model's lightweight Markdown description."""
    model_class = load_model_class(model_name)
    description = model_class.description()
    if not isinstance(description, str) or not description.strip():
        raise ModelLoadError(f"{model_class.__name__}.description() must return non-empty text")
    return description


def _normalize_model_setting(value, metadata):
    """Return one declared setting value corrected to type and bounds."""
    setting_type = metadata["type"]
    is_valid = {
        "int": isinstance(value, int) and not isinstance(value, bool),
        "float": isinstance(value, (int, float)) and not isinstance(value, bool),
        "str": isinstance(value, str),
        "bool": isinstance(value, bool),
    }[setting_type]
    if not is_valid:
        return metadata["default"], True

    normalized = float(value) if setting_type == "float" else value
    if "min" in metadata and normalized < metadata["min"]:
        return metadata["min"], True
    if "max" in metadata and normalized > metadata["max"]:
        return metadata["max"], True
    return normalized, False


def _prepare_model_configuration(config, model_name):
    """Return config merged with one model's declared settings and corrections."""
    model_class = load_model_class(model_name)
    metadata = validate_model_metadata(model_class)
    normalized_config = dict(config)
    normalized_config["model"] = model_name
    corrected_names = []
    for name, setting_metadata in metadata["settings"].items():
        if name not in normalized_config:
            normalized_config[name] = setting_metadata["default"]
            continue
        normalized_value, corrected = _normalize_model_setting(
            normalized_config[name], setting_metadata,
        )
        if corrected:
            normalized_config[name] = normalized_value
            corrected_names.append(name)
    return normalized_config, metadata, corrected_names


class SimulationGUI:
    """Tkinter GUI for population simulation control and visualization"""

    def __init__(self, root):
        """Initialize GUI
        
        Args:
            root: tkinter root window
        """
        self.root = root
        self.root.title("Chimp Evolution Simulator")
        self.root.geometry("1200x900")
        self.is_closing = False
        self.progress_window = None
        self.tooltips = TooltipManager(self.root)

        self.experiment_manager = ExperimentManager(Path.cwd())
        self.experiment_dir = None
        self._bootstrap_experiment_location()
        self.config_file = str(self.config_file)
        self.config = self._load_config()
        self.available_models = discover_models()
        self.config, self.model_metadata, _ = _prepare_model_configuration(
            self.config,
            self.config["model"],
        )
        self._set_config_dirty(False)
        self._loading_ui = False
        self.simulation = None
        self.is_running = False
        self.finalize_requested = threading.Event()
        self.notebook = None  # Will reference the tab control
        
        # Image storage for rescaling
        self.distribution_original_img = None
        self.survivorship_original_img = None
        self.betaoccurrence_original_img = None
        self.popup_original_img = None
        self.distribution_photo = None
        self.survivorship_photo = None
        self.betaoccurrence_photo = None
        self._popup_photo = None  # Store popup image to prevent GC
        self._rescale_jobs = {}
        self._gui_log_messages = queue.Queue()
        self._pending_graph_update = None
        self._pending_final_graph_update = None
        self._pending_summary_path = None
        self._pending_simulation_error = None
        self._simulation_finished = False
        
        # Set logging callback
        set_logger(self._log_to_gui)
        
        # Build UI
        self._create_widgets()
        self._load_config_to_ui()
        self._schedule_gui_refresh()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        log("GUI initialized")

    def _bootstrap_experiment_location(self):
        """Select the current experiment from default.conf or the first available data directory."""
        experiment_name = self.experiment_manager.get_active_experiment_name()
        if experiment_name:
            experiment_dir = self.experiment_manager.get_active_experiment_dir()
            if experiment_dir is not None and experiment_dir.is_dir():
                self.experiment_dir = experiment_dir
                self.config_file = str(experiment_dir / "config.json")
                return

        data_dir = Path("data")
        if data_dir.exists():
            experiment_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir())
            if experiment_dirs:
                self.experiment_manager.set_active_experiment(experiment_dirs[0].name)
                self.experiment_dir = experiment_dirs[0]
                self.config_file = str(experiment_dirs[0] / "config.json")
                return

        self.experiment_dir = None
        self.config_file = "config.json"

    def _result_root(self):
        """Return the active experiment's result directory."""
        experiment_dir = getattr(self, "experiment_dir", None)
        return experiment_dir / "result" if experiment_dir is not None else Path("result")

    def _load_config(self, file_path=None):
        """Load one JSON object merged with default settings."""
        target_file = Path(file_path or self.config_file)
        if not target_file.exists():
            return DEFAULT_SETTINGS.copy()
        with target_file.open() as config_file:
            loaded_config = json.load(config_file)
        if not isinstance(loaded_config, dict):
            raise ValueError("Configuration must be a JSON object")
        return {**DEFAULT_SETTINGS, **loaded_config}

    def _save_config(self, file_path=None):
        """Save configuration to file

        Args:
            file_path: optional target path; defaults to current config file
        """
        target_file = Path(file_path or self.config_file)
        atomic_write_text(target_file, json.dumps(self.config, indent=2))
        if target_file == Path(self.config_file):
            self._set_config_dirty(False)
        log(f"Config saved to {target_file}")

    def _set_config_dirty(self, is_dirty):
        """Set config dirty state and synchronize its top-panel indicator."""
        self.is_config_dirty = is_dirty
        if hasattr(self, "config_dirty_var"):
            self.config_dirty_var.set("Config modified" if is_dirty else "Config saved")
        if hasattr(self, "config_dirty_label"):
            self.config_dirty_label.configure(foreground="#0067c0" if is_dirty else "#666666")

    def _set_batch_dirty(self, is_dirty):
        """Set batch dirty state and synchronize its top-panel indicator."""
        self.is_batch_dirty = is_dirty
        if hasattr(self, "batch_dirty_var"):
            self.batch_dirty_var.set("Batch modified" if is_dirty else "Batch saved")
        if hasattr(self, "batch_dirty_label"):
            self.batch_dirty_label.configure(foreground="#0067c0" if is_dirty else "#666666")

    def _mark_config_dirty(self, *args):
        """Mark configuration memory as modified by an editable control."""
        if self._loading_ui:
            return
        self._set_config_dirty(True)

    def _on_reread_config(self):
        """Discard GUI config edits and reload the experiment config file."""
        if self.is_config_dirty and not messagebox.askyesno(
            "Discard configuration changes",
            "Discard unsaved configuration changes and re-read config.json?",
        ):
            return
        self._restore_canonical_config()

    def _create_widgets(self):
        """Build GUI layout with tabs and controls"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        experiment_frame = ttk.Frame(main_frame)
        experiment_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(experiment_frame, text="Experiment").pack(side=tk.LEFT)
        self.experiment_var = tk.StringVar(
            value=self.experiment_manager.get_active_experiment_name() or "",
        )
        self.experiment_selector = ttk.Combobox(
            experiment_frame,
            textvariable=self.experiment_var,
            values=self.experiment_manager.list_experiments(),
            state="readonly",
            width=30,
        )
        self.experiment_selector.pack(side=tk.LEFT, padx=5)
        self.tooltips.register(
            self.experiment_selector,
            "Select the active experiment and its configuration, batch, and results.",
        )
        self.experiment_selector.bind(
            "<<ComboboxSelected>>",
            self._on_experiment_selected,
        )
        ttk.Button(
            experiment_frame,
            text="New Experiment",
            command=self._on_new_experiment,
        ).pack(side=tk.LEFT, padx=5)
        self.clone_experiment_button = ttk.Button(
            experiment_frame,
            text="Clone...",
            command=self._on_clone_experiment,
        )
        self.clone_experiment_button.pack(side=tk.LEFT, padx=5)
        ttk.Button(
            experiment_frame,
            text="Delete Experiment",
            command=self._on_delete_experiment,
        ).pack(side=tk.LEFT, padx=5)
        self.open_experiment_dir_button = ttk.Button(
            experiment_frame,
            text="Open In File Explorer",
            command=self._on_open_experiment_dir,
        )
        self.open_experiment_dir_button.pack(side=tk.LEFT, padx=5)
        self.config_dirty_var = tk.StringVar(value="Config saved")
        self.config_dirty_label = tk.Label(
            experiment_frame,
            textvariable=self.config_dirty_var,
            foreground="#666666",
        )
        self.config_dirty_label.pack(side=tk.RIGHT, padx=8)
        self.tooltips.register(
            self.config_dirty_label,
            "Shows whether the configuration has unsaved changes.",
        )
        self.batch_dirty_var = tk.StringVar(value="Batch saved")
        self.batch_dirty_label = tk.Label(
            experiment_frame,
            textvariable=self.batch_dirty_var,
            foreground="#666666",
        )
        self.batch_dirty_label.pack(side=tk.RIGHT, padx=8)
        self.tooltips.register(
            self.batch_dirty_label,
            "Shows whether the batch table has unsaved changes.",
        )
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Settings
        settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(settings_tab, text="Settings")
        self.settings_canvas = tk.Canvas(settings_tab, highlightthickness=0)
        settings_scrollbar = ttk.Scrollbar(
            settings_tab,
            orient="vertical",
            command=self.settings_canvas.yview,
        )
        self.settings_canvas.configure(yscrollcommand=settings_scrollbar.set)
        settings_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.settings_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        settings_frame = ttk.Frame(self.settings_canvas)
        settings_window = self.settings_canvas.create_window(
            (0, 0),
            window=settings_frame,
            anchor=tk.NW,
        )
        settings_frame.bind(
            "<Configure>",
            lambda event: self.settings_canvas.configure(
                scrollregion=self.settings_canvas.bbox("all"),
            ),
        )
        self.settings_canvas.bind(
            "<Configure>",
            lambda event: self.settings_canvas.itemconfigure(settings_window, width=event.width),
        )
        self._create_settings_tab(settings_frame)
        
        batch_frame = ttk.Frame(self.notebook)
        self.notebook.add(batch_frame, text="Batch")
        self.batch_tab = batch_frame
        self._create_batch_tab(batch_frame)
        
        self._create_progress_window()

    def _create_settings_tab(self, parent):
        """Create settings input fields organized by groups"""
        model_frame = ttk.LabelFrame(parent, text="Model", padding=10)
        model_frame.pack(fill=tk.X, pady=5, padx=5)
        self.model_var = tk.StringVar(value=self.config["model"])
        self.model_selector = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            values=self.available_models,
            state="normal",
            width=30,
        )
        self.model_selector.pack(side=tk.LEFT)
        self.tooltips.register(
            self.model_selector,
            PARAMETER_DESCRIPTIONS["model"],
        )
        self.model_selector.bind("<<ComboboxSelected>>", self._on_model_selected)
        self.model_var.trace_add("write", self._mark_config_dirty)
        ttk.Button(model_frame, text="Load Model", command=self._on_model_selected).pack(side=tk.LEFT, padx=5)
        self.about_model_button = ttk.Button(
            model_frame,
            text="About Model...",
            command=self._on_about_model,
        )
        self.about_model_button.pack(side=tk.LEFT, padx=5)
        ttk.Button(
            model_frame,
            text="Load All Model Defaults",
            command=self._load_current_model_defaults,
        ).pack(side=tk.LEFT, padx=5)

        # Device selection
        device_frame = ttk.LabelFrame(parent, text="Device", padding=10)
        device_frame.pack(fill=tk.X, pady=5, padx=5)
        
        available_cuda = torch.cuda.is_available()
        selected_device = self.config.get("device", "cuda" if available_cuda else "cpu")
        if not available_cuda:
            selected_device = "cpu"
        self.device_var = tk.StringVar(value=selected_device)
        self.device_var.trace_add("write", self._mark_config_dirty)
        
        cuda_button = ttk.Radiobutton(
            device_frame,
            text="CUDA (GPU)" if available_cuda else "CUDA (not available)",
            variable=self.device_var,
            value="cuda",
            state=tk.NORMAL if available_cuda else tk.DISABLED,
        )
        cuda_button.pack(anchor=tk.W)
        cpu_button = ttk.Radiobutton(
            device_frame,
            text="CPU",
            variable=self.device_var,
            value="cpu",
        )
        cpu_button.pack(anchor=tk.W)
        self.tooltips.register(cuda_button, PARAMETER_DESCRIPTIONS["device"])
        self.tooltips.register(cpu_button, PARAMETER_DESCRIPTIONS["device"])
        
        # Performance note
        note_text = ("Note: GPU acceleration is effective only for populations of ~1 million or more. "
                     "For populations under 500,000 animals, CPU is faster due to data transfer overhead. "
                     "If using GPU, increase stat_generation_period and graph_generation_period to reduce transfers.")
        note_label = ttk.Label(device_frame, text=note_text, foreground="gray", 
                              font=("TkDefaultFont", 8), wraplength=400, justify=tk.LEFT)
        note_label.pack(anchor=tk.W, pady=(5, 0))
        
        self.setting_vars = {}
        self.core_setting_widgets = {}

        core_frame = ttk.LabelFrame(parent, text="Core Settings", padding=10)
        core_frame.pack(fill=tk.X, padx=5, pady=5)
        for row, name in enumerate(CORE_SETTING_NAMES):
            name_label = ttk.Label(core_frame, text=name, width=28)
            name_label.grid(row=row, column=0, sticky=tk.W)
            value_var = tk.StringVar(value=str(self.config[name]))
            self.setting_vars[name] = value_var
            value_var.trace_add("write", self._mark_config_dirty)
            value_entry = ttk.Entry(core_frame, textvariable=value_var, width=15)
            value_entry.grid(row=row, column=1, padx=5)
            minimum, maximum = PARAMETER_RANGES[name]
            bounds_label = ttk.Label(
                core_frame,
                text=f"{minimum} <= x <= {maximum}",
                foreground="gray",
            )
            bounds_label.grid(row=row, column=2, sticky=tk.W)
            description = PARAMETER_DESCRIPTIONS[name]
            self.core_setting_widgets[name] = (name_label, value_entry, bounds_label)
            for widget in self.core_setting_widgets[name]:
                self.tooltips.register(widget, description)

        self.model_settings_frame = ttk.LabelFrame(parent, text="Model Settings", padding=10)
        self.model_settings_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.model_setting_rows = {}
        self._rebuild_model_settings_grid()
        
        # Tag field
        tag_frame = ttk.Frame(parent)
        tag_frame.pack(fill=tk.X, padx=5, pady=5)
        tag_label = ttk.Label(tag_frame, text="Run tag", width=25)
        tag_label.pack(side=tk.LEFT)
        self.tag_var = tk.StringVar(value=self.config.get("tag", "default"))
        self.tag_var.trace_add("write", self._mark_config_dirty)
        tag_entry = ttk.Entry(tag_frame, textvariable=self.tag_var, width=15)
        tag_entry.pack(side=tk.LEFT, padx=5)
        self.tooltips.register(tag_label, PARAMETER_DESCRIPTIONS["tag"])
        self.tooltips.register(tag_entry, PARAMETER_DESCRIPTIONS["tag"])
        self.start_btn = ttk.Button(tag_frame, text="Start Simulation", command=self._start_simulation)
        self.start_btn.pack(side=tk.LEFT, padx=(15, 5))
        self.stop_btn = ttk.Button(tag_frame, text="Stop", command=self._stop_simulation, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        ttk.Button(tag_frame, text="Save Config", command=self._on_save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(tag_frame, text="Save Config As...", command=self._on_save_config_as).pack(side=tk.LEFT, padx=5)
        ttk.Button(tag_frame, text="Load Config ...", command=self._on_load_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(tag_frame, text="Re-read Config", command=self._on_reread_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            tag_frame,
            text="Show Progress Window",
            command=self._show_progress_window,
        ).pack(side=tk.LEFT, padx=5)

    def _read_experiment_state(self, experiment_name):
        """Return validated config and batch paths for one experiment."""
        experiment_dir = self.experiment_manager.project_root / "data" / experiment_name
        config_path = experiment_dir / "config.json"
        if not config_path.is_file():
            raise ValueError(f"Experiment is missing config.json: {experiment_name}")
        loaded_config = self._load_config(config_path)
        config, metadata, _ = _prepare_model_configuration(
            loaded_config,
            loaded_config["model"],
        )
        batch_path = experiment_dir / "multi.csv"
        if batch_path.exists():
            with batch_path.open(newline="", encoding="utf-8") as batch_file:
                reader = csv.DictReader(batch_file)
                if not reader.fieldnames or reader.fieldnames[0] != "tag":
                    raise ValueError("Batch CSV must have tag as its first column")
                list(reader)
        return experiment_dir, config_path, batch_path, config, metadata

    def _create_experiment(self, experiment_name, model_name):
        """Create and activate an experiment from one model's defaults."""
        model_class = load_model_class(model_name)
        metadata = validate_model_metadata(model_class)
        config = {
            **DEFAULT_SETTINGS,
            **{name: details["default"] for name, details in metadata["settings"].items()},
            "model": model_name,
        }
        experiment_dir = self.experiment_manager.create_experiment(
            experiment_name,
            config,
            model_class.add_batch(),
            activate=False,
        )
        return experiment_dir

    def _prompt_new_experiment(self):
        """Show the new-experiment form and return its created directory or None."""
        dialog = tk.Toplevel(self.root)
        dialog.title("New Experiment")
        dialog.geometry("700x600")
        dialog.minsize(620, 480)
        dialog.transient(self.root)
        dialog.columnconfigure(1, weight=1)
        dialog.rowconfigure(3, weight=1)

        explanation = tk.Message(
            dialog,
            text=(
                "Experiments keep independent configurations, batch tables, and results "
                "and can be switched at any time. Each experiment is stored in "
                "data/<name>/. The default.conf file contains the name of the active "
                "experiment."
            ),
            width=500,
            justify=tk.LEFT,
            foreground="#444444",
        )
        explanation.grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=20,
            pady=(20, 12),
        )

        ttk.Label(dialog, text="Experiment name:").grid(
            row=1,
            column=0,
            sticky=tk.W,
            padx=(20, 10),
            pady=10,
        )
        experiment_name_var = tk.StringVar()
        name_entry = ttk.Entry(dialog, textvariable=experiment_name_var)
        name_entry.grid(row=1, column=1, sticky="ew", padx=(0, 20), pady=10)

        ttk.Label(dialog, text="Model:").grid(
            row=2,
            column=0,
            sticky=tk.W,
            padx=(20, 10),
            pady=10,
        )
        default_model = DEFAULT_SETTINGS["model"]
        if default_model not in self.available_models and self.available_models:
            default_model = self.available_models[0]
        model_name_var = tk.StringVar(value=default_model)
        model_selector = ttk.Combobox(
            dialog,
            textvariable=model_name_var,
            values=self.available_models,
            state="readonly",
        )
        model_selector.grid(row=2, column=1, sticky="ew", padx=(0, 20), pady=10)

        description_frame = ttk.LabelFrame(dialog, text="Model description", padding=6)
        description_frame.grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="nsew",
            padx=20,
            pady=(4, 8),
        )
        description_view = LightweightMarkdownView(description_frame, height=14)
        description_view.pack(fill=tk.BOTH, expand=True)

        def update_description(event=None):
            """Render the currently selected model description in the form."""
            model_name = model_name_var.get().strip()
            try:
                description = _load_model_description(model_name)
            except (ModelLoadError, ValueError) as error:
                description = f"# Description unavailable\n\n{error}"
            description_view.set_markdown(description)

        model_selector.bind("<<ComboboxSelected>>", update_description)
        update_description()

        result = []

        def cancel():
            """Close the dialog without creating an experiment."""
            dialog.destroy()

        def create():
            """Validate the selected inputs and close after successful creation."""
            try:
                experiment_dir = self._create_experiment(
                    experiment_name_var.get().strip(),
                    model_name_var.get().strip(),
                )
            except (ModelLoadError, OSError, ValueError) as error:
                messagebox.showerror("Experiment error", str(error), parent=dialog)
                return
            result.append(experiment_dir)
            dialog.destroy()

        buttons = ttk.Frame(dialog)
        buttons.grid(
            row=4,
            column=0,
            columnspan=2,
            sticky=tk.SE,
            padx=20,
            pady=(18, 16),
        )
        ttk.Button(buttons, text="Create", command=create).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(buttons, text="Cancel", command=cancel).pack(side=tk.LEFT)
        dialog.protocol("WM_DELETE_WINDOW", cancel)
        dialog.bind("<Return>", lambda event: create())
        dialog.bind("<Escape>", lambda event: cancel())
        name_entry.focus_set()
        self.root.update_idletasks()
        dialog.update_idletasks()
        dialog_x = self.root.winfo_rootx() + (
            self.root.winfo_width() - dialog.winfo_width()
        ) // 2
        dialog_y = self.root.winfo_rooty() + (
            self.root.winfo_height() - dialog.winfo_height()
        ) // 2
        dialog.geometry(f"+{max(0, dialog_x)}+{max(0, dialog_y)}")
        dialog.grab_set()
        self.root.wait_window(dialog)
        return result[0] if result else None

    def _on_about_model(self):
        """Show a modal description of the model named in the main selector."""
        model_name = self.model_var.get().strip()
        try:
            description = _load_model_description(model_name)
        except (ModelLoadError, ValueError) as error:
            messagebox.showerror("Model error", f"Could not load model description: {error}")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title(f"About Model: {model_name}")
        dialog.geometry("700x560")
        dialog.minsize(520, 400)
        dialog.transient(self.root)
        description_view = LightweightMarkdownView(dialog, height=20)
        description_view.pack(fill=tk.BOTH, expand=True, padx=12, pady=(12, 6))
        description_view.set_markdown(description)
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(
            anchor=tk.E,
            padx=12,
            pady=(6, 12),
        )
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        dialog.bind("<Escape>", lambda event: dialog.destroy())
        dialog.grab_set()
        self.root.wait_window(dialog)

    def _on_new_experiment(self):
        """Create a new experiment and switch the GUI to it."""
        if self.is_running or self.is_batch_running:
            messagebox.showwarning("Experiment busy", "Stop the active calculation before creating an experiment.")
            return
        if not self._confirm_experiment_transition():
            return
        try:
            experiment_dir = self._prompt_new_experiment()
        except (ModelLoadError, OSError, ValueError) as error:
            messagebox.showerror("Experiment error", str(error))
            return
        if experiment_dir is None:
            return
        self.experiment_selector.configure(values=self.experiment_manager.list_experiments())
        self.experiment_var.set(experiment_dir.name)
        self._activate_experiment(experiment_dir.name)

    def _prompt_clone_experiment(self, source_name):
        """Show the clone form and return its new experiment directory or None."""
        source_dir = (
            self.experiment_manager.project_root
            / self.experiment_manager.data_dir_name
            / source_name
        )
        result_dir = source_dir / "result"
        has_results = result_dir.is_dir() and any(result_dir.iterdir())

        dialog = tk.Toplevel(self.root)
        dialog.title("Clone Experiment")
        dialog.geometry("480x190")
        dialog.minsize(440, 170)
        dialog.transient(self.root)
        dialog.columnconfigure(1, weight=1)

        ttk.Label(dialog, text=f"Clone experiment: {source_name}").grid(
            row=0,
            column=0,
            columnspan=2,
            sticky=tk.W,
            padx=20,
            pady=(20, 8),
        )
        ttk.Label(dialog, text="New experiment name:").grid(
            row=1,
            column=0,
            sticky=tk.W,
            padx=(20, 10),
            pady=8,
        )
        experiment_name_var = tk.StringVar()
        name_entry = ttk.Entry(dialog, textvariable=experiment_name_var)
        name_entry.grid(row=1, column=1, sticky="ew", padx=(0, 20), pady=8)

        copy_results_var = tk.BooleanVar(value=False)
        copy_results_check = ttk.Checkbutton(
            dialog,
            text="Copy results",
            variable=copy_results_var,
            state=tk.NORMAL if has_results else tk.DISABLED,
        )
        copy_results_check.grid(
            row=2,
            column=1,
            sticky=tk.W,
            padx=(0, 20),
            pady=8,
        )

        result = []

        def cancel():
            """Close the dialog without cloning the experiment."""
            dialog.destroy()

        def clone():
            """Validate the form and close after a successful clone."""
            try:
                experiment_dir = self.experiment_manager.clone_experiment(
                    source_name,
                    experiment_name_var.get().strip(),
                    copy_results=copy_results_var.get(),
                    activate=False,
                )
            except (OSError, ValueError) as error:
                messagebox.showerror("Clone Experiment", str(error), parent=dialog)
                return
            result.append(experiment_dir)
            dialog.destroy()

        buttons = ttk.Frame(dialog)
        buttons.grid(
            row=3,
            column=0,
            columnspan=2,
            sticky=tk.SE,
            padx=20,
            pady=(12, 16),
        )
        ttk.Button(buttons, text="OK", command=clone).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(buttons, text="Cancel", command=cancel).pack(side=tk.LEFT)
        dialog.protocol("WM_DELETE_WINDOW", cancel)
        dialog.bind("<Return>", lambda event: clone())
        dialog.bind("<Escape>", lambda event: cancel())
        name_entry.focus_set()
        self.root.update_idletasks()
        dialog.update_idletasks()
        dialog_x = self.root.winfo_rootx() + (
            self.root.winfo_width() - dialog.winfo_width()
        ) // 2
        dialog_y = self.root.winfo_rooty() + (
            self.root.winfo_height() - dialog.winfo_height()
        ) // 2
        dialog.geometry(f"+{max(0, dialog_x)}+{max(0, dialog_y)}")
        dialog.grab_set()
        self.root.wait_window(dialog)
        return result[0] if result else None

    def _on_clone_experiment(self):
        """Clone the active experiment and switch the GUI to the completed copy."""
        source_name = self.experiment_manager.get_active_experiment_name()
        if not source_name:
            messagebox.showwarning("Clone Experiment", "No active experiment is selected.")
            return
        if self.is_running or self.is_batch_running:
            messagebox.showwarning(
                "Experiment busy",
                "Stop the active calculation before cloning an experiment.",
            )
            return
        if not self._confirm_experiment_transition():
            return
        try:
            experiment_dir = self._prompt_clone_experiment(source_name)
        except (OSError, ValueError) as error:
            messagebox.showerror("Clone Experiment", str(error))
            return
        if experiment_dir is None:
            return
        self.experiment_selector.configure(values=self.experiment_manager.list_experiments())
        self.experiment_var.set(experiment_dir.name)
        try:
            self._activate_experiment(experiment_dir.name)
        except (ModelLoadError, OSError, ValueError) as error:
            self.experiment_var.set(source_name)
            messagebox.showerror("Clone Experiment", f"Could not activate clone: {error}")

    def _on_delete_experiment(self):
        """Confirm and delete the active experiment, then select or create a replacement."""
        experiment_name = self.experiment_manager.get_active_experiment_name()
        if not experiment_name:
            messagebox.showwarning("Delete Experiment", "No active experiment is selected.")
            return
        if self.is_running or self.is_batch_running:
            messagebox.showwarning("Experiment busy", "Stop the active calculation before deleting an experiment.")
            return
        if not self._confirm_experiment_transition():
            return
        if not messagebox.askyesno(
            "Delete Experiment",
            f"Permanently delete experiment '{experiment_name}' and all of its files?",
        ):
            return
        try:
            self.experiment_manager.delete_experiment(experiment_name)
            remaining_experiments = self.experiment_manager.list_experiments()
            self.experiment_selector.configure(values=remaining_experiments)
            if remaining_experiments:
                self.experiment_var.set(remaining_experiments[0])
                self._activate_experiment(remaining_experiments[0])
                return
            experiment_dir = self._prompt_new_experiment()
            if experiment_dir is None:
                self.root.destroy()
                return
            self.experiment_selector.configure(values=self.experiment_manager.list_experiments())
            self.experiment_var.set(experiment_dir.name)
            self._activate_experiment(experiment_dir.name)
        except (ModelLoadError, OSError, ValueError) as error:
            messagebox.showerror("Experiment error", str(error))

    def _on_open_experiment_dir(self):
        """Open the selected experiment directory in the system file manager."""
        experiment_dir = (
            self.experiment_manager.project_root
            / self.experiment_manager.data_dir_name
            / self.experiment_var.get()
        )
        try:
            if sys.platform == "win32":
                os.startfile(experiment_dir)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(experiment_dir)])
            else:
                subprocess.Popen(["xdg-open", str(experiment_dir)])
        except OSError as error:
            messagebox.showerror(
                "Open Experiment",
                f"Could not open experiment directory:\n{experiment_dir}\n\n{error}",
            )

    def _confirm_experiment_transition(self):
        """Resolve unsaved config and batch changes before leaving an experiment."""
        if not self.is_config_dirty and not self.is_batch_dirty:
            return True
        choice = messagebox.askyesnocancel(
            "Unsaved changes",
            "Configuration or batch data for experiment "
            f"'{self.experiment_manager.get_active_experiment_name() or self.experiment_var.get()}' was modified. "
            "Save changes before leaving it?",
        )
        if choice is None:
            return False
        if not choice:
            return True
        try:
            if not self._update_config_from_ui():
                return False
            self._save_config()
            self._save_batch_csv()
        except (OSError, ValueError) as error:
            messagebox.showerror("Experiment error", str(error))
            return False
        return True

    def _activate_experiment(self, target_name):
        """Load one validated experiment into all GUI editors and selectors."""
        experiment_dir, config_path, batch_path, config, metadata = self._read_experiment_state(
            target_name,
        )
        self.experiment_dir = experiment_dir
        self.config_file = str(config_path)
        self.config = config
        self.model_metadata = metadata
        self._set_config_dirty(False)
        self._rebuild_model_settings_grid()
        self._rebuild_dynamic_graph_tabs()
        self.batch_column_selector.configure(values=self._batch_column_options(metadata))
        self._load_config_to_ui()
        self._load_batch_csv(batch_path)
        self.experiment_manager.set_active_experiment(target_name)
        self.batch_status_var.set(f"Ready: {self._aggregate_result_count()} completed")

    def _on_experiment_selected(self, event=None):
        """Switch experiments after resolving unsaved changes and validating files."""
        current_name = self.experiment_manager.get_active_experiment_name() or ""
        target_name = self.experiment_var.get().strip()
        if not target_name or target_name == current_name:
            return
        if self.is_running or self.is_batch_running:
            self.experiment_var.set(current_name)
            messagebox.showwarning("Experiment busy", "Stop the active calculation before switching experiments.")
            return
        if not self._confirm_experiment_transition():
            self.experiment_var.set(current_name)
            return
        try:
            experiment_dir, config_path, batch_path, config, metadata = self._read_experiment_state(
                target_name,
            )
        except (ModelLoadError, OSError, ValueError) as error:
            self.experiment_var.set(current_name)
            messagebox.showerror("Experiment error", f"Could not switch experiment: {error}")
            return

        self._activate_experiment(target_name)

    def _create_progress_window(self):
        """Create the persistent non-modal progress window and hide it initially."""
        self.progress_window = tk.Toplevel(self.root)
        self.progress_window.title("Progress")
        self.progress_window.geometry("1200x900")
        self.progress_window.protocol("WM_DELETE_WINDOW", self._hide_progress_window)
        progress_frame = ttk.Frame(self.progress_window, padding=5)
        progress_frame.pack(fill=tk.BOTH, expand=True)
        self._create_progress_content(progress_frame)
        self.progress_window.withdraw()

    def _create_progress_content(self, parent):
        """Create progress graphs, statistics, controls, and log output."""
        self.graph_notebook = ttk.Notebook(parent)
        self.graph_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        legacy_graph_frame = ttk.Frame(self.graph_notebook, padding=8)
        self.graph_notebook.add(legacy_graph_frame, text="Legacy")
        self.dynamic_graph_canvases = {}
        self.dynamic_graph_images = {}
        self.dynamic_graph_photos = {}
        self._rebuild_dynamic_graph_tabs()

        graphs_row = ttk.Frame(legacy_graph_frame)
        graphs_row.pack(fill=tk.BOTH, expand=True)
        graphs_row.columnconfigure(0, weight=1, uniform="graphs")
        graphs_row.columnconfigure(1, weight=1, uniform="graphs")
        graphs_row.columnconfigure(2, weight=1, uniform="graphs")
        graphs_row.rowconfigure(0, weight=1)

        # Left graph: Distribution
        left_graph = ttk.Frame(graphs_row)
        left_graph.grid(row=0, column=0, sticky="nsew", padx=(0, 3))
        left_graph.columnconfigure(0, weight=1)
        left_graph.rowconfigure(1, weight=1)
        ttk.Label(left_graph, text="Age Distribution").grid(row=0, column=0, sticky="w")
        self.distribution_canvas = tk.Canvas(left_graph, bg="white", highlightthickness=0)
        self.distribution_canvas.grid(row=1, column=0, sticky="nsew")
        self.distribution_canvas.create_text(10, 10, anchor=tk.NW, text="No distribution graph yet", fill="gray")
        self.distribution_canvas.bind(
            "<Configure>",
            lambda event: self._schedule_graph_rescale("distribution"),
        )

        # Middle graph: Survivorship
        middle_graph = ttk.Frame(graphs_row)
        middle_graph.grid(row=0, column=1, sticky="nsew", padx=(3, 3))
        middle_graph.columnconfigure(0, weight=1)
        middle_graph.rowconfigure(1, weight=1)
        ttk.Label(middle_graph, text="Survivorship Curve").grid(row=0, column=0, sticky="w")
        self.survivorship_canvas = tk.Canvas(middle_graph, bg="white", highlightthickness=0)
        self.survivorship_canvas.grid(row=1, column=0, sticky="nsew")
        self.survivorship_canvas.create_text(10, 10, anchor=tk.NW, text="No survivorship graph yet", fill="gray")
        self.survivorship_canvas.bind(
            "<Configure>",
            lambda event: self._schedule_graph_rescale("survivorship"),
        )

        # Right graph: Beta Occurrence
        right_graph = ttk.Frame(graphs_row)
        right_graph.grid(row=0, column=2, sticky="nsew", padx=(3, 0))
        right_graph.columnconfigure(0, weight=1)
        right_graph.rowconfigure(1, weight=1)
        ttk.Label(right_graph, text="Beta Distribution").grid(row=0, column=0, sticky="w")
        self.betaoccurrence_canvas = tk.Canvas(right_graph, bg="white", highlightthickness=0)
        self.betaoccurrence_canvas.grid(row=1, column=0, sticky="nsew")
        self.betaoccurrence_canvas.create_text(10, 10, anchor=tk.NW, text="No beta graph yet", fill="gray")
        self.betaoccurrence_canvas.bind(
            "<Configure>",
            lambda event: self._schedule_graph_rescale("betaoccurrence"),
        )
        
        calculation_frame = ttk.LabelFrame(parent, text="Current Calculation", padding=10)
        calculation_frame.pack(fill=tk.X, padx=5, pady=5)
        self.progress_tag_var = tk.StringVar(value="-")
        self.progress_source_var = tk.StringVar(value="No calculation")
        ttk.Label(calculation_frame, text="Tag:").grid(row=0, column=0, sticky=tk.W)
        self.progress_tag_label = ttk.Label(calculation_frame, textvariable=self.progress_tag_var)
        self.progress_tag_label.grid(
            row=0,
            column=1,
            sticky=tk.W,
            padx=(8, 20),
        )
        ttk.Label(calculation_frame, text="Source:").grid(row=0, column=2, sticky=tk.W)
        self.progress_source_label = ttk.Label(
            calculation_frame,
            textvariable=self.progress_source_var,
            wraplength=700,
        )
        self.progress_source_label.grid(row=0, column=3, sticky=tk.W, padx=(8, 0))
        self.tooltips.register(self.progress_tag_label, "Tag of the active calculation.")
        self.tooltips.register(
            self.progress_source_label,
            "Configuration source and batch values used by the active calculation.",
        )
        calculation_frame.columnconfigure(3, weight=1)

        # Performance statistics panel
        stats_frame = ttk.LabelFrame(parent, text="Performance Statistics", padding=10)
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill=tk.X)
        
        # Row 1: Elapsed time
        ttk.Label(stats_grid, text="Elapsed Time:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.stat_elapsed_time = ttk.Label(stats_grid, text="0.000 s", font=("TkDefaultFont", 9, "bold"))
        self.stat_elapsed_time.grid(row=0, column=1, sticky=tk.W)
        self.tooltips.register(self.stat_elapsed_time, "Elapsed wall-clock time for the active simulation.")
        
        # Row 2: Average iteration time
        ttk.Label(stats_grid, text="Avg Iteration Time:").grid(row=0, column=2, sticky=tk.W, padx=(20, 10))
        self.stat_avg_iteration = ttk.Label(stats_grid, text="0.000000 s", font=("TkDefaultFont", 9, "bold"))
        self.stat_avg_iteration.grid(row=0, column=3, sticky=tk.W)
        self.tooltips.register(self.stat_avg_iteration, "Average wall-clock time per completed year.")
        
        # Row 3: Average per-element time
        ttk.Label(stats_grid, text="Avg Per-Element Time:").grid(row=0, column=4, sticky=tk.W, padx=(20, 10))
        self.stat_avg_element = ttk.Label(stats_grid, text="0.000 μs", font=("TkDefaultFont", 9, "bold"))
        self.stat_avg_element.grid(row=0, column=5, sticky=tk.W)
        self.tooltips.register(self.stat_avg_element, "Average processing time per animal across completed years.")
        
        # Log output
        log_frame = ttk.LabelFrame(parent, text="Log Output", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.log_autoscroll_var = tk.BooleanVar(value=True)
        self.log_autoscroll_button = ttk.Checkbutton(
            log_frame,
            text="Auto-scroll log",
            variable=self.log_autoscroll_var,
        )
        self.log_autoscroll_button.pack(anchor=tk.W, pady=(0, 5))
        self.tooltips.register(
            self.log_autoscroll_button,
            "Keep the newest log message visible while calculations run.",
        )
        log_body = ttk.Frame(log_frame)
        log_body.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_body, height=8, width=80, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        scrollbar = ttk.Scrollbar(log_body, orient="vertical", command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        controls = ttk.Frame(parent)
        controls.pack(fill=tk.X, padx=5, pady=(0, 5))
        self.progress_stop_btn = ttk.Button(
            controls,
            text="Stop Simulation",
            command=self._stop_active_calculation,
            state=tk.DISABLED,
        )
        self.progress_stop_btn.pack(side=tk.RIGHT)
        self.progress_finalize_btn = ttk.Button(
            controls,
            text="Finalize Simulation",
            command=self._finalize_active_calculation,
            state=tk.DISABLED,
        )
        self.progress_finalize_btn.pack(side=tk.RIGHT, padx=(0, 5))

    def _set_progress_calculation(self, tag, batch_row=None):
        """Display the current run tag and its default-config or batch-row source."""
        self.progress_tag_var.set(tag)
        if batch_row is None:
            self.progress_source_var.set("Default config")
            return
        details = ", ".join(f"{name}={value}" for name, value in batch_row.items())
        self.progress_source_var.set(f"Batch row: {details}")

    def _show_progress_window(self):
        """Show and raise the persistent non-modal progress window."""
        self.progress_window.deiconify()
        self.progress_window.lift()

    def _hide_progress_window(self):
        """Hide the progress window without discarding its current state."""
        self.progress_window.withdraw()

    def _create_batch_tab(self, parent):
        """Create the editable in-memory batch CSV table."""
        self.batch_path = self.experiment_dir / "multi.csv" if self.experiment_dir is not None else Path("multi.csv")
        self.batch_columns = []
        self.batch_rows = []
        self._set_batch_dirty(False)
        self.is_batch_running = False
        self.batch_cancel_requested = False
        self.batch_keep_changed_tags = set()
        self.batch_status_var = tk.StringVar(value="Ready")
        edit_controls = ttk.Frame(parent)
        edit_controls.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(edit_controls, text="Add Row", command=self._add_batch_row).pack(side=tk.LEFT)
        ttk.Button(
            edit_controls,
            text="Load Model Defaults",
            command=self._load_current_model_batch_defaults,
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            edit_controls,
            text="Clear Results",
            command=self._clear_result_directory,
        ).pack(side=tk.LEFT, padx=5)
        self.batch_column_var = tk.StringVar()
        self.batch_column_selector = ttk.Combobox(
            edit_controls,
            textvariable=self.batch_column_var,
            values=self._batch_column_options(),
            state="readonly",
            width=24,
        )
        self.batch_column_selector.pack(side=tk.LEFT, padx=(15, 0))
        self.tooltips.register(
            self.batch_column_selector,
            "Select a configuration field to add to or remove from the batch table.",
        )
        ttk.Button(edit_controls, text="Add Column", command=self._add_batch_column).pack(side=tk.LEFT, padx=5)
        ttk.Button(edit_controls, text="Delete Column", command=self._delete_batch_column).pack(side=tk.LEFT, padx=5)
        self.batch_status_var.set(f"Ready: {self._aggregate_result_count()} completed")
        self.batch_status_label = ttk.Label(parent, textvariable=self.batch_status_var)
        self.batch_status_label.pack(anchor=tk.W, padx=8)
        self.tooltips.register(
            self.batch_status_label,
            "Shows batch readiness, row progress, cancellation, or failure state.",
        )
        self.batch_tree = ttk.Treeview(parent, show="headings")
        self.batch_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.batch_tree.bind("<Double-1>", self._edit_batch_cell)

        action_controls = ttk.Frame(parent)
        action_controls.pack(fill=tk.X, padx=5, pady=(0, 5))
        ttk.Button(action_controls, text="Save Batch", command=self._on_save_batch).pack(side=tk.LEFT)
        ttk.Button(action_controls, text="Re-read Batch", command=self._on_reread_batch).pack(side=tk.LEFT, padx=5)
        self.batch_start_button = ttk.Button(
            action_controls,
            text="Start Batch",
            command=self._on_start_batch,
        )
        self.batch_start_button.pack(side=tk.RIGHT)
        ttk.Button(
            action_controls,
            text="Run Selected Row",
            command=self._on_run_selected_row,
        ).pack(side=tk.RIGHT, padx=5)
        self.batch_stop_button = ttk.Button(
            action_controls,
            text="Stop Batch",
            command=self._stop_batch,
            state=tk.DISABLED,
        )
        self.batch_stop_button.pack(side=tk.RIGHT, padx=5)
        self._load_batch_csv(self.batch_path)

    def _batch_column_options(self, metadata=None):
        """Return batch-varying model and setting column choices."""
        active_metadata = metadata or self.model_metadata
        return ["model", *active_metadata["settings"]]

    def _load_batch_csv(self, file_path):
        """Load tag-first CSV rows into the editable batch table."""
        path = Path(file_path)
        if not path.exists():
            self.batch_columns = ["tag"]
            self.batch_rows = []
        else:
            with path.open(newline="", encoding="utf-8") as batch_file:
                reader = csv.DictReader(batch_file)
                columns = reader.fieldnames or ["tag"]
                if not columns or columns[0] != "tag":
                    raise ValueError("Batch CSV must have tag as its first column")
                self.batch_columns = columns
                self.batch_rows = [
                    {name: row.get(name, "") for name in columns}
                    for row in reader
                ]
        self.batch_path = path
        self._set_batch_dirty(False)
        self._render_batch_grid()

    def _load_batch_text(self, batch_text):
        """Load one model-provided batch CSV string into memory."""
        rows = list(csv.DictReader(batch_text.strip().splitlines())) if batch_text.strip() else []
        columns = list(rows[0]) if rows else (batch_text.strip().splitlines()[0].split(",") if batch_text.strip() else ["tag"])
        if not columns or columns[0] != "tag":
            raise ValueError("Model batch CSV must have tag as its first column")
        self.batch_columns = columns
        self.batch_rows = [{name: row.get(name, "") for name in columns} for row in rows]
        self._set_batch_dirty(True)
        self._render_batch_grid()

    def _confirm_duplicate_batch_rows(self):
        """Confirm raw-identical batch rows while keeping duplicate tags invalid."""
        duplicate_tag_groups = find_duplicate_variant_tags(
            self.batch_columns,
            self.batch_rows,
        )
        if not duplicate_tag_groups:
            return True
        warning = format_duplicate_variant_warning(duplicate_tag_groups)
        return messagebox.askokcancel(
            "Duplicate batch rows",
            f"{warning}. Continue?",
        )

    def _save_batch_csv(self, file_path=None, confirm_duplicates=True):
        """Write the current valid batch table to CSV."""
        if not self.batch_columns or self.batch_columns[0] != "tag":
            raise ValueError("Batch CSV must have tag as its first column")
        tags = [row.get("tag", "").strip() for row in self.batch_rows]
        if any(not tag for tag in tags) or len(tags) != len(set(tags)):
            raise ValueError("Batch tags must be non-empty and unique")
        if confirm_duplicates and not self._confirm_duplicate_batch_rows():
            return False
        path = Path(file_path or self.batch_path)
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=self.batch_columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(self.batch_rows)
        atomic_write_text(path, output.getvalue())
        self.batch_path = path
        self._set_batch_dirty(False)
        return True

    def _render_batch_grid(self):
        """Render current batch rows and columns into the Treeview."""
        self.batch_tree.delete(*self.batch_tree.get_children())
        self.batch_tree.configure(columns=self.batch_columns)
        for name in self.batch_columns:
            self.batch_tree.heading(name, text=name)
            self.batch_tree.column(name, width=150, minwidth=80, stretch=True)
        for index, row in enumerate(self.batch_rows):
            self.batch_tree.insert("", tk.END, iid=str(index), values=[row.get(name, "") for name in self.batch_columns])

    def _add_batch_row(self):
        """Add one blank editable row to the current batch table."""
        self.batch_rows.append({name: "" for name in self.batch_columns})
        self._set_batch_dirty(True)
        self._render_batch_grid()

    def _add_batch_column(self):
        """Add the selected model setting as an editable batch column."""
        name = self.batch_column_var.get()
        if not name or name in self.batch_columns:
            return
        self.batch_columns.append(name)
        for row in self.batch_rows:
            row[name] = ""
        self._set_batch_dirty(True)
        self._render_batch_grid()

    def _delete_batch_column(self):
        """Delete the selected optional batch column while preserving required tag."""
        name = self.batch_column_var.get()
        if not name or name not in self.batch_columns:
            return
        if name == "tag":
            messagebox.showwarning("Batch column", "The required tag column cannot be deleted.")
            return
        self.batch_columns.remove(name)
        for row in self.batch_rows:
            row.pop(name, None)
        self.batch_column_var.set("")
        self._set_batch_dirty(True)
        self._render_batch_grid()

    def _edit_batch_cell(self, event):
        """Edit one batch Treeview cell in place after a double click."""
        item_id = self.batch_tree.identify_row(event.y)
        column_id = self.batch_tree.identify_column(event.x)
        if not item_id or not column_id:
            return
        column_index = int(column_id[1:]) - 1
        column_name = self.batch_columns[column_index]
        x, y, width, height = self.batch_tree.bbox(item_id, column_id)
        value_var = tk.StringVar(value=self.batch_rows[int(item_id)].get(column_name, ""))
        editor = ttk.Entry(self.batch_tree, textvariable=value_var)
        editor.place(x=x, y=y, width=width, height=height)
        editor.focus_set()

        def commit_edit(event=None):
            self.batch_rows[int(item_id)][column_name] = value_var.get()
            self._set_batch_dirty(True)
            editor.destroy()
            self._render_batch_grid()

        editor.bind("<Return>", commit_edit)
        editor.bind("<FocusOut>", commit_edit)

    def _on_save_batch(self):
        """Save the editable batch table and report validation errors."""
        try:
            if not self._save_batch_csv():
                return
        except (OSError, ValueError) as error:
            messagebox.showerror("Batch error", str(error))
            return
        messagebox.showinfo("Success", f"Batch saved to {self.batch_path}")

    def _on_reread_batch(self):
        """Discard in-memory batch edits and reload the experiment CSV."""
        if self.is_batch_dirty and not messagebox.askyesno(
            "Discard batch changes",
            "Discard unsaved batch changes and re-read multi.csv?",
        ):
            return
        try:
            self._load_batch_csv(self.batch_path)
        except (OSError, ValueError) as error:
            messagebox.showerror("Batch error", str(error))

    def _confirm_saved_batch_inputs(self):
        """Save dirty config and batch inputs after explicit confirmation."""
        if self.is_config_dirty and not messagebox.askyesno(
            "Unsaved configuration",
            "Save configuration before starting?",
        ):
            return False
        if self.is_config_dirty:
            self._save_config()
        if self.is_batch_dirty and not messagebox.askyesno(
            "Unsaved batch",
            "Save batch changes before starting?",
        ):
            return False
        if self.is_batch_dirty:
            self._save_batch_csv(confirm_duplicates=False)
        return True

    def _confirm_batch_configuration_changes(self, selected_tag=None):
        """Return (had_changes, keep_tags) after a dialog choice, or None on cancel."""
        configuration_changes = inspect_batch_configuration_changes(
            self.batch_path,
            self.config_file,
            result_dir=self._result_root(),
            selected_tag=selected_tag,
        )
        if not configuration_changes:
            return [], set()
        warning = format_configuration_change_warning(configuration_changes)
        decision = self._show_configuration_change_dialog(warning)
        if decision == "cancel":
            return None
        if decision == "keep":
            return configuration_changes, {change["tag"] for change in configuration_changes}
        return configuration_changes, set()

    def _show_configuration_change_dialog(self, warning_text):
        """Show a scrollable changed-configuration choice capped at 75% of the screen height."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Changed batch configurations")
        dialog.transient(self.root)
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(0, weight=1)

        text_frame = ttk.Frame(dialog)
        text_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=(20, 8))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        text_widget = tk.Text(
            text_frame,
            wrap=tk.WORD,
            width=84,
            height=warning_text.count("\n") + 1,
            relief=tk.FLAT,
            borderwidth=0,
        )
        text_widget.insert("1.0", warning_text)
        text_widget.configure(state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        text_widget.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        ttk.Label(
            dialog,
            text="Choose how to handle previously completed results for these tags.",
            wraplength=560,
            justify=tk.LEFT,
        ).grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 8))

        result = []

        def choose(decision):
            """Record the user's choice and close the dialog."""
            result.append(decision)
            dialog.destroy()

        buttons = ttk.Frame(dialog)
        buttons.grid(row=2, column=0, sticky=tk.SE, padx=20, pady=(0, 16))
        ttk.Button(
            buttons, text="Delete old results", command=lambda: choose("delete"),
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            buttons, text="Keep old results", command=lambda: choose("keep"),
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            buttons, text="Cancel", command=lambda: choose("cancel"),
        ).pack(side=tk.LEFT)
        dialog.protocol("WM_DELETE_WINDOW", lambda: choose("cancel"))
        dialog.bind("<Escape>", lambda event: choose("cancel"))

        dialog.update_idletasks()
        # Cap the natural height at 75% of the screen; smaller content stays smaller.
        max_height = int(dialog.winfo_screenheight() * 0.75)
        if dialog.winfo_reqheight() > max_height:
            dialog.geometry(f"{dialog.winfo_reqwidth()}x{max_height}")
            dialog.update_idletasks()
        dialog_x = self.root.winfo_rootx() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        dialog_y = self.root.winfo_rooty() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{max(0, dialog_x)}+{max(0, dialog_y)}")
        dialog.grab_set()
        self.root.wait_window(dialog)
        return result[0] if result else "cancel"

    def _on_start_batch(self):
        """Validate, save, and launch the current batch inside the GUI."""
        if self.is_batch_running or not self._update_config_from_ui():
            return
        if not self._confirm_duplicate_batch_rows():
            return
        try:
            if not self._confirm_saved_batch_inputs():
                return
            confirmation = self._confirm_batch_configuration_changes()
            if confirmation is None:
                return
            configuration_changes, keep_tags = confirmation
        except (OSError, ValueError) as error:
            messagebox.showerror("Batch error", str(error))
            return
        self.batch_keep_changed_tags = keep_tags
        result_dir = self._result_root()
        if not configuration_changes and result_dir.exists() and any(result_dir.iterdir()):
            completed_count = self._aggregate_result_count()
            if not messagebox.askyesno(
                "Existing results",
                f"Existing result files were found. {completed_count} completed rows will resume. Continue batch execution?",
            ):
                return
        self._launch_batch()

    def _on_run_selected_row(self):
        """Run exactly one saved and selected batch row."""
        selection = self.batch_tree.selection()
        if len(selection) != 1:
            messagebox.showwarning("Batch selection", "Select exactly one batch row.")
            return
        if not self._update_config_from_ui():
            return
        if not self._confirm_duplicate_batch_rows():
            return
        try:
            if not self._confirm_saved_batch_inputs():
                return
        except (OSError, ValueError) as error:
            messagebox.showerror("Batch error", str(error))
            return
        selected_tag = self.batch_rows[int(selection[0])].get("tag", "").strip()
        if not selected_tag:
            messagebox.showerror("Batch error", "Selected row must have a tag.")
            return
        try:
            confirmation = self._confirm_batch_configuration_changes(selected_tag)
            if confirmation is None:
                return
            configuration_changes, keep_tags = confirmation
        except (OSError, ValueError) as error:
            messagebox.showerror("Batch error", str(error))
            return
        self.batch_keep_changed_tags = keep_tags
        result_dir = self._result_root() / selected_tag
        if not configuration_changes and result_dir.exists() and any(result_dir.iterdir()):
            if not self._ask_tag_result_replacement(selected_tag):
                return
            archive_path(result_dir)
        self._launch_batch(selected_tag)

    def _aggregate_result_count(self):
        """Return the number of persisted aggregate batch result rows."""
        report_path = self._result_root() / "result.csv"
        if not report_path.is_file():
            return 0
        try:
            with report_path.open(newline="", encoding="utf-8") as report_file:
                return sum(1 for row in csv.DictReader(report_file) if row.get("tag"))
        except (OSError, csv.Error):
            return 0

    def _clear_result_directory(self):
        """Archive all stored simulation and aggregate results after confirmation."""
        result_dir = self._result_root()
        if not result_dir.exists():
            return
        if not messagebox.askyesno(
            "Clear results",
            "Archive all simulation results and aggregate batch reports?",
        ):
            return
        archive_path(result_dir)
        self.batch_status_var.set("Ready: 0 completed")

    def _launch_batch(self, selected_tag=None):
        """Start the saved batch in a daemon worker thread."""
        self.is_batch_running = True
        self.batch_cancel_requested = False
        self.batch_finalize_requested = False
        self.finalize_requested.clear()
        self.batch_failed = False
        self.batch_selected_tag = selected_tag
        self.batch_status_var.set("Starting batch")
        self._display_performance_stats(0.0, 0, 0)
        self.batch_start_button.config(state=tk.DISABLED)
        self.batch_stop_button.config(state=tk.NORMAL)
        self.progress_stop_btn.config(state=tk.NORMAL)
        self.progress_finalize_btn.config(state=tk.NORMAL)
        self.notebook.select(self.batch_tab)
        self._show_progress_window()
        worker = threading.Thread(target=self._run_batch_thread, daemon=True)
        worker.start()

    def _run_batch_thread(self):
        """Run batch rows and relay worker progress to the main Tk thread."""
        def report_progress(completed, total, tag, status):
            """Schedule one batch status update on the Tk event loop."""
            if not self.is_closing:
                self.root.after(0, self._update_batch_status, completed, total, tag, status)

        def report_graph(output_dir, year):
            """Queue one generated batch graph frame for the GUI viewer."""
            self._pending_graph_update = (str(output_dir), year)

        def report_performance(elapsed, years, total_animals):
            """Schedule one batch performance snapshot on the Tk event loop."""
            if not self.is_closing:
                self.root.after(
                    0,
                    self._display_performance_stats,
                    elapsed,
                    years,
                    total_animals,
                )

        try:
            run_batch(
                self.batch_path,
                self.config_file,
                should_cancel=lambda: self.batch_cancel_requested,
                should_finalize=self._consume_finalize_request,
                progress_callback=report_progress,
                graph_callback=report_graph,
                performance_callback=report_performance,
                selected_tag=self.batch_selected_tag,
                keep_changed_tags=self.batch_keep_changed_tags,
            )
        except Exception as error:
            self.batch_failed = True
            if not self.is_closing:
                self.root.after(0, messagebox.showerror, "Batch error", str(error))
        finally:
            if self.is_closing:
                self.is_batch_running = False
            else:
                self.root.after(0, self._finish_batch)

    def _update_batch_status(self, completed, total, tag, status):
        """Display one batch runner progress event."""
        self.batch_status_var.set(f"{completed}/{total}: {tag} ({status})")
        batch_row = next((row for row in self.batch_rows if row.get("tag", "").strip() == tag), None)
        self._set_progress_calculation(tag, batch_row)

    def _finish_batch(self):
        """Restore Batch tab controls after the worker exits."""
        self.is_batch_running = False
        self.batch_start_button.config(state=tk.NORMAL)
        self.batch_stop_button.config(state=tk.DISABLED)
        self.progress_stop_btn.config(state=tk.DISABLED)
        self.progress_finalize_btn.config(state=tk.DISABLED)
        if self.batch_cancel_requested:
            self.batch_status_var.set("Batch cancelled")
        elif self.batch_failed:
            self.batch_status_var.set("Batch failed")
        elif self.batch_finalize_requested:
            self.batch_status_var.set("Batch finalized")
        else:
            self.batch_status_var.set("Batch complete")

    def _stop_batch(self):
        """Request cooperative cancellation of the active batch worker."""
        if not self.is_batch_running:
            return
        self.batch_cancel_requested = True
        self.batch_status_var.set("Cancellation requested")
        self.batch_stop_button.config(state=tk.DISABLED)
        self.progress_stop_btn.config(state=tk.DISABLED)
        self.progress_finalize_btn.config(state=tk.DISABLED)

    def _load_current_model_batch_defaults(self):
        """Replace unsaved batch rows with active model defaults."""
        model_class = load_model_class(self.config["model"])
        batch_text = model_class.add_batch()
        if not batch_text:
            messagebox.showinfo("Batch defaults", "The selected model has no batch defaults.")
            return
        try:
            self._load_batch_text(batch_text)
        except ValueError as error:
            messagebox.showerror("Batch error", str(error))

    def _load_current_model_defaults(self):
        """Reset active model settings and batch defaults in memory only."""
        for name, metadata in self.model_metadata["settings"].items():
            self.config[name] = metadata["default"]
        self._set_config_dirty(True)
        self._load_config_to_ui()

        model_class = load_model_class(self.config["model"])
        batch_text = model_class.add_batch()
        if batch_text:
            self._load_batch_text(batch_text)
        messagebox.showinfo(
            "Model defaults loaded",
            "Model settings and batch defaults are loaded in memory. Save configuration and batch changes explicitly.",
        )

    def _rebuild_dynamic_graph_tabs(self):
        """Recreate viewer tabs for graphs declared by the active model."""
        for tab_id in self.graph_notebook.tabs()[1:]:
            self.graph_notebook.forget(tab_id)
        self.dynamic_graph_canvases = {}
        self.dynamic_graph_images = {}
        self.dynamic_graph_photos = {}
        for graph in self.model_metadata["graphs"]:
            graph_frame = ttk.Frame(self.graph_notebook, padding=8)
            self.graph_notebook.add(graph_frame, text=graph["title"])
            graph_frame.columnconfigure(0, weight=1)
            graph_frame.rowconfigure(0, weight=1)
            canvas = tk.Canvas(graph_frame, bg="white", highlightthickness=0)
            canvas.grid(row=0, column=0, sticky="nsew")
            canvas.create_text(
                10,
                10,
                anchor=tk.NW,
                text=f"No {graph['title']} graph yet",
                fill="gray",
            )
            canvas.bind(
                "<Configure>",
                lambda event, filename=graph["filename"]: self._rescale_dynamic_graph(filename),
            )
            self.dynamic_graph_canvases[graph["filename"]] = canvas

    def _rescale_dynamic_graph(self, filename):
        """Rescale one declared graph image to fit its canvas."""
        image = self.dynamic_graph_images.get(filename)
        if image is None:
            return
        canvas = self.dynamic_graph_canvases[filename]
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            return
        max_width = max(1, canvas_width - 10)
        max_height = max(1, canvas_height - 10)
        source_width, source_height = image.size
        scale = min(max_width / source_width, max_height / source_height)
        resized = image.resize(
            (max(1, int(source_width * scale)), max(1, int(source_height * scale))),
            Image.Resampling.LANCZOS,
        )
        photo = ImageTk.PhotoImage(resized)
        self.dynamic_graph_photos[filename] = photo
        canvas.delete("all")
        canvas.create_image(
            (canvas_width - resized.width) // 2,
            (canvas_height - resized.height) // 2,
            anchor=tk.NW,
            image=photo,
        )

    def _schedule_graph_rescale(self, graph_name):
        """Schedule one graph resize after geometry changes settle."""
        callbacks = {
            "distribution": self._rescale_distribution_graph,
            "survivorship": self._rescale_survivorship_graph,
            "betaoccurrence": self._rescale_betaoccurrence_graph,
        }
        previous_job = self._rescale_jobs.pop(graph_name, None)
        if previous_job is not None:
            self.root.after_cancel(previous_job)

        def rescale():
            """Run the latest requested resize for one graph."""
            self._rescale_jobs.pop(graph_name, None)
            callbacks[graph_name]()

        self._rescale_jobs[graph_name] = self.root.after(120, rescale)

    def _display_dynamic_graphs(self, result_dir, year, final=False):
        """Load declared annual or final graph files into dynamic tabs."""
        for graph in self.model_metadata["graphs"]:
            if final:
                if not graph["final"]:
                    continue
                image_path = Path(result_dir) / f"{graph['filename']}.png"
            else:
                if not graph["annual"]:
                    continue
                image_path = Path(result_dir) / f"{graph['filename']}_{year:07d}.png"
            if not image_path.exists():
                continue
            try:
                with Image.open(image_path) as graph_image:
                    self.dynamic_graph_images[graph["filename"]] = graph_image.copy()
                self._rescale_dynamic_graph(graph["filename"])
            except Exception as error:
                log(f"Error loading {graph['filename']} graph: {error}")

    def _rescale_distribution_graph(self):
        """Rescale distribution graph to fit its container"""
        if self.distribution_original_img is None:
            return
        
        try:
            canvas_width = self.distribution_canvas.winfo_width()
            canvas_height = self.distribution_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                return

            max_width = max(1, canvas_width - 10)
            max_height = max(1, canvas_height - 10)
            src_w, src_h = self.distribution_original_img.size
            scale = min(max_width / src_w, max_height / src_h)
            new_w = max(1, int(src_w * scale))
            new_h = max(1, int(src_h * scale))

            resized = self.distribution_original_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            self.distribution_photo = ImageTk.PhotoImage(resized)

            self.distribution_canvas.delete("all")
            x = (canvas_width - new_w) // 2
            y = (canvas_height - new_h) // 2
            self.distribution_canvas.create_image(x, y, anchor=tk.NW, image=self.distribution_photo)
        except Exception as e:
            log(f"Error rescaling distribution: {e}")

    def _rescale_survivorship_graph(self):
        """Rescale survivorship graph to fit its container"""
        if self.survivorship_original_img is None:
            return
        
        try:
            canvas_width = self.survivorship_canvas.winfo_width()
            canvas_height = self.survivorship_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                return

            max_width = max(1, canvas_width - 10)
            max_height = max(1, canvas_height - 10)
            src_w, src_h = self.survivorship_original_img.size
            scale = min(max_width / src_w, max_height / src_h)
            new_w = max(1, int(src_w * scale))
            new_h = max(1, int(src_h * scale))

            resized = self.survivorship_original_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            self.survivorship_photo = ImageTk.PhotoImage(resized)

            self.survivorship_canvas.delete("all")
            x = (canvas_width - new_w) // 2
            y = (canvas_height - new_h) // 2
            self.survivorship_canvas.create_image(x, y, anchor=tk.NW, image=self.survivorship_photo)
        except Exception as e:
            log(f"Error rescaling survivorship: {e}")

    def _rescale_betaoccurrence_graph(self):
        """Rescale beta occurrence graph to fit its container"""
        if self.betaoccurrence_original_img is None:
            return
        
        try:
            canvas_width = self.betaoccurrence_canvas.winfo_width()
            canvas_height = self.betaoccurrence_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                return

            max_width = max(1, canvas_width - 10)
            max_height = max(1, canvas_height - 10)
            src_w, src_h = self.betaoccurrence_original_img.size
            scale = min(max_width / src_w, max_height / src_h)
            new_w = max(1, int(src_w * scale))
            new_h = max(1, int(src_h * scale))

            resized = self.betaoccurrence_original_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            self.betaoccurrence_photo = ImageTk.PhotoImage(resized)

            self.betaoccurrence_canvas.delete("all")
            x = (canvas_width - new_w) // 2
            y = (canvas_height - new_h) // 2
            self.betaoccurrence_canvas.create_image(x, y, anchor=tk.NW, image=self.betaoccurrence_photo)
        except Exception as e:
            log(f"Error rescaling betaoccurrence: {e}")

    def _log_to_gui(self, message):
        """Queue a log message for the Tk event loop."""
        self._gui_log_messages.put(message)

    def _schedule_gui_refresh(self):
        """Schedule the next coalesced GUI refresh."""
        if not self.is_closing:
            self.root.after(200, self._refresh_gui)

    def _refresh_gui(self):
        """Apply pending GUI state and preserve the periodic refresh after errors."""
        try:
            self._apply_gui_refresh()
        except Exception as error:
            log(f"Error refreshing GUI: {error}")
        finally:
            self._schedule_gui_refresh()

    def _apply_gui_refresh(self):
        """Apply queued logs and the latest simulation state in Tk's thread."""
        latest_graph_update = self._pending_graph_update
        self._pending_graph_update = None
        if latest_graph_update is not None:
            result_dir, year = latest_graph_update
            self._display_year_graphs(result_dir, year)

        final_graph_update = self._pending_final_graph_update
        self._pending_final_graph_update = None
        if final_graph_update is not None:
            result_dir, year = final_graph_update
            self._display_dynamic_graphs(result_dir, year, True)

        summary_path = self._pending_summary_path
        self._pending_summary_path = None
        if summary_path is not None:
            self._show_summary_graph_popup(summary_path)

        if hasattr(self, "log_text"):
            has_messages = False
            while True:
                try:
                    message = self._gui_log_messages.get_nowait()
                except queue.Empty:
                    break
                self.log_text.insert(tk.END, message + "\n")
                has_messages = True
            if has_messages and self.log_autoscroll_var.get():
                self.log_text.see(tk.END)

        if self.is_running:
            self._update_performance_stats()

        if self._pending_simulation_error is not None:
            error = self._pending_simulation_error
            self._pending_simulation_error = None
            messagebox.showerror("Simulation Error", error)

        if self._simulation_finished:
            self._simulation_finished = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.progress_stop_btn.config(state=tk.DISABLED)
            self.progress_finalize_btn.config(state=tk.DISABLED)

    def _display_year_graphs(self, result_dir, year):
        """Load and display per-year distribution, survivorship, and beta occurrence images

        Args:
            result_dir: path to results directory
            year: simulation year used in file names
        """
        distribution_file = Path(result_dir) / f"distribution{year}.png"
        survivorship_file = Path(result_dir) / f"survivorship{year}.png"
        betaoccurrence_file = Path(result_dir) / f"betaoccurrence{year}.png"
        self._display_dynamic_graphs(result_dir, year)

        if distribution_file.exists():
            try:
                with Image.open(distribution_file) as dist_img:
                    self.distribution_original_img = dist_img.copy()
                self._rescale_distribution_graph()
            except Exception as e:
                log(f"Error loading distribution graph: {e}")

        if survivorship_file.exists():
            try:
                with Image.open(survivorship_file) as surv_img:
                    self.survivorship_original_img = surv_img.copy()
                self._rescale_survivorship_graph()
            except Exception as e:
                log(f"Error loading survivorship graph: {e}")

        if betaoccurrence_file.exists():
            try:
                with Image.open(betaoccurrence_file) as beta_img:
                    self.betaoccurrence_original_img = beta_img.copy()
                self._rescale_betaoccurrence_graph()
            except Exception as e:
                log(f"Error loading betaoccurrence graph: {e}")

    def _has_year_graphs(self, result_dir, year):
        """Return whether a year produced any legacy or declared graph file."""
        directory = Path(result_dir)
        legacy_files = (
            f"distribution{year}.png",
            f"survivorship{year}.png",
            f"betaoccurrence{year}.png",
        )
        if any((directory / filename).exists() for filename in legacy_files):
            return True
        return any(
            graph["annual"]
            and (directory / f"{graph['filename']}_{year:07d}.png").exists()
            for graph in self.model_metadata["graphs"]
        )

    def _load_config_to_ui(self):
        """Load config values into UI fields"""
        self._loading_ui = True
        try:
            self.model_var.set(self.config["model"])
            requested_device = self.config.get("device", "cuda")
            # Force CPU selection in UI when CUDA backend is unavailable.
            if requested_device == "cuda" and not torch.cuda.is_available():
                requested_device = "cpu"
            self.device_var.set(requested_device)
            self.tag_var.set(self.config.get("tag", DEFAULT_SETTINGS["tag"]))
            for param, var in self.setting_vars.items():
                var.set(str(self.config.get(param, DEFAULT_SETTINGS.get(param, ""))))
            for name, row in self.model_setting_rows.items():
                row["value"].set(str(self.config.get(name, "")))
        finally:
            self._loading_ui = False

    def _format_setting_bounds(self, metadata):
        """Return a human-readable inclusive range for one setting."""
        minimum = metadata.get("min")
        maximum = metadata.get("max")
        if minimum is None and maximum is None:
            return ""
        if minimum is None:
            return f"x <= {maximum}"
        if maximum is None:
            return f"{minimum} <= x"
        return f"{minimum} <= x <= {maximum}"

    def _rebuild_model_settings_grid(self):
        """Rebuild editable setting rows from active model metadata."""
        for widget in self.model_settings_frame.winfo_children():
            widget.destroy()
        self.model_setting_rows = {}
        active_settings = self.model_metadata["settings"]
        excluded_names = set(CORE_SETTING_NAMES) | {"device", "model", "tag"}
        unsupported_names = sorted(
            set(self.config) - set(active_settings) - excluded_names,
        )
        headings = ("", "Name", "Value", "Bounds", "Description")
        for column, heading in enumerate(headings):
            ttk.Label(
                self.model_settings_frame,
                text=heading,
                font=("TkDefaultFont", 9, "bold"),
            ).grid(row=0, column=column, padx=4, pady=(0, 4), sticky=tk.W)
        setting_rows = [
            (name, metadata, True)
            for name, metadata in active_settings.items()
        ] + [
            (name, None, False)
            for name in unsupported_names
        ]
        for row_index, (name, metadata, supported) in enumerate(setting_rows, start=1):
            value = tk.StringVar(
                value=str(self.config.get(name, metadata["default"] if metadata else "")),
            )
            value.trace_add("write", self._mark_config_dirty)
            bounds = tk.StringVar(value=self._format_setting_bounds(metadata) if metadata else "")
            description = tk.StringVar(
                value=metadata["description"] if metadata else "Not supported by the active model",
            )
            self.model_setting_rows[name] = {
                "value": value,
                "bounds": bounds,
                "description": description,
                "supported": supported,
            }
            support_label = ttk.Label(
                self.model_settings_frame,
                text="\u2713" if supported else "\u2717",
                foreground="green" if supported else "red",
                width=2,
            )
            support_label.grid(
                row=row_index, column=0, padx=4, sticky=tk.W,
            )
            name_label = ttk.Label(self.model_settings_frame, text=name, width=28)
            name_label.grid(
                row=row_index, column=1, padx=4, sticky=tk.W,
            )
            value_entry = ttk.Entry(
                self.model_settings_frame,
                textvariable=value,
                width=18,
                state=tk.NORMAL if supported else tk.DISABLED,
            )
            value_entry.grid(
                row=row_index, column=2, padx=4, sticky=tk.W,
            )
            hint = description.get()
            bounds_label = ttk.Label(
                self.model_settings_frame,
                textvariable=bounds,
                width=25,
            )
            bounds_label.grid(
                row=row_index, column=3, padx=4, sticky=tk.W,
            )
            description_label = ttk.Label(
                self.model_settings_frame,
                textvariable=description,
            )
            description_label.grid(
                row=row_index, column=4, padx=4, sticky=tk.W,
            )
            widgets = (
                support_label,
                name_label,
                value_entry,
                bounds_label,
                description_label,
            )
            self.model_setting_rows[name]["widgets"] = widgets
            for widget in widgets:
                self.tooltips.register(widget, hint)

    def _on_save_config(self):
        """Save button handler"""
        if not self._update_config_from_ui():
            return
        self._save_config()
        messagebox.showinfo("Success", "Configuration saved")

    def _on_save_config_as(self):
        """Save As button handler"""
        if not self._update_config_from_ui():
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile=Path(self.config_file).name
        )
        if not file_path:
            return

        self._save_config(file_path)
        messagebox.showinfo("Success", f"Configuration saved to {file_path}")

    def _on_load_config(self):
        """Load button handler"""
        file_path = filedialog.askopenfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            try:
                loaded_config = self._load_config(file_path)
                config, metadata, corrected_names = _prepare_model_configuration(
                    loaded_config,
                    loaded_config["model"],
                )
            except (ModelLoadError, OSError, ValueError) as error:
                messagebox.showerror("Configuration error", f"Could not load configuration: {error}")
                return
            self.config = config
            self.model_metadata = metadata
            self._set_config_dirty(True)
            self._rebuild_model_settings_grid()
            self._rebuild_dynamic_graph_tabs()
            self.batch_column_selector.configure(values=self._batch_column_options(metadata))
            self._load_config_to_ui()
            if corrected_names:
                messagebox.showwarning(
                    "Model settings corrected",
                    "Corrected settings: " + ", ".join(corrected_names),
                )
            messagebox.showinfo("Success", f"Configuration loaded from {file_path}")

    def _on_model_selected(self, event=None):
        """Load selected model metadata into the unsaved GUI configuration."""
        if not self._update_config_from_ui(validate_ranges=False):
            return
        previous_model = self.config["model"]
        model_name = self.model_var.get().strip()
        try:
            config, metadata, corrected_names = _prepare_model_configuration(
                self.config,
                model_name,
            )
        except (ModelLoadError, ValueError) as error:
            self.model_var.set(self.config["model"])
            messagebox.showerror("Model error", f"Could not load model: {error}")
            return
        self.config = config
        self.model_metadata = metadata
        self._set_config_dirty(True)
        self._rebuild_model_settings_grid()
        self._rebuild_dynamic_graph_tabs()
        self.batch_column_selector.configure(values=self._batch_column_options(metadata))
        self._load_config_to_ui()
        if corrected_names:
            messagebox.showwarning(
                "Model settings corrected",
                "Corrected settings: " + ", ".join(corrected_names),
            )
        if model_name != previous_model:
            self._offer_model_batch_defaults(model_name)

    def _offer_model_batch_defaults(self, model_name):
        """Offer unsaved batch defaults when a newly selected model provides them."""
        model_class = load_model_class(model_name)
        batch_text = model_class.add_batch()
        if not batch_text:
            return
        if messagebox.askyesno(
            "Replace batch defaults",
            "Replace the in-memory batch table with defaults from the selected model?",
        ):
            self._load_batch_text(batch_text)

    def _parse_setting_value(self, name, value_text, metadata):
        """Parse one setting string according to declared metadata."""
        setting_type = metadata["type"]
        if setting_type == "int":
            return int(value_text)
        if setting_type == "float":
            return float(value_text)
        if setting_type == "str":
            return value_text
        if value_text.lower() in {"true", "1", "yes"}:
            return True
        if value_text.lower() in {"false", "0", "no"}:
            return False
        raise ValueError(f"{name} must be a boolean")

    def _update_config_from_ui(self, validate_ranges=True):
        """Read fixed and model-specific settings from UI controls."""
        previous_config = dict(self.config)
        self.config["device"] = self.device_var.get()
        self.config["tag"] = self.tag_var.get()
        for param, var in self.setting_vars.items():
            try:
                val = int(var.get())
                if validate_ranges and param in PARAMETER_RANGES:
                    min_val, max_val = PARAMETER_RANGES[param]
                    if not (min_val <= val <= max_val):
                        messagebox.showwarning("Validation", f"{param} must be in [{min_val}, {max_val}]")
                        var.set(str(self.config.get(param, DEFAULT_SETTINGS[param])))
                        return False
                
                self.config[param] = val
            except ValueError:
                messagebox.showerror("Error", f"Invalid value for {param}: {var.get()}")
                return False

        for name, row in self.model_setting_rows.items():
            if not row["supported"]:
                continue
            metadata = self.model_metadata["settings"][name]
            try:
                value = self._parse_setting_value(name, row["value"].get(), metadata)
            except ValueError as error:
                messagebox.showerror("Error", str(error))
                return False
            if validate_ranges:
                value, corrected = _normalize_model_setting(value, metadata)
                if corrected:
                    messagebox.showwarning(
                        "Model settings corrected",
                        f"Corrected setting: {name}",
                    )
                    row["value"].set(str(value))
            self.config[name] = value

        if self.config != previous_config:
            self._set_config_dirty(True)
        return True

    def _restore_canonical_config(self):
        """Restore the canonical configuration file into GUI memory and controls."""
        try:
            loaded_config = self._load_config()
            config, metadata, corrected_names = _prepare_model_configuration(
                loaded_config,
                loaded_config["model"],
            )
        except (ModelLoadError, OSError, ValueError) as error:
            messagebox.showerror(
                "Configuration error",
                f"Could not restore configuration: {error}",
            )
            return False
        self.config = config
        self.model_metadata = metadata
        self._set_config_dirty(False)
        self._rebuild_model_settings_grid()
        self._rebuild_dynamic_graph_tabs()
        self.batch_column_selector.configure(values=self._batch_column_options(metadata))
        self._load_config_to_ui()
        if corrected_names:
            messagebox.showwarning(
                "Model settings corrected",
                "Corrected settings: " + ", ".join(corrected_names),
            )
        return True

    def _ask_tag_result_replacement(self, tag):
        """Return whether to replace the existing result directory for one tag."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Existing tag results")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        choice = {"replace": False}
        ttk.Label(
            dialog,
            text=(
                f"The result directory for tag '{tag}' is not empty.\n"
                "Delete previous calculations and start a new simulation?"
            ),
            justify=tk.LEFT,
            padding=12,
        ).pack(fill=tk.BOTH, expand=True)
        buttons = ttk.Frame(dialog, padding=(12, 0, 12, 12))
        buttons.pack(fill=tk.X)

        def confirm():
            """Approve result replacement and close the dialog."""
            choice["replace"] = True
            dialog.destroy()

        ttk.Button(buttons, text="Yes", command=confirm).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT)
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        dialog.update_idletasks()
        dialog_x = self.root.winfo_rootx() + (
            self.root.winfo_width() - dialog.winfo_reqwidth()
        ) // 2
        dialog_y = self.root.winfo_rooty() + (
            self.root.winfo_height() - dialog.winfo_reqheight()
        ) // 2
        dialog.geometry(f"+{max(0, dialog_x)}+{max(0, dialog_y)}")
        dialog.grab_set()
        self.root.wait_window(dialog)
        return choice["replace"]

    def _confirm_tag_result_replacement(self):
        """Confirm deletion of existing results for the configured single-run tag."""
        tag = self.config["tag"]
        result_dir = self._result_root() / tag
        if not result_dir.exists() or not any(result_dir.iterdir()):
            return True
        should_replace = self._ask_tag_result_replacement(tag)
        if should_replace:
            archive_path(result_dir)
        return should_replace

    def _start_simulation(self):
        """Launch after saving, discarding, or retaining the current GUI configuration."""
        if not self._update_config_from_ui():
            return
        if self.is_config_dirty:
            choice = messagebox.askyesnocancel(
                "Unsaved configuration",
                "Yes: save and start.\n"
                "No: discard changes, restore the saved configuration, and start.\n"
                "Cancel: do not start.",
            )
            if choice is None:
                return
            if choice:
                self._save_config()
            elif not self._restore_canonical_config():
                return
        if not self._confirm_tag_result_replacement():
            return
        self._launch_simulation()

    def _launch_simulation(self):
        """Launch the configured simulation in a background thread."""
        self.is_running = True
        self.finalize_requested.clear()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.progress_stop_btn.config(state=tk.NORMAL)
        self.progress_finalize_btn.config(state=tk.NORMAL)
        self._set_progress_calculation(self.config["tag"])
        self._show_progress_window()
        
        # Clear previous log
        self.log_text.delete(1.0, tk.END)
        self.distribution_canvas.delete("all")
        self.distribution_canvas.create_text(10, 10, anchor=tk.NW, text="No distribution graph yet", fill="gray")
        self.survivorship_canvas.delete("all")
        self.survivorship_canvas.create_text(10, 10, anchor=tk.NW, text="No survivorship graph yet", fill="gray")
        self.betaoccurrence_canvas.delete("all")
        self.betaoccurrence_canvas.create_text(10, 10, anchor=tk.NW, text="No beta graph yet", fill="gray")
        self.distribution_original_img = None
        self.survivorship_original_img = None
        self.betaoccurrence_original_img = None
        self.distribution_photo = None
        self.survivorship_photo = None
        self.betaoccurrence_photo = None
        
        # Run in background thread to keep GUI responsive
        sim_thread = threading.Thread(target=self._run_simulation_thread, daemon=True)
        sim_thread.start()

    def _run_simulation_thread(self):
        """Background thread for simulation execution"""
        try:
            log(f"Starting simulation with config: {self.config}")
            
            self.simulation = PopulationSimulation(
                self.config,
                result_root=self._result_root(),
            )
            # Initialize start time for performance tracking
            self.simulation.start_time = time.perf_counter()
            completed_naturally = False
            
            # Run iterative steps (allows stop button to work)
            while self.is_running:
                has_next = self.simulation.step(
                    should_finalize=self._consume_finalize_request,
                )
                if self.simulation.year > 0:
                    latest_year = self.simulation.year - 1
                    output_dir = str(self.simulation.output_dir)
                    if self._has_year_graphs(output_dir, latest_year):
                        self._pending_graph_update = (output_dir, latest_year)
                if not has_next:
                    completed_naturally = True
                    break
            
            # Generate summary graph and animations on completion or manual stop
            if self.simulation:
                output_dir = self.simulation.output_dir
                last_year = self.simulation.year - 1
                
                # Ensure graphs exist for the last year (in case of manual stop)
                if last_year >= 0:
                    log(f"Generating graphs for final year {last_year} before export")
                    self.simulation._generate_year_graphs(last_year)
                
                # Export all results (CSV + summary graph + GIFs)
                self.simulation.export_results(successful=completed_naturally)
                if completed_naturally:
                    self._pending_final_graph_update = (str(output_dir), last_year)
                log("Results exported (normal completion)" if completed_naturally else "Results exported (manual stop)")
                
                # Show summary graph popup (call from main thread)
                summary_path = output_dir / "results_summary.png"
                if summary_path.exists():
                    self._pending_summary_path = str(summary_path)
            
        except Exception as e:
            log(f"Error during simulation: {e}")
            self._pending_simulation_error = str(e)
        finally:
            self.is_running = False
            self._simulation_finished = True

    def _update_performance_stats(self):
        """Update performance statistics display from current simulation state"""
        if not self.simulation or not self.simulation.start_time:
            return

        elapsed_sec = time.perf_counter() - self.simulation.start_time
        self._display_performance_stats(
            elapsed_sec,
            self.simulation.year,
            self.simulation.total_animals_processed,
        )

    def _display_performance_stats(self, elapsed_sec, year, total_animals):
        """Display one simulation or batch performance snapshot."""
        # Elapsed time
        self.stat_elapsed_time.config(text=f"{elapsed_sec:.3f} s")

        # Average iteration time
        if year > 0:
            avg_iteration = elapsed_sec / year
            self.stat_avg_iteration.config(text=f"{avg_iteration:.6f} s")

            # Average per-element time in microseconds
            if total_animals > 0:
                avg_per_element_sec = elapsed_sec / total_animals
                avg_per_element_us = avg_per_element_sec * 1_000_000
                self.stat_avg_element.config(text=f"{avg_per_element_us:.3f} μs")
            else:
                self.stat_avg_element.config(text="0.000 μs")
        else:
            self.stat_avg_iteration.config(text="0.000000 s")
            self.stat_avg_element.config(text="0.000 μs")
    
    def _show_summary_graph_popup(self, image_path):
        """Display summary graph in a popup window
        
        Args:
            image_path: path to results_summary.png
        """
        try:
            # Create new window
            popup = tk.Toplevel(self.root)
            popup.title("Simulation Summary - Results")
            popup.geometry("1000x800")
            
            # Create frame for the image and button
            main_frame = ttk.Frame(popup)
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Create canvas for image
            canvas = tk.Canvas(main_frame, bg="white")
            canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Load and display image
            with Image.open(image_path) as img:
                self.popup_original_img = img.copy()
            
            # Bind canvas resize to rescale image (supports both downscale and upscale)
            def on_popup_resize(event):
                if self.popup_original_img is None:
                    return

                max_width = max(1, event.width - 10)
                max_height = max(1, event.height - 10)

                src_w, src_h = self.popup_original_img.size
                scale = min(max_width / src_w, max_height / src_h)
                new_w = max(1, int(src_w * scale))
                new_h = max(1, int(src_h * scale))

                resized_img = self.popup_original_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                self._popup_photo = ImageTk.PhotoImage(resized_img)

                canvas.delete("all")
                x = (event.width - new_w) // 2
                y = (event.height - new_h) // 2
                canvas.create_image(x, y, anchor=tk.NW, image=self._popup_photo)

            canvas.bind("<Configure>", on_popup_resize)
            
            # Add close button
            button_frame = ttk.Frame(popup)
            button_frame.pack(fill=tk.X, padx=5, pady=5)
            close_btn = ttk.Button(button_frame, text="Close", command=popup.destroy)
            close_btn.pack(pady=5)
            
            log(f"Summary graph displayed in popup: {image_path}")
        except Exception as e:
            log(f"Error displaying summary graph: {e}")
            messagebox.showerror("Display Error", f"Failed to display summary: {e}")

    def _stop_simulation(self):
        """Stop running simulation"""
        self.is_running = False
        self.stop_btn.config(state=tk.DISABLED)
        self.progress_stop_btn.config(state=tk.DISABLED)
        self.progress_finalize_btn.config(state=tk.DISABLED)
        log("Simulation stopped by user")

    def _consume_finalize_request(self):
        """Consume one thread-safe successful-finalization request."""
        if not self.finalize_requested.is_set():
            return False
        self.finalize_requested.clear()
        return True

    def _finalize_active_calculation(self):
        """Request successful completion of the active calculation and batch."""
        if not self.is_running and not self.is_batch_running:
            return
        self.finalize_requested.set()
        self.batch_finalize_requested = self.is_batch_running
        self.progress_finalize_btn.config(state=tk.DISABLED)
        if self.is_batch_running:
            self.batch_status_var.set("Finalization requested")
        log("Simulation finalization requested by user")

    def _stop_active_calculation(self):
        """Stop the active single simulation or request cancellation of the active batch."""
        if self.is_batch_running:
            self._stop_batch()
            return
        if self.is_running:
            self._stop_simulation()

    def _on_close(self):
        """Resolve unsaved state and stop workers before destroying the GUI."""
        if self.is_running or self.is_batch_running:
            if not messagebox.askyesno(
                "Calculation running",
                "Stop the active calculation and close the application?",
            ):
                return
        if not self._confirm_experiment_transition():
            return
        self.is_closing = True
        self.is_running = False
        self.batch_cancel_requested = True
        set_logger(None)
        self._wait_for_workers_before_close()

    def _wait_for_workers_before_close(self):
        """Destroy the root after active workers acknowledge cancellation."""
        if self.is_batch_running:
            self.root.after(100, self._wait_for_workers_before_close)
            return
        self.root.destroy()


def main():
    """Launch GUI"""
    root = tk.Tk()
    manager = ExperimentManager(Path.cwd())
    missing_selector = manager.get_active_experiment_name() is None
    existing_experiments = manager.list_experiments()
    gui = SimulationGUI(root)
    if missing_selector and existing_experiments:
        messagebox.showwarning(
            "Experiment selected",
            f"default.conf was missing. Selected experiment: {existing_experiments[0]}",
            parent=root,
        )
    elif not existing_experiments:
        try:
            experiment_dir = gui._prompt_new_experiment()
        except (ModelLoadError, OSError, ValueError) as error:
            messagebox.showerror("Experiment error", str(error), parent=root)
            root.destroy()
            return
        if experiment_dir is None:
            root.destroy()
            return
        gui.experiment_selector.configure(values=manager.list_experiments())
        gui.experiment_var.set(experiment_dir.name)
        gui._activate_experiment(experiment_dir.name)
    root.mainloop()


if __name__ == "__main__":
    main()
