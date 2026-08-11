"""
Tkinter GUI for chimp evolution simulation
Allows parameter configuration, execution control, and result visualization
"""

import json
import csv
import shutil
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import torch
from PIL import Image, ImageTk

from main import PopulationSimulation, set_logger, log
from batch import run_batch
from model_loader import ModelLoadError, discover_models, load_model_class
from model_metadata import validate_model_metadata
from settings import DEFAULT_SETTINGS, PARAMETER_RANGES


CORE_SETTING_NAMES = (
    "stat_generation_period",
    "graph_generation_period",
    "max_iterations",
)


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
        
        self.config_file = "config.json"
        self.config = self._load_config()
        self.available_models = discover_models()
        self.config, self.model_metadata, _ = _prepare_model_configuration(
            self.config,
            self.config["model"],
        )
        self.is_config_dirty = False
        self.simulation = None
        self.is_running = False
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
        
        # Set logging callback
        set_logger(self._log_to_gui)
        
        # Build UI
        self._create_widgets()
        self._load_config_to_ui()
        
        log("GUI initialized")

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
        with target_file.open("w") as config_file:
            json.dump(self.config, config_file, indent=2)
        if target_file == Path(self.config_file):
            self.is_config_dirty = False
        log(f"Config saved to {target_file}")

    def _create_widgets(self):
        """Build GUI layout with tabs and controls"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Settings
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="Settings")
        self._create_settings_tab(settings_frame)
        
        # Tab 2: Progress & Results
        progress_frame = ttk.Frame(self.notebook)
        self.notebook.add(progress_frame, text="Progress")
        self._create_progress_tab(progress_frame)

        batch_frame = ttk.Frame(self.notebook)
        self.notebook.add(batch_frame, text="Batch")
        self.batch_tab = batch_frame
        self._create_batch_tab(batch_frame)
        
        # Control buttons at bottom
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.start_btn = ttk.Button(control_frame, text="Start Simulation", command=self._start_simulation)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self._stop_simulation, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Save Config", command=self._on_save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Save Config As...", command=self._on_save_config_as).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Load Config", command=self._on_load_config).pack(side=tk.LEFT, padx=5)

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
        self.model_selector.bind("<<ComboboxSelected>>", self._on_model_selected)
        ttk.Button(model_frame, text="Load Model", command=self._on_model_selected).pack(side=tk.LEFT, padx=5)
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
        
        ttk.Radiobutton(device_frame, text="CUDA (GPU)" if available_cuda else "CUDA (not available)", 
                       variable=self.device_var, value="cuda", 
                       state=tk.NORMAL if available_cuda else tk.DISABLED).pack(anchor=tk.W)
        ttk.Radiobutton(device_frame, text="CPU", variable=self.device_var, value="cpu").pack(anchor=tk.W)
        
        # Performance note
        note_text = ("Note: GPU acceleration is effective only for populations of ~1 million or more. "
                     "For populations under 500,000 animals, CPU is faster due to data transfer overhead. "
                     "If using GPU, increase stat_generation_period and graph_generation_period to reduce transfers.")
        note_label = ttk.Label(device_frame, text=note_text, foreground="gray", 
                              font=("TkDefaultFont", 8), wraplength=400, justify=tk.LEFT)
        note_label.pack(anchor=tk.W, pady=(5, 0))
        
        self.setting_vars = {}

        core_frame = ttk.LabelFrame(parent, text="Core Settings", padding=10)
        core_frame.pack(fill=tk.X, padx=5, pady=5)
        for row, name in enumerate(CORE_SETTING_NAMES):
            ttk.Label(core_frame, text=name, width=28).grid(row=row, column=0, sticky=tk.W)
            value_var = tk.StringVar(value=str(self.config[name]))
            self.setting_vars[name] = value_var
            ttk.Entry(core_frame, textvariable=value_var, width=15).grid(row=row, column=1, padx=5)
            minimum, maximum = PARAMETER_RANGES[name]
            ttk.Label(
                core_frame,
                text=f"{minimum} <= x <= {maximum}",
                foreground="gray",
            ).grid(row=row, column=2, sticky=tk.W)

        self.model_settings_frame = ttk.LabelFrame(parent, text="Model Settings", padding=10)
        self.model_settings_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.model_setting_rows = {}
        self._rebuild_model_settings_grid()
        
        # Tag field
        tag_frame = ttk.Frame(parent)
        tag_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(tag_frame, text="Run tag", width=25).pack(side=tk.LEFT)
        self.tag_var = tk.StringVar(value=self.config.get("tag", "default"))
        ttk.Entry(tag_frame, textvariable=self.tag_var, width=15).pack(side=tk.LEFT, padx=5)

    def _create_progress_tab(self, parent):
        """Create progress and statistics display"""
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
        self.distribution_canvas.bind("<Configure>", lambda e: self._rescale_distribution_graph())

        # Middle graph: Survivorship
        middle_graph = ttk.Frame(graphs_row)
        middle_graph.grid(row=0, column=1, sticky="nsew", padx=(3, 3))
        middle_graph.columnconfigure(0, weight=1)
        middle_graph.rowconfigure(1, weight=1)
        ttk.Label(middle_graph, text="Survivorship Curve").grid(row=0, column=0, sticky="w")
        self.survivorship_canvas = tk.Canvas(middle_graph, bg="white", highlightthickness=0)
        self.survivorship_canvas.grid(row=1, column=0, sticky="nsew")
        self.survivorship_canvas.create_text(10, 10, anchor=tk.NW, text="No survivorship graph yet", fill="gray")
        self.survivorship_canvas.bind("<Configure>", lambda e: self._rescale_survivorship_graph())

        # Right graph: Beta Occurrence
        right_graph = ttk.Frame(graphs_row)
        right_graph.grid(row=0, column=2, sticky="nsew", padx=(3, 0))
        right_graph.columnconfigure(0, weight=1)
        right_graph.rowconfigure(1, weight=1)
        ttk.Label(right_graph, text="Beta Distribution").grid(row=0, column=0, sticky="w")
        self.betaoccurrence_canvas = tk.Canvas(right_graph, bg="white", highlightthickness=0)
        self.betaoccurrence_canvas.grid(row=1, column=0, sticky="nsew")
        self.betaoccurrence_canvas.create_text(10, 10, anchor=tk.NW, text="No beta graph yet", fill="gray")
        self.betaoccurrence_canvas.bind("<Configure>", lambda e: self._rescale_betaoccurrence_graph())
        
        # Performance statistics panel
        stats_frame = ttk.LabelFrame(parent, text="Performance Statistics", padding=10)
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill=tk.X)
        
        # Row 1: Elapsed time
        ttk.Label(stats_grid, text="Elapsed Time:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.stat_elapsed_time = ttk.Label(stats_grid, text="0.000 s", font=("TkDefaultFont", 9, "bold"))
        self.stat_elapsed_time.grid(row=0, column=1, sticky=tk.W)
        
        # Row 2: Average iteration time
        ttk.Label(stats_grid, text="Avg Iteration Time:").grid(row=0, column=2, sticky=tk.W, padx=(20, 10))
        self.stat_avg_iteration = ttk.Label(stats_grid, text="0.000000 s", font=("TkDefaultFont", 9, "bold"))
        self.stat_avg_iteration.grid(row=0, column=3, sticky=tk.W)
        
        # Row 3: Average per-element time
        ttk.Label(stats_grid, text="Avg Per-Element Time:").grid(row=0, column=4, sticky=tk.W, padx=(20, 10))
        self.stat_avg_element = ttk.Label(stats_grid, text="0.000 μs", font=("TkDefaultFont", 9, "bold"))
        self.stat_avg_element.grid(row=0, column=5, sticky=tk.W)
        
        # Log output
        log_frame = ttk.LabelFrame(parent, text="Log Output", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = tk.Text(log_frame, height=8, width=80, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _create_batch_tab(self, parent):
        """Create the editable in-memory batch CSV table."""
        self.batch_path = Path("multi.csv")
        self.batch_columns = []
        self.batch_rows = []
        self.is_batch_dirty = False
        self.is_batch_running = False
        self.batch_cancel_requested = False
        self.batch_status_var = tk.StringVar(value="Ready")
        controls = ttk.Frame(parent)
        controls.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(controls, text="Add Row", command=self._add_batch_row).pack(side=tk.LEFT)
        ttk.Button(controls, text="Save Batch", command=self._on_save_batch).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            controls,
            text="Load Model Defaults",
            command=self._load_current_model_batch_defaults,
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            controls,
            text="Clear Results",
            command=self._clear_result_directory,
        ).pack(side=tk.LEFT, padx=5)
        self.batch_column_var = tk.StringVar()
        self.batch_column_selector = ttk.Combobox(
            controls,
            textvariable=self.batch_column_var,
            values=list(self.model_metadata["settings"]),
            state="readonly",
            width=24,
        )
        self.batch_column_selector.pack(side=tk.LEFT, padx=(15, 0))
        ttk.Button(controls, text="Add Column", command=self._add_batch_column).pack(side=tk.LEFT, padx=5)
        self.batch_start_button = ttk.Button(
            controls,
            text="Start Batch",
            command=self._on_start_batch,
        )
        self.batch_start_button.pack(side=tk.RIGHT)
        self.batch_stop_button = ttk.Button(
            controls,
            text="Stop Batch",
            command=self._stop_batch,
            state=tk.DISABLED,
        )
        self.batch_stop_button.pack(side=tk.RIGHT, padx=5)
        self.batch_status_var.set(f"Ready: {self._aggregate_result_count()} completed")
        ttk.Label(parent, textvariable=self.batch_status_var).pack(anchor=tk.W, padx=8)
        self.batch_tree = ttk.Treeview(parent, show="headings")
        self.batch_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.batch_tree.bind("<Double-1>", self._edit_batch_cell)
        self._load_batch_csv(self.batch_path)

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
        self.is_batch_dirty = False
        self._render_batch_grid()

    def _load_batch_text(self, batch_text):
        """Load one model-provided batch CSV string into memory."""
        rows = list(csv.DictReader(batch_text.strip().splitlines())) if batch_text.strip() else []
        columns = list(rows[0]) if rows else (batch_text.strip().splitlines()[0].split(",") if batch_text.strip() else ["tag"])
        if not columns or columns[0] != "tag":
            raise ValueError("Model batch CSV must have tag as its first column")
        self.batch_columns = columns
        self.batch_rows = [{name: row.get(name, "") for name in columns} for row in rows]
        self.is_batch_dirty = True
        self._render_batch_grid()

    def _save_batch_csv(self, file_path=None):
        """Write the current valid batch table to CSV."""
        if not self.batch_columns or self.batch_columns[0] != "tag":
            raise ValueError("Batch CSV must have tag as its first column")
        tags = [row.get("tag", "").strip() for row in self.batch_rows]
        if any(not tag for tag in tags) or len(tags) != len(set(tags)):
            raise ValueError("Batch tags must be non-empty and unique")
        path = Path(file_path or self.batch_path)
        with path.open("w", newline="", encoding="utf-8") as batch_file:
            writer = csv.DictWriter(batch_file, fieldnames=self.batch_columns)
            writer.writeheader()
            writer.writerows(self.batch_rows)
        self.batch_path = path
        self.is_batch_dirty = False

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
        self.is_batch_dirty = True
        self._render_batch_grid()

    def _add_batch_column(self):
        """Add the selected model setting as an editable batch column."""
        name = self.batch_column_var.get()
        if not name or name in self.batch_columns:
            return
        self.batch_columns.append(name)
        for row in self.batch_rows:
            row[name] = ""
        self.is_batch_dirty = True
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
            self.is_batch_dirty = True
            editor.destroy()
            self._render_batch_grid()

        editor.bind("<Return>", commit_edit)
        editor.bind("<FocusOut>", commit_edit)

    def _on_save_batch(self):
        """Save the editable batch table and report validation errors."""
        try:
            self._save_batch_csv()
        except (OSError, ValueError) as error:
            messagebox.showerror("Batch error", str(error))
            return
        messagebox.showinfo("Success", f"Batch saved to {self.batch_path}")

    def _on_start_batch(self):
        """Validate, save, and launch the current batch inside the GUI."""
        if self.is_batch_running or not self._update_config_from_ui():
            return
        if self.is_config_dirty:
            if not messagebox.askyesno(
                "Unsaved configuration",
                "Save configuration before starting the batch?",
            ):
                return
            self._save_config()
        if self.is_batch_dirty:
            if not messagebox.askyesno(
                "Unsaved batch",
                "Save batch changes before starting the batch?",
            ):
                return
            try:
                self._save_batch_csv()
            except (OSError, ValueError) as error:
                messagebox.showerror("Batch error", str(error))
                return
        result_dir = Path("result")
        if result_dir.exists() and any(result_dir.iterdir()):
            completed_count = self._aggregate_result_count()
            if not messagebox.askyesno(
                "Existing results",
                f"Existing result files were found. {completed_count} completed rows will resume. Continue batch execution?",
            ):
                return
        self._launch_batch()

    def _aggregate_result_count(self):
        """Return the number of persisted aggregate batch result rows."""
        report_path = Path("result") / "result.csv"
        if not report_path.is_file():
            return 0
        try:
            with report_path.open(newline="", encoding="utf-8") as report_file:
                return sum(1 for row in csv.DictReader(report_file) if row.get("tag"))
        except (OSError, csv.Error):
            return 0

    def _clear_result_directory(self):
        """Delete all stored simulation and aggregate results after confirmation."""
        result_dir = Path("result")
        if not result_dir.exists():
            return
        if not messagebox.askyesno(
            "Clear results",
            "Delete all simulation results and aggregate batch reports?",
        ):
            return
        shutil.rmtree(result_dir)
        self.batch_status_var.set("Ready: 0 completed")

    def _launch_batch(self):
        """Start the saved batch in a daemon worker thread."""
        self.is_batch_running = True
        self.batch_cancel_requested = False
        self.batch_status_var.set("Starting batch")
        self.batch_start_button.config(state=tk.DISABLED)
        self.batch_stop_button.config(state=tk.NORMAL)
        self.notebook.select(self.batch_tab)
        worker = threading.Thread(target=self._run_batch_thread, daemon=True)
        worker.start()

    def _run_batch_thread(self):
        """Run batch rows and relay worker progress to the main Tk thread."""
        def report_progress(completed, total, tag, status):
            """Schedule one batch status update on the Tk event loop."""
            self.root.after(0, self._update_batch_status, completed, total, tag, status)

        try:
            run_batch(
                self.batch_path,
                self.config_file,
                should_cancel=lambda: self.batch_cancel_requested,
                progress_callback=report_progress,
            )
        except Exception as error:
            self.root.after(0, messagebox.showerror, "Batch error", str(error))
        finally:
            self.root.after(0, self._finish_batch)

    def _update_batch_status(self, completed, total, tag, status):
        """Display one batch runner progress event."""
        self.batch_status_var.set(f"{completed}/{total}: {tag} ({status})")

    def _finish_batch(self):
        """Restore Batch tab controls after the worker exits."""
        self.is_batch_running = False
        self.batch_start_button.config(state=tk.NORMAL)
        self.batch_stop_button.config(state=tk.DISABLED)
        if self.batch_cancel_requested:
            self.batch_status_var.set("Batch cancelled")
        else:
            self.batch_status_var.set("Batch complete")

    def _stop_batch(self):
        """Request cooperative cancellation of the active batch worker."""
        if not self.is_batch_running:
            return
        self.batch_cancel_requested = True
        self.batch_status_var.set("Cancellation requested")
        self.batch_stop_button.config(state=tk.DISABLED)

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
        self.is_config_dirty = True
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
        """Callback to display log messages in GUI text widget"""
        if hasattr(self, "log_text"):
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)
            self.root.update()

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

    def _load_config_to_ui(self):
        """Load config values into UI fields"""
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
            row["configured"].set("X" if name in self.config else "")
            row["value"].set(str(self.config.get(name, "")))

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
        headings = ("Configured", "Name", "Value", "Bounds", "Description")
        for column, heading in enumerate(headings):
            ttk.Label(
                self.model_settings_frame,
                text=heading,
                font=("TkDefaultFont", 9, "bold"),
            ).grid(row=0, column=column, padx=4, pady=(0, 4), sticky=tk.W)
        for row_index, (name, metadata) in enumerate(self.model_metadata["settings"].items(), start=1):
            configured = tk.StringVar(value="X" if name in self.config else "")
            value = tk.StringVar(value=str(self.config.get(name, metadata["default"])))
            bounds = tk.StringVar(value=self._format_setting_bounds(metadata))
            description = tk.StringVar(value=metadata["description"])
            self.model_setting_rows[name] = {
                "configured": configured,
                "value": value,
                "bounds": bounds,
                "description": description,
            }
            ttk.Label(self.model_settings_frame, textvariable=configured, width=10).grid(
                row=row_index, column=0, padx=4, sticky=tk.W,
            )
            ttk.Label(self.model_settings_frame, text=name, width=28).grid(
                row=row_index, column=1, padx=4, sticky=tk.W,
            )
            ttk.Entry(self.model_settings_frame, textvariable=value, width=18).grid(
                row=row_index, column=2, padx=4, sticky=tk.W,
            )
            ttk.Label(self.model_settings_frame, textvariable=bounds, width=25).grid(
                row=row_index, column=3, padx=4, sticky=tk.W,
            )
            ttk.Label(self.model_settings_frame, textvariable=description).grid(
                row=row_index, column=4, padx=4, sticky=tk.W,
            )

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
            self.is_config_dirty = True
            self._rebuild_model_settings_grid()
            self._rebuild_dynamic_graph_tabs()
            self.batch_column_selector.configure(values=list(metadata["settings"]))
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
        self.is_config_dirty = True
        self._rebuild_model_settings_grid()
        self._rebuild_dynamic_graph_tabs()
        self.batch_column_selector.configure(values=list(metadata["settings"]))
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
            self.is_config_dirty = True
        return True

    def _start_simulation(self):
        """Save unsaved changes before launching a simulation."""
        if not self._update_config_from_ui():
            return
        if self.is_config_dirty:
            should_save = messagebox.askyesno(
                "Unsaved configuration",
                "Save configuration before starting the simulation?",
            )
            if not should_save:
                return
            self._save_config()
        self._launch_simulation()

    def _launch_simulation(self):
        """Launch the configured simulation in a background thread."""
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # Switch to Progress tab
        self.notebook.select(1)
        
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
            
            self.simulation = PopulationSimulation(self.config)
            # Initialize start time for performance tracking
            self.simulation.start_time = time.perf_counter()
            completed_naturally = False
            
            # Run iterative steps (allows stop button to work)
            while self.is_running:
                has_next = self.simulation.step()
                if self.simulation.year > 0:
                    latest_year = self.simulation.year - 1
                    self.root.after(0, self._display_year_graphs, str(self.simulation.output_dir), latest_year)
                    # Update performance statistics
                    self.root.after(0, self._update_performance_stats)
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
                    self.root.after(
                        0,
                        self._display_dynamic_graphs,
                        str(output_dir),
                        last_year,
                        True,
                    )
                log("Results exported (normal completion)" if completed_naturally else "Results exported (manual stop)")
                
                # Show summary graph popup (call from main thread)
                summary_path = output_dir / "results_summary.png"
                if summary_path.exists():
                    self.root.after(0, self._show_summary_graph_popup, str(summary_path))
            
        except Exception as e:
            log(f"Error during simulation: {e}")
            messagebox.showerror("Simulation Error", str(e))
        finally:
            self.is_running = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    def _update_performance_stats(self):
        """Update performance statistics display from current simulation state"""
        if not self.simulation or not self.simulation.start_time:
            return
        
        elapsed_sec = time.perf_counter() - self.simulation.start_time
        year = self.simulation.year
        total_animals = self.simulation.total_animals_processed
        
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
        log("Simulation stopped by user")


def main():
    """Launch GUI"""
    root = tk.Tk()
    gui = SimulationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
