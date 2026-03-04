"""
Tkinter GUI for chimp evolution simulation
Allows parameter configuration, execution control, and result visualization
"""

import json
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import torch
from PIL import Image, ImageTk

from main import PopulationSimulation, set_logger, log
from settings import DEFAULT_SETTINGS, PARAMETER_RANGES, PARAMETER_DESCRIPTIONS, PARAMETER_GROUPS


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

    def _load_config(self):
        """Load configuration from file or use defaults"""
        if Path(self.config_file).exists():
            with open(self.config_file) as f:
                return json.load(f)
        else:
            return DEFAULT_SETTINGS.copy()

    def _save_config(self):
        """Save configuration to file"""
        with open(self.config_file, "w") as f:
            json.dump(self.config, f, indent=2)
        log(f"Config saved to {self.config_file}")

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
        
        # Control buttons at bottom
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.start_btn = ttk.Button(control_frame, text="Start Simulation", command=self._start_simulation)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self._stop_simulation, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Save Config", command=self._on_save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Load Config", command=self._on_load_config).pack(side=tk.LEFT, padx=5)

    def _create_settings_tab(self, parent):
        """Create settings input fields organized by groups"""
        # Device selection
        device_frame = ttk.LabelFrame(parent, text="Device", padding=10)
        device_frame.pack(fill=tk.X, pady=5, padx=5)
        
        available_cuda = torch.cuda.is_available()
        self.device_var = tk.StringVar(value=self.config.get("device", "cuda" if available_cuda else "cpu"))
        
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
        
        # Scrollable parameters frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create input fields organized by groups
        self.setting_vars = {}
        
        for group_name, param_list in PARAMETER_GROUPS.items():
            # Create group frame
            group_frame = ttk.LabelFrame(scrollable_frame, text=group_name, padding=10)
            group_frame.pack(fill=tk.X, padx=5, pady=8)
            
            for param in param_list:
                if param not in DEFAULT_SETTINGS:
                    continue
                    
                default_val = DEFAULT_SETTINGS[param]
                frame = ttk.Frame(group_frame)
                frame.pack(fill=tk.X, pady=3)
                
                # Parameter name (left-aligned)
                ttk.Label(frame, text=param, width=25).pack(side=tk.LEFT)
                
                # Input field
                var = tk.StringVar(value=str(self.config.get(param, default_val)))
                self.setting_vars[param] = var
                ttk.Entry(frame, textvariable=var, width=15).pack(side=tk.LEFT, padx=5)
                
                # Range indicator
                if param in PARAMETER_RANGES:
                    min_val, max_val = PARAMETER_RANGES[param]
                    ttk.Label(frame, text=f"[{min_val}, {max_val}]", foreground="gray", width=18).pack(side=tk.LEFT)
                
                # Description
                if param in PARAMETER_DESCRIPTIONS:
                    ttk.Label(frame, text=PARAMETER_DESCRIPTIONS[param], foreground="blue", font=("TkDefaultFont", 8)).pack(side=tk.LEFT, padx=5)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Tag field
        tag_frame = ttk.Frame(parent)
        tag_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(tag_frame, text="Run tag", width=25).pack(side=tk.LEFT)
        self.tag_var = tk.StringVar(value=self.config.get("tag", "default"))
        ttk.Entry(tag_frame, textvariable=self.tag_var, width=15).pack(side=tk.LEFT, padx=5)

    def _create_progress_tab(self, parent):
        """Create progress and statistics display"""
        graph_frame = ttk.LabelFrame(parent, text="Yearly Graphs", padding=8)
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        graphs_row = ttk.Frame(graph_frame)
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
        self.device_var.set(self.config.get("device", "cuda"))
        for param, var in self.setting_vars.items():
            var.set(str(self.config.get(param, DEFAULT_SETTINGS.get(param, ""))))

    def _on_save_config(self):
        """Save button handler"""
        self._update_config_from_ui()
        self._save_config()
        messagebox.showinfo("Success", "Configuration saved")

    def _on_load_config(self):
        """Load button handler"""
        file_path = filedialog.askopenfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            self.config_file = file_path
            self.config = self._load_config()
            self._load_config_to_ui()
            messagebox.showinfo("Success", f"Configuration loaded from {file_path}")

    def _update_config_from_ui(self):
        """Read config from UI fields and validate"""
        self.config["device"] = self.device_var.get()
        self.config["tag"] = self.tag_var.get()
        
        for param, var in self.setting_vars.items():
            try:
                val_str = var.get()
                # Try to parse as float first, then int
                if "." in val_str:
                    val = float(val_str)
                else:
                    val = int(val_str) if param not in ["lambda", "alpha", "beta_initial", "mutation_probability", 
                                                          "mutation_x", "mutation_s"] else float(val_str)
                
                # Validate range
                if param in PARAMETER_RANGES:
                    min_val, max_val = PARAMETER_RANGES[param]
                    if not (min_val <= val <= max_val):
                        messagebox.showwarning("Validation", f"{param} must be in [{min_val}, {max_val}]")
                        var.set(str(self.config.get(param, DEFAULT_SETTINGS[param])))
                        return False
                
                self.config[param] = val
            except ValueError:
                messagebox.showerror("Error", f"Invalid value for {param}: {var.get()}")
                return False
        
        return True

    def _start_simulation(self):
        """Start simulation in background thread"""
        if not self._update_config_from_ui():
            return
        
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
            
            # Run iterative steps (allows stop button to work)
            while self.is_running:
                has_next = self.simulation.step()
                if self.simulation.year > 0:
                    latest_year = self.simulation.year - 1
                    self.root.after(0, self._display_year_graphs, str(self.simulation.output_dir), latest_year)
                    # Update performance statistics
                    self.root.after(0, self._update_performance_stats)
                if not has_next:
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
                self.simulation.export_results()
                log("Results exported (normal completion)" if self.is_running else "Results exported (manual stop)")
                
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
