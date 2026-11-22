"""Desktop GUI for SCATS traffic forecasting and traffic-aware route search.

Run from the repository root with ``python -m tbrgs.gui``.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from ml.predict_model import MODEL_NAMES, WINDOW_SIZES, predict_traffic
from tbrgs.app import run_tbrgs

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SITES = BASE_DIR / "data" / "tbrgs" / "scats_sites.csv"
DEFAULT_LINKS = BASE_DIR / "data" / "tbrgs" / "road_links.csv"


class TrafficRoutingGUI(tk.Tk):
    """Tkinter interface for the existing prediction and A* integration."""

    BG = "#f3f6fb"
    CARD = "#ffffff"
    INK = "#17243a"
    MUTED = "#607089"
    BLUE = "#2864dc"
    GREEN = "#138a68"

    def __init__(self) -> None:
        super().__init__()
        self.title("SCATS Traffic Forecast & Route Planner")
        self.geometry("980x720")
        self.minsize(820, 620)
        self.configure(bg=self.BG)
        self._events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._busy = False
        self._setup_styles()
        self._build_ui()
        self.after(120, self._drain_events)

    def _setup_styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=self.BG)
        style.configure("Card.TFrame", background=self.CARD)
        style.configure("TLabel", background=self.BG, foreground=self.INK, font=("Arial", 10))
        style.configure("Card.TLabel", background=self.CARD, foreground=self.INK, font=("Arial", 10))
        style.configure("Title.TLabel", background=self.BG, foreground=self.INK, font=("Arial", 22, "bold"))
        style.configure("Subtitle.TLabel", background=self.BG, foreground=self.MUTED, font=("Arial", 10))
        style.configure("Section.TLabel", background=self.CARD, foreground=self.INK, font=("Arial", 13, "bold"))
        style.configure("Metric.TLabel", background=self.CARD, foreground=self.BLUE, font=("Arial", 22, "bold"))
        style.configure("Accent.TButton", font=("Arial", 10, "bold"), padding=(14, 9))
        style.map("Accent.TButton", background=[("!disabled", self.BLUE), ("disabled", "#aab8d0")], foreground=[("!disabled", "white")])
        style.configure("TNotebook", background=self.BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(18, 10), font=("Arial", 10, "bold"))
        style.configure("Treeview", rowheight=27, font=("Arial", 9))
        style.configure("Treeview.Heading", font=("Arial", 9, "bold"))

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=(28, 22, 28, 18))
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Traffic Forecast & Route Planner", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text="Forecast the next 15-minute SCATS traffic flow, then use it as the A* route cost.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(5, 17))

        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill="both", expand=True)
        self.forecast_tab = ttk.Frame(self.notebook, padding=20)
        self.route_tab = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(self.forecast_tab, text="Traffic forecast")
        self.notebook.add(self.route_tab, text="Route planner")
        self._build_forecast_tab()
        self._build_route_tab()

        self.status_var = tk.StringVar(value="Ready")
        status = ttk.Label(outer, textvariable=self.status_var, style="Subtitle.TLabel")
        status.pack(anchor="w", pady=(12, 0))

    @staticmethod
    def _entry(parent: ttk.Frame, row: int, col: int, label: str, variable: tk.StringVar, width: int = 24) -> None:
        ttk.Label(parent, text=label, style="Card.TLabel").grid(row=row, column=col, sticky="w", padx=(0, 12), pady=8)
        ttk.Entry(parent, textvariable=variable, width=width).grid(row=row, column=col + 1, sticky="ew", pady=8)

    @staticmethod
    def _model_fields(parent: ttk.Frame, row: int, model_var: tk.StringVar, window_var: tk.StringVar) -> None:
        ttk.Label(parent, text="Model", style="Card.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 12), pady=8)
        ttk.Combobox(parent, textvariable=model_var, values=MODEL_NAMES, state="readonly", width=21).grid(row=row, column=1, sticky="ew", pady=8)
        ttk.Label(parent, text="Lookback", style="Card.TLabel").grid(row=row, column=2, sticky="w", padx=(18, 12), pady=8)
        ttk.Combobox(parent, textvariable=window_var, values=tuple(map(str, WINDOW_SIZES)), state="readonly", width=10).grid(row=row, column=3, sticky="ew", pady=8)
        ttk.Label(parent, text="15-minute steps", style="Card.TLabel").grid(row=row, column=4, sticky="w", padx=(8, 0), pady=8)

    def _build_forecast_tab(self) -> None:
        tab = self.forecast_tab
        card = ttk.Frame(tab, style="Card.TFrame", padding=22)
        card.pack(fill="x", anchor="n")
        ttk.Label(card, text="Predict one SCATS site", style="Section.TLabel").grid(row=0, column=0, columnspan=5, sticky="w", pady=(0, 12))
        for col in (1, 3):
            card.columnconfigure(col, weight=1)
        self.forecast_site = tk.StringVar(value="2000")
        self.forecast_time = tk.StringVar(value="2006-10-27 08:00")
        self.forecast_model = tk.StringVar(value="GRU")
        self.forecast_window = tk.StringVar(value="8")
        self._entry(card, 1, 0, "SCATS site ID", self.forecast_site)
        self._entry(card, 2, 0, "Prediction time", self.forecast_time)
        self._model_fields(card, 3, self.forecast_model, self.forecast_window)
        self.forecast_button = ttk.Button(card, text="Predict traffic", style="Accent.TButton", command=self._predict)
        self.forecast_button.grid(row=4, column=0, columnspan=5, sticky="w", pady=(12, 2))

        result = ttk.Frame(tab, style="Card.TFrame", padding=22)
        result.pack(fill="x", pady=(18, 0))
        ttk.Label(result, text="Prediction result", style="Section.TLabel").pack(anchor="w")
        self.prediction_value = tk.StringVar(value="—")
        ttk.Label(result, textvariable=self.prediction_value, style="Metric.TLabel").pack(anchor="w", pady=(12, 3))
        self.prediction_detail = tk.StringVar(value="Vehicles in the next 15-minute interval")
        ttk.Label(result, textvariable=self.prediction_detail, style="Card.TLabel").pack(anchor="w")
        ttk.Label(
            tab,
            text="The selected timestamp must be on a 15-minute boundary and have consecutive historical data.",
            style="Subtitle.TLabel",
            wraplength=800,
        ).pack(anchor="w", pady=(16, 0))

    def _build_route_tab(self) -> None:
        tab = self.route_tab
        form = ttk.Frame(tab, style="Card.TFrame", padding=20)
        form.pack(fill="x")
        ttk.Label(form, text="Route request", style="Section.TLabel").grid(row=0, column=0, columnspan=5, sticky="w", pady=(0, 8))
        for col in (1, 3):
            form.columnconfigure(col, weight=1)
        self.origin_var = tk.StringVar(value="2000")
        self.destination_var = tk.StringVar(value="3002")
        self.route_time = tk.StringVar(value="2006-10-27 08:00")
        self.route_model = tk.StringVar(value="GRU")
        self.route_window = tk.StringVar(value="8")
        self.sites_path = tk.StringVar(value=str(DEFAULT_SITES))
        self.links_path = tk.StringVar(value=str(DEFAULT_LINKS))
        self.flow_source = tk.StringVar(value="destination")
        self.speed_branch = tk.StringVar(value="free_flow")
        self._entry(form, 1, 0, "Origin SCATS ID", self.origin_var)
        self._entry(form, 2, 0, "Destination SCATS ID", self.destination_var)
        self._entry(form, 3, 0, "Prediction time", self.route_time)
        self._model_fields(form, 4, self.route_model, self.route_window)
        self._file_row(form, 5, "Sites CSV", self.sites_path)
        self._file_row(form, 6, "Road links CSV", self.links_path)
        ttk.Label(form, text="Link flow source", style="Card.TLabel").grid(row=7, column=0, sticky="w", padx=(0, 12), pady=8)
        ttk.Combobox(form, textvariable=self.flow_source, values=("origin", "destination"), state="readonly", width=20).grid(row=7, column=1, sticky="ew", pady=8)
        ttk.Label(form, text="Speed curve branch", style="Card.TLabel").grid(row=7, column=2, sticky="w", padx=(18, 12), pady=8)
        ttk.Combobox(form, textvariable=self.speed_branch, values=("free_flow", "congested"), state="readonly", width=18).grid(row=7, column=3, sticky="ew", pady=8)
        self.route_button = ttk.Button(form, text="Find best route", style="Accent.TButton", command=self._route)
        self.route_button.grid(row=8, column=0, columnspan=5, sticky="w", pady=(12, 2))

        summary = ttk.Frame(tab, style="Card.TFrame", padding=18)
        summary.pack(fill="x", pady=(14, 12))
        ttk.Label(summary, text="Route result", style="Section.TLabel").pack(anchor="w")
        self.route_summary = tk.StringVar(value="Select populated site and road-link CSV files to run a real-network search.")
        ttk.Label(summary, textvariable=self.route_summary, style="Card.TLabel", wraplength=850, justify="left").pack(anchor="w", pady=(8, 0))

        table_frame = ttk.Frame(tab, style="Card.TFrame", padding=12)
        table_frame.pack(fill="both", expand=True)
        columns = ("link", "distance", "flow", "speed", "travel")
        self.edge_table = ttk.Treeview(table_frame, columns=columns, show="headings", height=6)
        headings = (("link", "Directed link"), ("distance", "Distance km"), ("flow", "Flow / hour"), ("speed", "Speed km/h"), ("travel", "Travel seconds"))
        for key, label in headings:
            self.edge_table.heading(key, text=label)
            self.edge_table.column(key, anchor="center", width=130)
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.edge_table.yview)
        self.edge_table.configure(yscrollcommand=scrollbar.set)
        self.edge_table.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        ttk.Label(
            tab,
            text="A* route costs use travel time derived from the selected model. The CSV road network must contain verified directed links.",
            style="Subtitle.TLabel",
            wraplength=850,
        ).pack(anchor="w", pady=(10, 0))

    def _file_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar) -> None:
        ttk.Label(parent, text=label, style="Card.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 12), pady=8)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, columnspan=3, sticky="ew", pady=8)
        ttk.Button(parent, text="Browse…", command=lambda v=variable: self._browse_csv(v)).grid(row=row, column=4, padx=(10, 0), pady=8)

    def _browse_csv(self, variable: tk.StringVar) -> None:
        selected = filedialog.askopenfilename(
            title="Select a CSV file",
            filetypes=(("CSV files", "*.csv"), ("All files", "*")),
        )
        if selected:
            variable.set(selected)

    def _start_job(self, operation: str, action: Callable[[], object]) -> None:
        if self._busy:
            return
        self._busy = True
        self.forecast_button.configure(state="disabled")
        self.route_button.configure(state="disabled")
        self.status_var.set("Working…")

        def worker() -> None:
            try:
                result = action()
                self._events.put((operation, result))
            except Exception as error:  # Surface runtime/data/model errors in the GUI.
                self._events.put(("error", f"{type(error).__name__}: {error}"))

        threading.Thread(target=worker, daemon=True).start()

    def _predict(self) -> None:
        try:
            site_id = int(self.forecast_site.get().strip())
            lookback = int(self.forecast_window.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Enter an integer SCATS site ID and select a lookback.")
            return
        model_name = self.forecast_model.get()
        prediction_time = self.forecast_time.get().strip()
        self._start_job(
            "prediction",
            lambda: predict_traffic(site_id, prediction_time, model_name=model_name, lookback=lookback),
        )

    def _route(self) -> None:
        try:
            origin = int(self.origin_var.get().strip())
            destination = int(self.destination_var.get().strip())
            lookback = int(self.route_window.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Origin, destination, and lookback must be valid integers.")
            return
        if not Path(self.sites_path.get()).is_file() or not Path(self.links_path.get()).is_file():
            messagebox.showerror(
                "Road network files not found",
                "Choose populated SCATS sites and directed road-links CSV files. The repository currently contains templates, not a verified route network.",
            )
            return
        args = (
            origin,
            destination,
            self.route_time.get().strip(),
            self.route_model.get(),
            lookback,
            Path(self.sites_path.get()),
            Path(self.links_path.get()),
            self.flow_source.get(),
            self.speed_branch.get(),
        )
        self._start_job("route", lambda: run_tbrgs(*args))

    def _drain_events(self) -> None:
        try:
            while True:
                operation, value = self._events.get_nowait()
                if operation == "prediction":
                    self.prediction_value.set(f"{value.predicted_flow:,.1f} vehicles / 15 min")
                    self.prediction_detail.set(
                        f"SCATS {value.scats_id} · {value.model_name} · {self.forecast_window.get()} × 15-minute history · {value.prediction_time}"
                    )
                    self.status_var.set("Prediction complete")
                elif operation == "route":
                    result, estimates, config = value
                    if result.goal is None:
                        self.route_summary.set("No route found between the selected sites.")
                    else:
                        path = " → ".join(map(str, result.path))
                        self.route_summary.set(
                            f"Route: {path}\nEstimated time: {result.path_cost / 60:.2f} minutes ({result.path_cost:.1f} seconds) · A* created {result.nodes_created} nodes · {len(estimates)} predicted links · flow={config.flow_source}, speed={config.speed_branch}"
                        )
                    for item in self.edge_table.get_children():
                        self.edge_table.delete(item)
                    for (source, target), estimate in estimates.items():
                        self.edge_table.insert(
                            "", "end",
                            values=(
                                f"{source} → {target}",
                                f"{estimate.distance_km:.2f}",
                                f"{estimate.flow_vehicles_per_hour:.1f}",
                                f"{estimate.speed_kmh:.1f}",
                                f"{estimate.total_seconds:.1f}",
                            ),
                        )
                    self.status_var.set("Route search complete")
                elif operation == "error":
                    self.status_var.set("Action failed")
                    messagebox.showerror("Could not complete request", str(value))
                self._busy = False
                self.forecast_button.configure(state="normal")
                self.route_button.configure(state="normal")
        except queue.Empty:
            pass
        self.after(120, self._drain_events)


def main() -> None:
    app = TrafficRoutingGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
