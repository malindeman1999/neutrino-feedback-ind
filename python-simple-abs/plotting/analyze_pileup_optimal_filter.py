"""Analyze saved pulse sweeps with a single-pulse optimal filter."""

from __future__ import annotations

from pathlib import Path
import pickle
import sys
import tkinter as tk
from tkinter import filedialog, ttk

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pileup_sweep import analyze_pulse_sweep_dataset


PLOT_DIR = Path(__file__).resolve().parent
SAVES_DIR = PLOT_DIR / "saves"
PULSE_DATA_DIR = PLOT_DIR / "pulses and noise"
ANALYZER_STATE_FILE = SAVES_DIR / "analyze_pileup_gui_state.pkl"


def _labels(dataset: dict) -> tuple[str, str]:
    return str(dataset.get("readout_label", "Readout")), str(dataset.get("units", {}).get("trace", "1"))


def _min_double_energy(dataset: dict) -> np.ndarray:
    return np.min(np.asarray(dataset["double"]["energies_eV"], dtype=float), axis=1)


def _lag_norm(results: dict[str, dict[str, np.ndarray]]) -> Normalize:
    lag_max_us = max(float(np.max(results["double"]["lag_s"])) * 1.0e6, 1.0e-12)
    return Normalize(vmin=0.0, vmax=lag_max_us)


def _single_pulse_fit_trace(dataset: dict, fit_energy_eV: float, fit_time_s: float) -> np.ndarray:
    n_samples = int(dataset["generation"]["n_samples"])
    template_fft = np.asarray(dataset["optimal_filter"]["template_fft_per_eV"], dtype=complex)
    freqs_hz = np.asarray(dataset["frequencies_hz"], dtype=float)
    shift = np.exp(-2.0j * np.pi * freqs_hz * float(fit_time_s))
    return np.fft.irfft(float(fit_energy_eV) * template_fft * shift, n=n_samples)


def _plot_view(x: np.ndarray, *ys: np.ndarray, max_points: int = 20_000) -> tuple[np.ndarray, ...]:
    if x.size <= max_points:
        return (x, *ys)
    step = int(np.ceil(x.size / max_points))
    return (x[::step], *(y[::step] for y in ys))


def _attach_event_picker(
    fig: plt.Figure,
    dataset: dict,
    results: dict[str, dict[str, np.ndarray]],
    artists: dict[object, tuple[str, np.ndarray]],
) -> None:
    label, unit = _labels(dataset)
    time_ms = np.asarray(dataset["time_s"], dtype=float) * 1.0e3

    def open_event_plot(category: str, idx: int) -> None:
        trace = np.asarray(dataset[category]["trace"][idx], dtype=float)
        fit_energy_eV = float(results[category]["fit_energy_eV"][idx])
        fit_time_s = float(results[category]["fit_time_s"][idx])
        fit = _single_pulse_fit_trace(dataset, fit_energy_eV, fit_time_s)
        residual = trace - fit
        plot_time_ms, plot_trace, plot_fit, plot_residual = _plot_view(time_ms, trace, fit, residual)
        energies = np.atleast_1d(np.asarray(dataset[category]["energies_eV"][idx], dtype=float))
        chi2 = float(results[category]["reduced_chi2"][idx])
        if category == "single":
            description = f"Single pulse: E={energies[0]:.4g} eV"
        else:
            lag_us = float(dataset[category]["lag_s"][idx]) * 1.0e6
            description = f"Double pulse: E1={energies[0]:.4g} eV, E2={energies[1]:.4g} eV, lag={lag_us:.4g} us"
        detail, (ax_trace, ax_resid) = plt.subplots(
            2,
            1,
            figsize=(8.4, 6.0),
            sharex=True,
            gridspec_kw={"height_ratios": [2.3, 1.0]},
            constrained_layout=True,
        )
        ax_trace.plot(plot_time_ms, plot_trace, label="Sweep")
        ax_trace.plot(plot_time_ms, plot_fit, linestyle="--", label="Single-pulse fit")
        ax_trace.set_ylabel(f"{label} [{unit}]")
        ax_trace.set_title(f"{description}; fit E={fit_energy_eV:.4g} eV; reduced chi2={chi2:.4g}")
        ax_trace.grid(True, alpha=0.25)
        ax_trace.legend(loc="best")
        ax_resid.plot(plot_time_ms, plot_residual, color="tab:red")
        ax_resid.axhline(0.0, color="black", linestyle=":", linewidth=0.9)
        ax_resid.set_xlabel("Time [ms]")
        ax_resid.set_ylabel(f"Residual [{unit}]")
        ax_resid.grid(True, alpha=0.25)
        detail.show()

    def on_pick(event) -> None:
        selected = artists.get(event.artist)
        if selected is None or len(event.ind) == 0:
            return
        category, indices = selected
        open_event_plot(category, int(indices[int(event.ind[0])]))

    fig.canvas.mpl_connect("pick_event", on_pick)


def make_energy_closure_figure(dataset: dict, results: dict[str, dict[str, np.ndarray]]) -> plt.Figure:
    label, _ = _labels(dataset)
    single = results["single"]
    double = results["double"]
    x_double = _min_double_energy(dataset)
    lag_us = np.asarray(double["lag_s"]) * 1.0e6
    fig, ax = plt.subplots(figsize=(6.4, 5.1), constrained_layout=True)
    single_points = ax.scatter(single["true_total_energy_eV"], single["fit_energy_eV"], color="#222222", marker="o", label="Single", picker=5)
    points = ax.scatter(x_double, double["fit_energy_eV"], c=lag_us, cmap="rainbow", norm=_lag_norm(results), marker="x", label="Double", picker=5)
    ax.set_xlabel("Smaller pulse energy [eV]")
    ax.set_ylabel("Single-pulse filter energy [eV]")
    ax.set_title("Optimal-Filter Energy Closure")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.colorbar(points, ax=ax, label="Pulse separation [us]")
    fig.suptitle(f"{label} optimal-filter study")
    _attach_event_picker(fig, dataset, results, {
        single_points: ("single", np.arange(len(single["fit_energy_eV"]), dtype=int)),
        points: ("double", np.arange(len(double["fit_energy_eV"]), dtype=int)),
    })
    return fig


def make_energy_vs_chi2_figure(dataset: dict, results: dict[str, dict[str, np.ndarray]]) -> plt.Figure:
    label, _ = _labels(dataset)
    single = results["single"]
    double = results["double"]
    x_double = _min_double_energy(dataset)
    lag_us = np.asarray(double["lag_s"]) * 1.0e6
    fig, ax = plt.subplots(figsize=(6.4, 5.1), constrained_layout=True)
    single_points = ax.scatter(single["true_total_energy_eV"], single["reduced_chi2"], color="#222222", marker="o", label="Single", picker=5)
    points = ax.scatter(x_double, double["reduced_chi2"], c=lag_us, cmap="rainbow", norm=_lag_norm(results), marker="x", label="Double", picker=5)
    ax.set_xlabel("Smaller pulse energy [eV]")
    ax.set_ylabel("Single-pulse fit reduced chi2")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Pile-Up Discrimination Statistic")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    fig.colorbar(points, ax=ax, label="Pulse separation [us]")
    fig.suptitle(f"{label} optimal-filter study")
    _attach_event_picker(fig, dataset, results, {
        single_points: ("single", np.arange(len(single["fit_energy_eV"]), dtype=int)),
        points: ("double", np.arange(len(double["fit_energy_eV"]), dtype=int)),
    })
    return fig


def make_lag_vs_chi2_figure(dataset: dict, results: dict[str, dict[str, np.ndarray]]) -> plt.Figure:
    label, _ = _labels(dataset)
    double = results["double"]
    single_chi2 = np.asarray(results["single"]["reduced_chi2"], dtype=float)
    energies = np.asarray(dataset["double"]["energies_eV"], dtype=float)
    smaller = np.min(energies, axis=1)
    larger = np.max(energies, axis=1)
    lag_us = np.asarray(double["lag_s"], dtype=float) * 1.0e6
    levels = np.sort(np.unique(smaller))[::-1]
    nrows = int(levels.size)
    fig, axes = plt.subplots(nrows, 1, figsize=(7.3, max(3.2, 2.45 * nrows)), squeeze=False, constrained_layout=True)
    axes_arr = axes[:, 0]
    norm = Normalize(vmin=float(np.min(larger)), vmax=float(np.max(larger)))
    points = None
    artists: dict[object, tuple[str, np.ndarray]] = {}
    for ax, level in zip(axes_arr, levels):
        mask = smaller == level
        indices = np.flatnonzero(mask)
        points = ax.scatter(
            lag_us[mask],
            np.asarray(double["reduced_chi2"], dtype=float)[mask],
            c=larger[mask],
            cmap="rainbow",
            norm=norm,
            marker="x",
            label="Double",
            picker=5,
        )
        artists[points] = ("double", indices)
        single_points = ax.scatter(np.zeros(single_chi2.size), single_chi2, color="#222222", marker="o", s=18, alpha=0.65, label="Single", picker=5)
        artists[single_points] = ("single", np.arange(single_chi2.size, dtype=int))
        ax.set_yscale("log")
        ax.set_ylabel("Reduced chi2")
        ax.set_title(f"Smaller pulse energy = {level:.3g} eV")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best")
    axes_arr[-1].set_xlabel("Pulse separation [us]")
    if points is not None:
        fig.colorbar(points, ax=list(axes_arr), label="Larger pulse energy [eV]")
    fig.suptitle(f"{label}: Chi2 Versus Pulse Separation")
    _attach_event_picker(fig, dataset, results, artists)
    return fig


class AnalyzerGui:
    def __init__(self) -> None:
        SAVES_DIR.mkdir(parents=True, exist_ok=True)
        state = self._load_state()
        self.root = tk.Tk()
        self.root.title("Pile-Up Optimal Filter Analyzer")
        self.root.geometry("780x260")
        self.dataset_var = tk.StringVar(value=state.get("last_dataset_path", ""))
        self.results_var = tk.StringVar(value="")
        self.save_results_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Select a pulse sweep dataset and choose a plot.")
        self.results_entry: ttk.Entry | None = None
        self.results_button: ttk.Button | None = None
        self.analysis_signature: tuple[str, int, int] | None = None
        self.cached_dataset: dict | None = None
        self.cached_results: dict[str, dict[str, np.ndarray]] | None = None
        self._build()

    @staticmethod
    def _load_state() -> dict[str, str]:
        if not ANALYZER_STATE_FILE.exists():
            candidates = list(PULSE_DATA_DIR.glob("*.pkl")) if PULSE_DATA_DIR.exists() else []
            if candidates:
                return {"last_dataset_path": str(max(candidates, key=lambda path: path.stat().st_mtime))}
            return {}
        try:
            with ANALYZER_STATE_FILE.open("rb") as handle:
                state = pickle.load(handle)
            if isinstance(state, dict) and isinstance(state.get("last_dataset_path"), str):
                return {"last_dataset_path": state["last_dataset_path"]}
        except Exception:
            pass
        return {}

    def _save_state(self) -> None:
        try:
            with ANALYZER_STATE_FILE.open("wb") as handle:
                pickle.dump({"last_dataset_path": self.dataset_var.get().strip()}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    def _clear_state_for_path(self, dataset_path: Path) -> None:
        try:
            if not ANALYZER_STATE_FILE.exists():
                return
            with ANALYZER_STATE_FILE.open("rb") as handle:
                state = pickle.load(handle)
            if not isinstance(state, dict) or not isinstance(state.get("last_dataset_path"), str):
                return
            saved_path = Path(state["last_dataset_path"])
            if saved_path.resolve() == dataset_path.resolve():
                ANALYZER_STATE_FILE.unlink()
        except Exception:
            pass

    @staticmethod
    def _load_dataset(dataset_path: Path) -> dict:
        try:
            with dataset_path.open("rb") as handle:
                dataset = pickle.load(handle)
        except (EOFError, pickle.UnpicklingError) as exc:
            raise ValueError(
                f"{dataset_path.name} is incomplete or corrupt. Regenerate and save the pulse sweep dataset."
            ) from exc
        if not isinstance(dataset, dict):
            raise ValueError(f"{dataset_path.name} is not a pulse sweep dataset")
        return dataset

    def _build(self) -> None:
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text="Input dataset (.pkl):").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(frame, textvariable=self.dataset_var).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Button(frame, text="Browse", command=self._pick_dataset).grid(row=0, column=2, sticky="ew", pady=4)
        ttk.Checkbutton(frame, text="Save fit results", variable=self.save_results_var, command=self._toggle_results).grid(row=1, column=0, sticky="w", pady=4)
        self.results_entry = ttk.Entry(frame, textvariable=self.results_var)
        self.results_entry.grid(row=1, column=1, sticky="ew", pady=4)
        self.results_button = ttk.Button(frame, text="Browse", command=self._pick_results)
        self.results_button.grid(row=1, column=2, sticky="ew", pady=4)
        buttons = ttk.Frame(frame)
        buttons.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 6))
        for column in range(4):
            buttons.columnconfigure(column, weight=1)
        ttk.Button(buttons, text="Energy Closure", command=lambda: self._run_plot("energy")).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(buttons, text="Energy vs Chi2", command=lambda: self._run_plot("energy_chi2")).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(buttons, text="Lag vs Chi2", command=lambda: self._run_plot("lag_chi2")).grid(row=0, column=2, sticky="ew", padx=4)
        ttk.Button(buttons, text="Close", command=self.root.destroy).grid(row=0, column=3, sticky="ew", padx=(4, 0))
        ttk.Label(frame, textvariable=self.status_var, wraplength=750, foreground="#333").grid(row=3, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self._toggle_results()

    def _pick_dataset(self) -> None:
        current = self.dataset_var.get().strip()
        initial_dir = Path(current).parent if current else SAVES_DIR
        path = filedialog.askopenfilename(parent=self.root, initialdir=initial_dir, initialfile=Path(current).name if current else "", filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
        if path:
            self.dataset_var.set(path)

    def _pick_results(self) -> None:
        path = filedialog.asksaveasfilename(parent=self.root, initialdir=SAVES_DIR, initialfile="pileup_optimal_filter_results.pkl", defaultextension=".pkl", filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
        if path:
            self.results_var.set(path)

    def _toggle_results(self) -> None:
        state = tk.NORMAL if self.save_results_var.get() else tk.DISABLED
        if self.results_entry is not None:
            self.results_entry.configure(state=state)
        if self.results_button is not None:
            self.results_button.configure(state=state)

    def _run_plot(self, plot_kind: str) -> None:
        dataset_path = Path(self.dataset_var.get().strip()) if self.dataset_var.get().strip() else None
        if dataset_path is None or not dataset_path.exists():
            self.status_var.set("Choose an existing dataset file.")
            return
        try:
            stat = dataset_path.stat()
            signature = (str(dataset_path.resolve()), stat.st_mtime_ns, stat.st_size)
            if signature != self.analysis_signature or self.cached_dataset is None or self.cached_results is None:
                self.status_var.set("Running optimal-filter analysis...")
                self.root.update_idletasks()
                dataset = self._load_dataset(dataset_path)

                def report_progress(message: str) -> None:
                    self.status_var.set(message)
                    self.root.update_idletasks()

                results = analyze_pulse_sweep_dataset(dataset, progress=report_progress)
                self.analysis_signature = signature
                self.cached_dataset = dataset
                self.cached_results = results
            else:
                dataset = self.cached_dataset
                results = self.cached_results
            if self.save_results_var.get():
                output = self.results_var.get().strip()
                if not output:
                    raise ValueError("choose a fit-results output path or uncheck Save fit results")
                result_path = Path(output)
                result_path.parent.mkdir(parents=True, exist_ok=True)
                with result_path.open("wb") as handle:
                    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
            self._save_state()
            if plot_kind == "energy":
                make_energy_closure_figure(dataset, results)
            elif plot_kind == "energy_chi2":
                make_energy_vs_chi2_figure(dataset, results)
            else:
                make_lag_vs_chi2_figure(dataset, results)
            self.status_var.set("Analysis complete. Plot window opened.")
            plt.show()
        except Exception as exc:
            if isinstance(exc, ValueError) and "incomplete or corrupt" in str(exc):
                self._clear_state_for_path(dataset_path)
            self.status_var.set(f"Analysis failed: {exc}")

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    AnalyzerGui().run()


if __name__ == "__main__":
    main()
