"""
PV + battery sizing: desktop economics explorer (no server, no browser, no network).

Run:  python app_desktop.py
Uses Tkinter (ships with Python), matplotlib and the shared econ_core module.
"""
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import pandas as pd
from dataclasses import replace
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3-D projection)


from econ_core import (PROFILES, SURFACES, SWEEP_PARAMS, Economics, allocation_table,
                       best_mix_vs_size, daily_profile, envelope, evaluate, full_report,
                       landscape_grid, load_appl_profile, load_res, optimum, optimum_fast,
                       orientation_scatter_table, pv_incentive, roof_ew_grid,
                       sensitivity_1d, threshold_note, tornado)

_cache = {}

# Original input values, matching the economics and subsidy block in postproc_v3_2.m
DEFAULT_MODE = "whole"
DEFAULT_TIERS = [(10, 150), (20, 140), (100, 130), (1000, 120)]


def _load(path, kind):
    """Load a RES file once per (path, kind, modification time). Keeps one file in memory."""
    key = (path, kind, os.path.getmtime(path))
    if key not in _cache:
        _cache.clear()
        _cache[key] = load_res(path, kind)
    return _cache[key]


def _res_path(folder, profile):
    return os.path.join(folder, f"RES_{profile}.mat")


def run_profile(profile, dir_pv, dir_batt, e):
    """Return (pv_out, batt_out) with full evaluation; either is None if its file is missing."""
    pv_out = batt_out = None
    if os.path.exists(_res_path(dir_pv, profile)):
        pv_out = evaluate(_load(_res_path(dir_pv, profile), "pv"), "pv", e)
    if os.path.exists(_res_path(dir_batt, profile)):
        batt_out = evaluate(_load(_res_path(dir_batt, profile), "batt"), "batt", e)
    return pv_out, batt_out


def _df_or_none(folder, profile, kind):
    path = _res_path(folder, profile)
    return _load(path, kind) if os.path.exists(path) else None


def full_report_all(dir_pv, dir_batt, e):
    parts = []
    for prof in PROFILES:
        pv_df = _df_or_none(dir_pv, prof, "pv")
        batt_df = _df_or_none(dir_batt, prof, "batt")
        if pv_df is None and batt_df is None:
            continue
        parts.append(full_report(prof, pv_df, batt_df, e))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


class SensitivityWindow(tk.Toplevel):
    """Separate window: one-parameter sweeps and a tornado analysis."""

    def __init__(self, parent: "App"):
        super().__init__(parent)
        self.parent = parent
        self.title("Sensitivity analysis")
        self.geometry("1250x820")
        self.last_table = None

        ctrl = ttk.Frame(self, padding=8)
        ctrl.pack(side="left", fill="y")
        right = ttk.Frame(self, padding=8)
        right.pack(side="right", fill="both", expand=True)

        self.var_profile = tk.StringVar(value=parent.var_profile.get())
        self.var_case = tk.StringVar(value="PV only")
        self.var_param = tk.StringVar(value=SWEEP_PARAMS[0][0])
        self.var_min = tk.StringVar()
        self.var_max = tk.StringVar()
        self.var_steps = tk.StringVar(value="9")
        self.var_pct = tk.StringVar(value="20")

        ttk.Label(ctrl, text="Profile").pack(anchor="w")
        ttk.Combobox(ctrl, textvariable=self.var_profile, values=PROFILES,
                     state="readonly", width=34).pack(anchor="w")
        ttk.Label(ctrl, text="Case").pack(anchor="w", pady=(6, 0))
        ttk.Combobox(ctrl, textvariable=self.var_case, values=["PV only", "PV + battery"],
                     state="readonly", width=34).pack(anchor="w")

        ttk.Label(ctrl, text="One-parameter sweep", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 0))
        self.cmb_param = ttk.Combobox(ctrl, textvariable=self.var_param,
                                      values=[s[0] for s in SWEEP_PARAMS], state="readonly", width=34)
        self.cmb_param.pack(anchor="w")
        self.cmb_param.bind("<<ComboboxSelected>>", lambda _e: self.set_default_range())
        grid = ttk.Frame(ctrl)
        grid.pack(anchor="w", pady=2)
        for i, (lbl, var) in enumerate([("From", self.var_min), ("To", self.var_max), ("Steps", self.var_steps)]):
            ttk.Label(grid, text=lbl).grid(row=i, column=0, sticky="w", padx=2)
            ttk.Entry(grid, textvariable=var, width=12).grid(row=i, column=1, padx=2, pady=1)
        ttk.Button(ctrl, text="Run sweep", command=self.run_sweep).pack(anchor="w", pady=4)

        ttk.Label(ctrl, text="Tornado", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 0))
        row = ttk.Frame(ctrl)
        row.pack(anchor="w")
        ttk.Label(row, text="Change ±%").pack(side="left")
        ttk.Entry(row, textvariable=self.var_pct, width=8).pack(side="left", padx=4)
        ttk.Button(ctrl, text="Run tornado (both cases)", command=self.run_tornado).pack(anchor="w", pady=4)

        ttk.Button(ctrl, text="Save last table as CSV", command=self.save_csv).pack(anchor="w", pady=(12, 0))

        self.text = tk.Text(right, height=12, width=90, font=("Consolas", 9))
        self.text.pack(fill="x")
        self.fig = Figure(figsize=(9, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, pady=(8, 0))
        self.set_default_range()

    def set_default_range(self):
        """Fill the range with ±50% around the current main-window value (in display units)."""
        label = self.var_param.get()
        _, field, _, _ = next(s for s in SWEEP_PARAMS if s[0] == label)
        try:
            base = float(self.parent.vars[field].get().replace(",", "."))
        except (KeyError, ValueError):
            base = 1.0
        self.var_min.set(f"{base * 0.5:.4g}")
        self.var_max.set(f"{base * 1.5:.4g}")

    def _case(self):
        prof = self.var_profile.get()
        if self.var_case.get() == "PV + battery":
            return "batt", _df_or_none(self.parent.var_batt.get(), prof, "batt")
        return "pv", _df_or_none(self.parent.var_pv.get(), prof, "pv")

    def _show(self, msg):
        self.text.delete("1.0", "end")
        self.text.insert("end", msg)

    def run_sweep(self):
        try:
            e = self.parent.read_economics()
            label = self.var_param.get()
            _, field, scale, batt_only = next(s for s in SWEEP_PARAMS if s[0] == label)
            kind, df = self._case()
            if df is None:
                messagebox.showwarning("No results", "No RES file for this profile and case.")
                return
            if batt_only and kind == "pv":
                messagebox.showinfo("Not applicable",
                                    "This parameter only affects the PV + battery case. Choose that case.")
                return
            lo, hi = float(self.var_min.get()) / scale, float(self.var_max.get()) / scale
            n = int(self.var_steps.get())
            if n < 2:
                raise ValueError("Steps must be at least 2")
            table = sensitivity_1d(df, kind, e, field, np.linspace(lo, hi, n))
        except Exception as exc:
            messagebox.showerror("Problem", str(exc))
            return

        table.insert(0, label, table["value"] * scale)
        self.last_table = table.assign(Profile=self.var_profile.get(), Case=self.var_case.get())
        show = table[[label, "kWp", "kWh", "EAC_total", "GridDem_kWh", "Incentive_PV", "Incentive_Batt"]]
        self._show(f"{self.var_profile.get()} | {self.var_case.get()} | sweep of {label}\n\n"
                   + show.round(2).to_string(index=False))

        self.fig.clear()
        ax1 = self.fig.add_subplot(1, 2, 1)
        ax1.plot(table[label], table["kWp"], "o-", label="PV [kWp]")
        if kind == "batt":
            ax1.plot(table[label], table["kWh"], "s--", label="Battery [kWh]")
        ax1.set_xlabel(label); ax1.set_ylabel("Optimal size"); ax1.grid(True); ax1.legend()
        ax2 = self.fig.add_subplot(1, 2, 2)
        ax2.plot(table[label], table["EAC_total"], "o-", color="tab:green")
        ax2.set_xlabel(label); ax2.set_ylabel("Optimal total EAC [€/y]"); ax2.grid(True)
        ax2.set_title("Optimal annual cost")
        self.fig.tight_layout()
        self.canvas.draw()

    def run_tornado(self):
        try:
            e = self.parent.read_economics()
            pct = float(self.var_pct.get().replace(",", ".")) / 100
            prof = self.var_profile.get()
            results = []
            for case, kind, folder in (("PV only", "pv", self.parent.var_pv.get()),
                                       ("PV + battery", "batt", self.parent.var_batt.get())):
                df = _df_or_none(folder, prof, kind)
                if df is None:
                    continue
                t, base_opt = tornado(df, kind, e, pct)
                results.append((case, t, base_opt))
            if not results:
                messagebox.showwarning("No results", "No RES files for this profile.")
                return
        except Exception as exc:
            messagebox.showerror("Problem", str(exc))
            return

        lines = [f"Tornado for {prof}, each input changed by ±{pct*100:g}%", ""]
        for case, t, base_opt in results:
            lines.append(f"{case}: baseline EAC {base_opt['EAC_total']:,.0f} €/y, "
                         f"PV {base_opt['kWp']:.2f} kWp, battery {base_opt['kWh']:.1f} kWh")
            lines.append(t[["Parameter", "Swing [€/y]", "Δ low [€/y]", "Δ high [€/y]",
                            "kWp low", "kWp high"]].round(1).to_string(index=False))
            lines.append("")
        self._show("\n".join(lines))

        self.fig.clear()
        for k, (case, t, base_opt) in enumerate(results):
            ax = self.fig.add_subplot(1, len(results), k + 1)
            y = np.arange(len(t))
            ax.hlines(y, t["Δ low [€/y]"], t["Δ high [€/y]"], linewidth=8, color="tab:blue")
            ax.axvline(0, color="k", lw=0.8)
            ax.set_yticks(y); ax.set_yticklabels(t["Parameter"], fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel(f"Change in optimal EAC [€/y] ({case})")
            ax.grid(True, axis="x")
        self.fig.tight_layout()
        self.canvas.draw()
        self.last_table = pd.concat(
            [t.assign(Case=case, Profile=prof) for case, t, _ in results], ignore_index=True)

    def save_csv(self):
        if self.last_table is None:
            messagebox.showinfo("Nothing to save", "Run a sweep or tornado first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                            initialfile="sensitivity.csv")
        if path:
            self.last_table.to_csv(path, index=False)



class OrientationWindow(tk.Toplevel):
    """Separate window: how the optimum splits PV between balcony, east and west roofs."""

    def __init__(self, parent: "App"):
        super().__init__(parent)
        self.parent = parent
        self.title("PV orientation analysis")
        self.geometry("1250x850")

        ctrl = ttk.Frame(self, padding=8)
        ctrl.pack(side="left", fill="y")
        right = ttk.Frame(self, padding=8)
        right.pack(side="right", fill="both", expand=True)

        self.var_profile = tk.StringVar(value=parent.var_profile.get())
        self.var_case = tk.StringVar(value="PV only")
        self.var_bin = tk.StringVar(value="1.0")

        ttk.Label(ctrl, text="Profile").pack(anchor="w")
        ttk.Combobox(ctrl, textvariable=self.var_profile, values=PROFILES,
                     state="readonly", width=34).pack(anchor="w")
        ttk.Label(ctrl, text="Case").pack(anchor="w", pady=(6, 0))
        ttk.Combobox(ctrl, textvariable=self.var_case, values=["PV only", "PV + battery"],
                     state="readonly", width=34).pack(anchor="w")
        ttk.Label(ctrl, text="Size bin [kWp]").pack(anchor="w", pady=(6, 0))
        ttk.Entry(ctrl, textvariable=self.var_bin, width=12).pack(anchor="w")

        ttk.Button(ctrl, text="Show orientation for profile",
                   command=self.run_profile).pack(anchor="w", pady=(10, 2))
        ttk.Button(ctrl, text="Optimal allocation: all profiles",
                   command=self.run_all).pack(anchor="w", pady=2)

        self.text = tk.Text(right, height=6, width=100, font=("Consolas", 9))
        self.text.pack(fill="x")
        self.fig = Figure(figsize=(11, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, pady=(6, 0))
        self.text.insert("end", "Choose a profile and case, then press 'Show orientation for profile'.\n"
                                "Surface labels are inferred from the MATLAB names: S = balcony, "
                                "RE/RW = roof east/west, ReF/RwF = flat roof east/west.")

    def _df(self, kind, prof):
        folder = self.parent.var_batt.get() if kind == "batt" else self.parent.var_pv.get()
        return _df_or_none(folder, prof, kind)

    def run_profile(self):
        try:
            e = self.parent.read_economics()
            prof = self.var_profile.get()
            kind = "batt" if self.var_case.get() == "PV + battery" else "pv"
            df = self._df(kind, prof)
            if df is None:
                messagebox.showwarning("No results", "No RES file for this profile and case.")
                return
            bin_kwp = float(self.var_bin.get().replace(",", "."))
            if bin_kwp <= 0:
                raise ValueError("Size bin must be positive")
            base = optimum_fast(df, kind, e)
            mix = best_mix_vs_size(df, kind, e, bin_kwp)
            grid = roof_ew_grid(df, kind, e, base)
        except Exception as exc:
            messagebox.showerror("Problem", str(exc))
            return

        areas = {lbl: base[c] * e.eta_pv for c, lbl in SURFACES}
        self.text.delete("1.0", "end")
        self.text.insert("end",
            f"{prof} | {self.var_case.get()} | optimum {base['kWp']:.2f} kWp, "
            f"battery {base['kWh']:.1f} kWh, EAC {base['EAC_total']:,.0f} €/y\n"
            "Optimal kWp per surface: " +
            ", ".join(f"{k} {v:.2f}" for k, v in areas.items()))

        self.fig.clear()
        labels = [lbl for _, lbl in SURFACES]
        cols = [c for c, _ in SURFACES]
        ax1 = self.fig.add_subplot(2, 2, 1)
        ax1.stackplot(mix["kWp"], [mix[c] for c in cols], labels=labels, alpha=0.85)
        ax1.axvline(base["kWp"], color="k", ls="--", lw=0.8)
        ax1.set_xlabel("Total installed PV [kWp]"); ax1.set_ylabel("kWp per surface")
        ax1.set_title("Best surface allocation for each total size")
        ax1.legend(fontsize=7, loc="upper left"); ax1.grid(True)

        ax2 = self.fig.add_subplot(2, 2, 2)
        ax2.plot(mix["kWp"], mix["EAC_total"], "o-", ms=3, color="tab:green")
        ax2.plot(base["kWp"], base["EAC_total"], "r*", ms=14, label="Optimum")
        ax2.set_xlabel("Total installed PV [kWp]"); ax2.set_ylabel("Total EAC [€/y]")
        ax2.set_title("Best achievable cost for each size"); ax2.grid(True); ax2.legend()

        ax3 = self.fig.add_subplot(2, 2, 3)
        xs = grid.columns.to_numpy(dtype=float)
        ys = grid.index.to_numpy(dtype=float)
        mesh = ax3.pcolormesh(xs, ys, grid.to_numpy(), shading="nearest", cmap="viridis")
        self.fig.colorbar(mesh, ax=ax3, label="Total EAC [€/y]")
        ax3.plot(base["A_RE"] * e.eta_pv, base["A_RW"] * e.eta_pv, "r*", ms=14)
        ax3.set_xlabel("East roof [kWp]"); ax3.set_ylabel("West roof [kWp]")
        ax3.set_title("East vs west roof (other surfaces at optimum)")

        ax4 = self.fig.add_subplot(2, 2, 4)
        ax4.set_title("Optimal allocation: all profiles (press the button)")
        ax4.axis("off")
        self.fig.tight_layout()
        self.canvas.draw()

    def run_all(self):
        try:
            e = self.parent.read_economics()
            kind = "batt" if self.var_case.get() == "PV + battery" else "pv"
            rows = []
            for prof in PROFILES:
                df = self._df(kind, prof)
                if df is None:
                    continue
                b = optimum_fast(df, kind, e)
                rows.append({"Profile": prof, **{lbl: b[c] * e.eta_pv for c, lbl in SURFACES}})
            if not rows:
                messagebox.showwarning("No results", "No RES files for this case.")
                return
        except Exception as exc:
            messagebox.showerror("Problem", str(exc))
            return

        t = pd.DataFrame(rows).set_index("Profile")
        self.text.delete("1.0", "end")
        self.text.insert("end", "Optimal kWp per surface, all profiles\n\n" + t.round(2).to_string())

        self.fig.clear()
        ax = self.fig.add_subplot(1, 1, 1)
        bottom = np.zeros(len(t))
        for lbl in t.columns:
            ax.bar(np.arange(len(t)), t[lbl].to_numpy(), bottom=bottom, label=lbl)
            bottom += t[lbl].to_numpy()
        ax.set_xticks(np.arange(len(t)))
        ax.set_xticklabels(t.index, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Optimal installed PV [kWp]")
        ax.set_title(f"Optimal PV allocation by surface: {self.var_case.get()}")
        ax.legend(fontsize=8); ax.grid(True, axis="y")
        self.fig.tight_layout()
        self.canvas.draw()



class GraphicsWindow(tk.Toplevel):
    """Report-style graphics: allocation bars, battery landscape, orientation scatter, daily profile."""

    KINDS = ["Allocation: all profiles", "Battery landscape", "Orientation scatter", "Daily PV vs load"]

    def __init__(self, parent: "App"):
        super().__init__(parent)
        self.parent = parent
        self.title("Report graphics")
        self.geometry("1300x900")
        self.pv_csv = None

        ctrl = ttk.Frame(self, padding=8)
        ctrl.pack(side="left", fill="y")
        right = ttk.Frame(self, padding=8)
        right.pack(side="right", fill="both", expand=True)

        self.var_kind = tk.StringVar(value=self.KINDS[0])
        self.var_profile = tk.StringVar(value=parent.var_profile.get())
        self.var_target = tk.StringVar(value="8.0")
        self.var_tol = tk.StringVar(value="0.25")
        self.var_day = tk.StringVar(value="0")
        self.var_load = tk.StringVar(value="Appl")

        ttk.Label(ctrl, text="Graphic").pack(anchor="w")
        ttk.Combobox(ctrl, textvariable=self.var_kind, values=self.KINDS,
                     state="readonly", width=32).pack(anchor="w")
        ttk.Label(ctrl, text="Profile").pack(anchor="w", pady=(6, 0))
        ttk.Combobox(ctrl, textvariable=self.var_profile, values=PROFILES,
                     state="readonly", width=32).pack(anchor="w")

        opts = ttk.LabelFrame(ctrl, text="Options", padding=6)
        opts.pack(anchor="w", pady=8, fill="x")
        for i, (lbl, var) in enumerate([("Orientation: total kWp", self.var_target),
                                        ("  tolerance ± kWp", self.var_tol),
                                        ("Daily: start step (0 = 1 Jan)", self.var_day),
                                        ("Daily: load profile name", self.var_load)]):
            ttk.Label(opts, text=lbl).grid(row=i, column=0, sticky="w")
            ttk.Entry(opts, textvariable=var, width=10).grid(row=i, column=1, padx=4, pady=1)

        ttk.Button(ctrl, text="Draw", command=self.draw).pack(anchor="w", pady=4)
        ttk.Button(ctrl, text="Load PV time series (CSV)...", command=self.pick_csv).pack(anchor="w", pady=2)
        self.lbl_csv = ttk.Label(ctrl, text="No PV time series loaded", wraplength=230)
        self.lbl_csv.pack(anchor="w")

        self.text = tk.Text(right, height=4, width=110, font=("Consolas", 9))
        self.text.pack(fill="x")
        self.fig = Figure(figsize=(11, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, pady=(6, 0))

    def _msg(self, s):
        self.text.delete("1.0", "end")
        self.text.insert("end", s)

    def _df(self, kind, prof):
        folder = self.parent.var_batt.get() if kind == "batt" else self.parent.var_pv.get()
        return _df_or_none(folder, prof, kind)

    def pick_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if path:
            self.pv_csv = path
            self.lbl_csv.configure(text=os.path.basename(path))

    def draw(self):
        try:
            e = self.parent.read_economics()
            kind_name = self.var_kind.get()
            prof = self.var_profile.get()
            if kind_name == "Allocation: all profiles":
                self._allocation(e)
            elif kind_name == "Battery landscape":
                self._landscape(e, prof)
            elif kind_name == "Orientation scatter":
                self._orientation(e, prof)
            else:
                self._daily(e, prof)
        except Exception as exc:
            messagebox.showerror("Problem", str(exc))

    # -------- 1. allocation bars, PV and PV+battery per profile (report style)
    def _allocation(self, e):
        pdict = {p: (self._df("pv", p), self._df("batt", p)) for p in PROFILES}
        t = allocation_table(pdict, e)
        if t.empty:
            messagebox.showwarning("No results", "No RES files found.")
            return
        labels = [f"{p}\n{c}" for p, c in zip(t["Profile"], t["Case"])]
        x = np.arange(len(t))
        self.fig.clear()
        ax = self.fig.add_subplot(1, 1, 1)
        bottom = np.zeros(len(t))
        colors = ["#d9776b", "#f5b97f", "#c4c4c4", "#c99473", "#5f5f5f"]
        for (c, lbl), col in zip(SURFACES, colors):
            ax.bar(x, t[lbl].to_numpy(), bottom=bottom, color=col, edgecolor="k", lw=0.5, label=lbl, width=0.6)
            bottom += t[lbl].to_numpy()
        ax.set_ylabel("PV [kWp]")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.grid(True, axis="y", alpha=0.4)
        ax2 = ax.twinx()
        ax2.bar(x, t["kWh"].to_numpy(), width=0.6, fill=False, linestyle="--", edgecolor="k", lw=1,
                label="E_Batt [kWh]")
        ax2.set_ylabel("Battery [kWh]")
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, ncol=3, loc="upper center")
        self.fig.tight_layout()
        self.canvas.draw()
        self._msg("Optimal PV surfaces (stacked) and battery (dashed, right axis) for each profile.\n"
                  + t.round(2).to_string(index=False))

    # -------- 2. battery landscape: 3-D surface and contour with near-optimal designs
    def _landscape(self, e, prof):
        df = self._df("batt", prof)
        if df is None:
            messagebox.showwarning("No results", "No battery RES file for this profile.")
            return
        kwh_v, kwp_v, Z = landscape_grid(df, e)
        Z = np.where(np.isfinite(Z), Z, np.nanmin(Z))
        KK, BB = np.meshgrid(kwp_v, kwh_v, indexing="ij")
        i, j = np.unravel_index(np.nanargmax(Z), Z.shape)
        best = Z[i, j]
        tol = 0.01 * abs(best)

        self.fig.clear()
        ax1 = self.fig.add_subplot(2, 1, 1, projection="3d")
        ax1.plot_surface(BB, KK, Z / 1000, cmap="viridis", alpha=0.85, edgecolor="none")
        ax1.scatter([kwh_v[j]], [kwp_v[i]], [best / 1000], color="r", s=120, marker="*")
        ax1.set_xlabel("Battery [kWh]"); ax1.set_ylabel("PV [kWp]"); ax1.set_zlabel("EAC [k€/y]")
        ax1.set_title(f"{prof}: EAC landscape, optimum {kwp_v[i]:.1f} kWp / {kwh_v[j]:g} kWh")

        ax2 = self.fig.add_subplot(2, 1, 2)
        cs = ax2.contourf(BB, KK, Z / 1000, 20, cmap="viridis")
        self.fig.colorbar(cs, ax=ax2, label="EAC [k€/y]")
        near = Z > best - tol
        ax2.plot(BB[near], KK[near], "w.", ms=6)
        ax2.plot(kwh_v[j], kwp_v[i], "r*", ms=16)
        ax2.set_xlabel("Battery [kWh]"); ax2.set_ylabel("PV [kWp]")
        ax2.set_title(f"Contour; white dots within 1% of the optimum ({tol:.0f} €/y)")
        self.fig.tight_layout()
        self.canvas.draw()
        self._msg(f"{prof}: optimum {kwp_v[i]:.2f} kWp, {kwh_v[j]:g} kWh, "
                  f"EAC {best:,.0f} €/y. Grid: {len(kwp_v)} PV sizes × {len(kwh_v)} battery sizes.")

    # -------- 3. orientation scatter (like the 8 kWp plot): KPIs vs roof O/W balance
    def _orientation(self, e, prof):
        df = self._df("pv", prof)
        if df is None:
            messagebox.showwarning("No results", "No PV-only RES file for this profile.")
            return
        target = float(self.var_target.get().replace(",", "."))
        tol = float(self.var_tol.get().replace(",", "."))
        t = orientation_scatter_table(df, e, target, tol)
        if t.empty:
            messagebox.showwarning("No designs", "No designs near this size. Increase the tolerance.")
            return
        cols = ["PV yield [kWh/y]", "Self-consumption [%]", "PV to grid [kWh/y]",
                "Grid demand [kWh/y]", "EAC [€/y]"]
        self.fig.clear()
        for k, col in enumerate(cols):
            ax = self.fig.add_subplot(len(cols), 1, k + 1)
            sc = ax.scatter(t["OW"], t[col], c=100 * t["Balcony share"], cmap="viridis", s=22)
            ax.set_ylabel(col, fontsize=8); ax.grid(True, alpha=0.4)
            self.fig.colorbar(sc, ax=ax, label="Balcony share [%]")
            if k == len(cols) - 1:
                ax.set_xlabel("Roof O/W  (−1 = west, +1 = east)")
        self.fig.suptitle(f"{prof}: designs near {target:g} kWp (±{tol:g}), n = {len(t)}")
        self.fig.tight_layout()
        self.canvas.draw()
        self._msg(f"Roof-only orientation index; balcony share colour. Rows with no roof are excluded.")

    # -------- 4. daily PV vs load (needs the PV time series exported from MATLAB)
    def _daily(self, e, prof):
        if self.pv_csv is None:
            messagebox.showinfo("Need PV time series",
                                "Load the PV time series CSV first (see the MATLAB export line).")
            return
        df = self._df("pv", prof)
        if df is None:
            messagebox.showwarning("No results", "No PV-only RES file for this profile.")
            return
        pv_series = np.loadtxt(self.pv_csv, delimiter=",", ndmin=2)
        load_path = os.path.join(os.path.dirname(self.parent.var_pv.get()), "Appl.mat")
        if not os.path.exists(load_path):
            load_path = filedialog.askopenfilename(title="Locate Appl.mat", filetypes=[("MAT", "*.mat")])
            if not load_path:
                return
        load = load_appl_profile(load_path, self.var_load.get())
        b = optimum_fast(df, "pv", e)
        areas = {"S": b["A_S"], "RE": b["A_RE"], "RW": b["A_RW"], "RfE": b["A_ReF"], "RfW": b["A_RwF"]}
        start = int(self.var_day.get())
        n = 2 * 96  # two days of 15-minute steps
        d = daily_profile(pv_series, areas, load, start, n)
        hours = np.arange(n) * 0.25

        self.fig.clear()
        ax = self.fig.add_subplot(1, 1, 1)
        ax.plot(hours, d["pv_kw"], label=f"PV {b['kWp']:.2f} kWp (optimum)")
        ax.plot(hours, d["load_kw"], "k--", label=f"Load {self.var_load.get()}")
        ax.set_xlabel("Time [h from start]"); ax.set_ylabel("Power [kW]")
        ax.set_title(f"{prof}: PV vs household load, two days")
        ax.grid(True); ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()
        self._msg(f"Start step {start}. Optimum PV {b['kWp']:.2f} kWp, areas m²: " +
                  ", ".join(f"{k} {v:.1f}" for k, v in areas.items()))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PV + battery sizing")
        self.geometry("1350x820")
        self.vars = {}
        self.defaults = {}
        self.sens_window = None

        side = ttk.Frame(self, padding=8)
        side.pack(side="left", fill="y")
        main = ttk.Frame(self, padding=8)
        main.pack(side="right", fill="both", expand=True)

        ttk.Label(side, text="Data", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.var_pv = tk.StringVar(value=os.path.abspath("RES_profiles"))
        self.var_batt = tk.StringVar(value=os.path.abspath("RES_profiles_battery"))
        self._folder_row(side, "PV-only folder", self.var_pv)
        self._folder_row(side, "PV+battery folder", self.var_batt)

        self.var_profile = tk.StringVar(value=PROFILES[0])
        ttk.Label(side, text="Load profile").pack(anchor="w", pady=(6, 0))
        ttk.Combobox(side, textvariable=self.var_profile, values=PROFILES,
                     state="readonly", width=38).pack(anchor="w")

        ttk.Label(side, text="Economics", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 0))
        self._fields(side, [
            ("cost_cur", "Electricity price [€/kWh]", 0.33),
            ("gain_sell", "Feed-in tariff [€/kWh]", 0.05),
            ("r", "Discount rate [%]", 3.5),
            ("eta_pv", "Module efficiency [kWp/m²]", 0.23),
            ("inv_pv", "PV investment [€/kWp]", 1500),
            ("life_pv", "PV lifetime [y]", 20),
            ("inv_batt", "Battery investment [€/kWh]", 500),
            ("life_batt", "Battery lifetime [y]", 12),
        ])

        ttk.Label(side, text="Subsidies", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 0))
        self._fields(side, [
            ("land_pv_rate", "State PV rate [€/kWp]", 125),
            ("land_pv_cap", "State PV cap [kWp]", 20),
            ("land_batt_rate", "State battery [€/kWh]", 0),
            ("bund_batt_rate", "Federal battery [€/kWh]", 150),
            ("bund_batt_cap", "Federal battery cap [kWh]", 50),
        ])

        ttk.Label(side, text="Federal PV rule").pack(anchor="w", pady=(6, 0))
        self.var_mode = tk.StringVar(value=DEFAULT_MODE)
        ttk.Combobox(side, textvariable=self.var_mode, values=["whole", "marginal"],
                     state="readonly", width=38).pack(anchor="w")

        ttk.Label(side, text="Federal PV bands: upper bound [kWp] / rate [€/kWp]").pack(anchor="w", pady=(6, 0))
        tiers_frame = ttk.Frame(side)
        tiers_frame.pack(anchor="w")
        self.tier_vars = []
        for i, (u, rt) in enumerate(DEFAULT_TIERS):
            vu, vr = tk.StringVar(value=str(u)), tk.StringVar(value=str(rt))
            ttk.Entry(tiers_frame, textvariable=vu, width=10).grid(row=i, column=0, padx=2, pady=1)
            ttk.Entry(tiers_frame, textvariable=vr, width=10).grid(row=i, column=1, padx=2, pady=1)
            self.tier_vars.append((vu, vr))

        btns = ttk.Frame(side)
        btns.pack(anchor="w", pady=(12, 0))
        ttk.Button(btns, text="Run selected profile", command=self.on_run).pack(anchor="w", pady=2)
        ttk.Button(btns, text="Reset inputs to original", command=self.reset_inputs).pack(anchor="w", pady=2)
        ttk.Button(btns, text="Sensitivity analysis...", command=self.open_sensitivity).pack(anchor="w", pady=2)
        ttk.Button(btns, text="Full report: all profiles", command=self.on_report).pack(anchor="w", pady=2)
        ttk.Button(btns, text="PV orientation...", command=self.open_orientation).pack(anchor="w", pady=2)
        ttk.Button(btns, text="Report graphics...", command=self.open_graphics).pack(anchor="w", pady=2)

        self.text = tk.Text(main, height=12, width=100, font=("Consolas", 10))
        self.text.pack(fill="x")
        self.fig = Figure(figsize=(11, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=main)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, pady=(8, 0))
        self.write("Choose a profile and press 'Run selected profile'.")

    # ---------------------------------------------------------- helpers
    def _folder_row(self, parent, label, var):
        ttk.Label(parent, text=label).pack(anchor="w", pady=(4, 0))
        row = ttk.Frame(parent)
        row.pack(anchor="w")
        ttk.Entry(row, textvariable=var, width=34).pack(side="left")
        ttk.Button(row, text="...", width=3,
                   command=lambda: var.set(filedialog.askdirectory() or var.get())).pack(side="left", padx=2)

    def _fields(self, parent, fields):
        grid = ttk.Frame(parent)
        grid.pack(anchor="w")
        for i, (key, label, default) in enumerate(fields):
            ttk.Label(grid, text=label).grid(row=i, column=0, sticky="w", padx=2, pady=1)
            var = tk.StringVar(value=str(default))
            ttk.Entry(grid, textvariable=var, width=12).grid(row=i, column=1, padx=2, pady=1)
            self.vars[key] = var
            self.defaults[key] = default

    def reset_inputs(self):
        """Restore every economics and subsidy input, the federal PV rule, the bands and the profile
        to their original values. Data folders are kept, since they point to where the files are."""
        for key, value in self.defaults.items():
            self.vars[key].set(str(value))
        self.var_mode.set(DEFAULT_MODE)
        for (vu, vr), (u, rt) in zip(self.tier_vars, DEFAULT_TIERS):
            vu.set(str(u)); vr.set(str(rt))
        self.var_profile.set(PROFILES[0])
        self.write("Inputs reset to the original values. Press 'Run selected profile' to recalculate.")

    def write(self, msg):
        self.text.delete("1.0", "end")
        self.text.insert("end", msg)

    def read_economics(self):
        def num(key):
            try:
                return float(self.vars[key].get().replace(",", "."))
            except ValueError:
                raise ValueError(f"'{key}' must be a number")
        upper, rates = [], []
        for vu, vr in self.tier_vars:
            try:
                upper.append(float(vu.get())); rates.append(float(vr.get()))
            except ValueError:
                raise ValueError("Federal band entries must be numbers")
        if np.any(np.diff(upper) <= 0):
            raise ValueError("Federal band upper bounds must be increasing")
        return Economics(
            cost_cur=num("cost_cur"), gain_sell=num("gain_sell"),
            r=num("r") / 100, eta_pv=num("eta_pv"),
            inv_pv=num("inv_pv"), life_pv=int(num("life_pv")),
            inv_batt=num("inv_batt"), life_batt=int(num("life_batt")),
            land_pv_rate=num("land_pv_rate"), land_pv_cap_kwp=num("land_pv_cap"),
            land_batt_rate=num("land_batt_rate"), bund_batt_rate=num("bund_batt_rate"),
            bund_batt_cap_kwh=num("bund_batt_cap"),
            bund_pv_upper=tuple(upper), bund_pv_rates=tuple(rates),
            bund_pv_mode=self.var_mode.get(),
        )

    # ---------------------------------------------------------- actions
    def on_run(self):
        try:
            e = self.read_economics()
            prof = self.var_profile.get()
            pv_out, batt_out = run_profile(prof, self.var_pv.get(), self.var_batt.get(), e)
        except Exception as exc:
            messagebox.showerror("Problem", str(exc))
            return
        if pv_out is None and batt_out is None:
            messagebox.showwarning("No results", f"No RES file found for {prof} in these folders.")
            return

        lines = [f"Profile: {prof}    Federal PV rule: {e.bund_pv_mode}", ""]
        for label, out in [("PV only", pv_out), ("PV + battery", batt_out)]:
            if out is None:
                continue
            b = optimum(out)
            lines.append(f"{label}: PV {b['kWp']:.2f} kWp | battery {b['kWh']:.1f} kWh | "
                         f"EAC {b['EAC_total']:,.0f} €/y | grid {b['GridDem_kWh']:,.0f} kWh/y | "
                         f"PV subsidy {b['Incentive_PV']:,.0f} € | battery subsidy {b['Incentive_Batt']:,.0f} €")
            note = threshold_note(b["kWp"], e)
            if note:
                lines.append("  WARNING: " + note)
        if pv_out is not None and batt_out is not None:
            gain = optimum(batt_out)["EAC_total"] - optimum(pv_out)["EAC_total"]
            lines.append(f"Gain from adding a battery: {gain:,.0f} €/y")
        self.write("\n".join(lines))
        self.plot(prof, pv_out, batt_out, e)

    def plot(self, prof, pv_out, batt_out, e):
        self.fig.clear()
        ax1 = self.fig.add_subplot(1, 2, 1)
        if pv_out is not None:
            env = envelope(pv_out, by_battery=False)
            ax1.plot(env["kWp_r"], env["EAC_total"], label="PV only", lw=1.6)
        if batt_out is not None:
            envb = envelope(batt_out, by_battery=True)
            for kwh, g in envb.groupby("kWh"):
                if kwh == 0:
                    continue
                ax1.plot(g["kWp_r"], g["EAC_total"], lw=1, alpha=0.8, label=f"{kwh:g} kWh")
        ax1.set_xlabel("Installed PV [kWp]"); ax1.set_ylabel("Total EAC [€/y]")
        ax1.set_title(f"EAC vs PV size: {prof}"); ax1.grid(True); ax1.legend(fontsize=7)

        ax2 = self.fig.add_subplot(1, 2, 2)
        x = np.linspace(0, 40, 801)
        other = "marginal" if e.bund_pv_mode == "whole" else "whole"
        e_other = replace(e, bund_pv_mode=other)
        ax2.plot(x, x * e.inv_pv - pv_incentive(x, e), label=f"Net PV investment ({e.bund_pv_mode})")
        ax2.plot(x, x * e.inv_pv - pv_incentive(x, e_other), "--", label=f"Other rule ({other})")
        ax2.set_xlabel("Installed PV [kWp]"); ax2.set_ylabel("Net investment [€]")
        ax2.set_title("Net PV investment after subsidies"); ax2.grid(True); ax2.legend(fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw()

    def open_sensitivity(self):
        try:
            self.read_economics()  # make sure inputs are valid before opening
        except Exception as exc:
            messagebox.showerror("Problem", str(exc))
            return
        if self.sens_window is None or not self.sens_window.winfo_exists():
            self.sens_window = SensitivityWindow(self)
        else:
            self.sens_window.lift()

    def open_orientation(self):
        try:
            self.read_economics()
        except Exception as exc:
            messagebox.showerror("Problem", str(exc))
            return
        if getattr(self, "orient_window", None) is None or not self.orient_window.winfo_exists():
            self.orient_window = OrientationWindow(self)
        else:
            self.orient_window.lift()

    def open_graphics(self):
        try:
            self.read_economics()
        except Exception as exc:
            messagebox.showerror("Problem", str(exc))
            return
        if getattr(self, "graph_window", None) is None or not self.graph_window.winfo_exists():
            self.graph_window = GraphicsWindow(self)
        else:
            self.graph_window.lift()

    def on_report(self):
        try:
            e = self.read_economics()
            report = full_report_all(self.var_pv.get(), self.var_batt.get(), e)
        except Exception as exc:
            messagebox.showerror("Problem", str(exc))
            return
        if report.empty:
            messagebox.showwarning("No results", "No RES files found in these folders.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                            initialfile="full_report_python.csv")
        if path:
            report.to_csv(path, index=False)
        show = report[["Profile", "Case", "PV [kWp]", "Battery [kWh]", "Total EAC [€/y]",
                       "Gain vs without PV/battery [€/y]", "PV subsidy [€]", "Battery subsidy [€]"]]
        self.write("Full report: optimum for every profile and case\n\n" + show.round(1).to_string(index=False))
        self.fig.clear(); self.canvas.draw()


if __name__ == "__main__":
    App().mainloop()
