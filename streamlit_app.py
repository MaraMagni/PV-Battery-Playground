"""PV + battery sizing: web version of the desktop app.

Same layout as the desktop app: inputs and actions in the sidebar, results in
the main area, and the other windows as tabs. The results are read from the
RES_profiles_small and RES_profiles_battery_small folders in this repository.
"""
import io
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the 3-D projection)

from econ_core import (PROFILES, SURFACES, SWEEP_PARAMS, Economics, allocation_table,
                       best_mix_vs_size, daily_profile, envelope, evaluate, full_report,
                       landscape_grid, load_appl_profile, load_res, optimum, optimum_fast,
                       orientation_scatter_table, pv_incentive, roof_ew_grid,
                       sensitivity_1d, threshold_note, tornado)

st.set_page_config(page_title="PV + battery sizing", layout="wide")

HERE = Path(__file__).parent
PV_DIR = HERE / "RES_profiles_small"
BATT_DIR = HERE / "RES_profiles_battery_small"

DEFAULT_TIERS = [(10, 150), (20, 140), (100, 130), (1000, 120)]
DEFAULTS = {
    "cost_cur": 0.33, "gain_sell": 0.05, "r": 3.5, "eta_pv": 0.23,
    "inv_pv": 1500.0, "life_pv": 20, "inv_batt": 500.0, "life_batt": 12,
    "land_pv_rate": 125.0, "land_pv_cap": 20.0, "land_batt_rate": 0.0,
    "bund_batt_rate": 150.0, "bund_batt_cap": 50.0, "mode": "whole",
}
for i, (u, rt) in enumerate(DEFAULT_TIERS):
    DEFAULTS[f"up{i}"] = float(u)
    DEFAULTS[f"rt{i}"] = float(rt)
for _key, _value in DEFAULTS.items():
    st.session_state.setdefault(_key, _value)


def reset_inputs():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


@st.cache_data(show_spinner=False, max_entries=4)
def load_case(path: str, kind: str, mtime: float) -> pd.DataFrame:
    """Load one RES file. The mtime argument refreshes the cache when a file changes."""
    return load_res(path, kind)


@st.cache_data(show_spinner=False)
def load_appl(data: bytes, name: str) -> np.ndarray:
    fd, path = tempfile.mkstemp(suffix=".mat")
    with open(fd, "wb") as f:
        f.write(data)
    try:
        return load_appl_profile(path, name)
    finally:
        Path(path).unlink(missing_ok=True)


def get_df(folder: Path, prof: str, kind: str):
    """Results for one profile and case, or None if the file is not in the folder."""
    path = folder / f"RES_{prof}.mat"
    if not path.exists():
        return None
    return load_case(str(path), kind, path.stat().st_mtime)


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Data")
    st.caption("Results come from RES_profiles_small and RES_profiles_battery_small in the repository.")
    available = [p for p in PROFILES
                 if (PV_DIR / f"RES_{p}.mat").exists() or (BATT_DIR / f"RES_{p}.mat").exists()]
    if not available:
        st.error("No RES files found in the data folders.")
        st.stop()
    profile = st.selectbox("Load profile", available, key="profile")

    st.header("Economics")
    st.number_input("Electricity price [€/kWh]", step=0.01, key="cost_cur")
    st.number_input("Feed-in tariff [€/kWh]", step=0.01, key="gain_sell")
    st.number_input("Discount rate [%]", step=0.1, key="r")
    st.number_input("Module efficiency [kWp/m²]", step=0.01, key="eta_pv")
    st.number_input("PV investment [€/kWp]", step=50.0, key="inv_pv")
    st.number_input("PV lifetime [y]", step=1, key="life_pv")
    st.number_input("Battery investment [€/kWh]", step=25.0, key="inv_batt")
    st.number_input("Battery lifetime [y]", step=1, key="life_batt")

    st.header("Subsidies")
    st.number_input("State PV rate [€/kWp]", step=5.0, key="land_pv_rate")
    st.number_input("State PV cap [kWp]", step=1.0, key="land_pv_cap")
    st.number_input("State battery [€/kWh]", step=10.0, key="land_batt_rate")
    st.number_input("Federal battery [€/kWh]", step=10.0, key="bund_batt_rate")
    st.number_input("Federal battery cap [kWh]", step=5.0, key="bund_batt_cap")
    st.selectbox("Federal PV rule", ["whole", "marginal"], key="mode")
    st.caption("Federal PV bands: upper bound [kWp] / rate [€/kWp]")
    for i in range(len(DEFAULT_TIERS)):
        c1, c2 = st.columns(2)
        c1.number_input(f"Upper {i + 1} [kWp]", key=f"up{i}")
        c2.number_input(f"Rate {i + 1} [€/kWp]", key=f"rt{i}")

    st.header("Actions")
    run_main = st.button("Run selected profile", type="primary")
    st.button("Reset inputs to original", on_click=reset_inputs)
    run_report = st.button("Full report: all profiles")

ss = st.session_state
uppers = [ss[f"up{i}"] for i in range(len(DEFAULT_TIERS))]
rates = [ss[f"rt{i}"] for i in range(len(DEFAULT_TIERS))]
if np.any(np.diff(uppers) <= 0):
    st.error("Federal band upper bounds must be increasing.")
    st.stop()

e = Economics(
    cost_cur=ss["cost_cur"], gain_sell=ss["gain_sell"], r=ss["r"] / 100, eta_pv=ss["eta_pv"],
    inv_pv=ss["inv_pv"], life_pv=int(ss["life_pv"]), inv_batt=ss["inv_batt"],
    life_batt=int(ss["life_batt"]),
    land_pv_rate=ss["land_pv_rate"], land_pv_cap_kwp=ss["land_pv_cap"],
    land_batt_rate=ss["land_batt_rate"], bund_batt_rate=ss["bund_batt_rate"],
    bund_batt_cap_kwh=ss["bund_batt_cap"],
    bund_pv_upper=tuple(uppers), bund_pv_rates=tuple(rates), bund_pv_mode=ss["mode"],
)


# ---------------------------------------------------------------- main window
def render_main(e, prof):
    pv_df = get_df(PV_DIR, prof, "pv")
    bt_df = get_df(BATT_DIR, prof, "batt")
    lines = [f"Profile: {prof}    Federal PV rule: {e.bund_pv_mode}", ""]
    outs, best = {}, {}
    for label, kind, df in (("PV only", "pv", pv_df), ("PV + battery", "batt", bt_df)):
        if df is None:
            continue
        outs[kind] = evaluate(df, kind, e)
        b = optimum_fast(df, kind, e)
        best[kind] = b
        lines.append(f"{label}: PV {b['kWp']:.2f} kWp | battery {b['kWh']:.1f} kWh | "
                     f"EAC {b['EAC_total']:,.0f} €/y | grid {b['GridDem_kWh']:,.0f} kWh/y | "
                     f"PV subsidy {b['Incentive_PV']:,.0f} € | battery subsidy {b['Incentive_Batt']:,.0f} €")
        split = ", ".join(f"{lbl} {b[c] * e.eta_pv:.2f} kWp" for c, lbl in SURFACES)
        roof = b["A_RE"] + b["A_RW"]
        ow = (b["A_RE"] - b["A_RW"]) / roof if roof > 0 else float("nan")
        lines.append(f"  Orientation: {split} | roof O/W index {ow:+.2f}")
        note = threshold_note(b["kWp"], e)
        if note:
            lines.append("  WARNING: " + note)
    if not best:
        st.warning(f"No RES file found for {prof}.")
        return
    if "pv" in best and "batt" in best:
        gain = best["batt"]["EAC_total"] - best["pv"]["EAC_total"]
        lines.append(f"Gain from adding a battery: {gain:,.0f} €/y")
    st.code("\n".join(lines))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    if "pv" in outs:
        env = envelope(outs["pv"], by_battery=False)
        ax1.plot(env["kWp_r"], env["EAC_total"], lw=1.6, label="PV only")
    if "batt" in outs:
        envb = envelope(outs["batt"], by_battery=True)
        for kwh, g in envb.groupby("kWh"):
            if kwh == 0:
                continue
            ax1.plot(g["kWp_r"], g["EAC_total"], lw=1, alpha=0.8, label=f"{kwh:g} kWh")
    ax1.set_xlabel("Installed PV [kWp]")
    ax1.set_ylabel("Total EAC [€/y]")
    ax1.set_title(f"EAC vs PV size: {prof}")
    ax1.grid(True)
    ax1.legend(fontsize=7)

    x = np.linspace(0, 40, 801)
    other = "marginal" if e.bund_pv_mode == "whole" else "whole"
    e_other = replace(e, bund_pv_mode=other)
    ax2.plot(x, x * e.inv_pv - pv_incentive(x, e), label=f"Net PV investment ({e.bund_pv_mode})")
    ax2.plot(x, x * e.inv_pv - pv_incentive(x, e_other), "--", label=f"Other rule ({other})")
    ax2.set_xlabel("Installed PV [kWp]")
    ax2.set_ylabel("Net investment [€]")
    ax2.set_title("Net PV investment after subsidies")
    ax2.grid(True)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    show_fig(fig)


def render_report(e):
    rows = []
    bar = st.progress(0.0, text="Building the full report")
    for k, p in enumerate(available):
        rows.append(full_report(p, get_df(PV_DIR, p, "pv"), get_df(BATT_DIR, p, "batt"), e))
        bar.progress((k + 1) / len(available))
    report = pd.concat(rows, ignore_index=True)
    cols = ["Profile", "Case", "PV [kWp]", "Battery [kWh]", "Total EAC [€/y]",
            "Gain vs without PV/battery [€/y]", "PV subsidy [€]", "Battery subsidy [€]"]
    st.dataframe(report[cols].round(1), hide_index=True)
    st.download_button("Save full report as CSV", report.to_csv(index=False),
                       file_name="full_report_python.csv", mime="text/csv")


# ---------------------------------------------------------------- orientation helpers
def render_orientation(df, kind, prof, case, bin_kwp, e):
    base = optimum_fast(df, kind, e)
    mix = best_mix_vs_size(df, kind, e, bin_kwp)
    grid = roof_ew_grid(df, kind, e, base)
    areas = {lbl: base[c] * e.eta_pv for c, lbl in SURFACES}
    st.code(f"{prof} | {case} | optimum {base['kWp']:.2f} kWp, battery {base['kWh']:.1f} kWh, "
            f"EAC {base['EAC_total']:,.0f} €/y\n"
            "Optimal kWp per surface: " + ", ".join(f"{k} {v:.2f}" for k, v in areas.items()))

    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    ax1, ax2, ax3, ax4 = axs.ravel()
    ax1.stackplot(mix["kWp"], [mix[c] for c, _ in SURFACES],
                  labels=[lbl for _, lbl in SURFACES], alpha=0.85)
    ax1.axvline(base["kWp"], color="k", ls="--", lw=0.8)
    ax1.set_xlabel("Total installed PV [kWp]")
    ax1.set_ylabel("kWp per surface")
    ax1.set_title("Best surface allocation for each total size")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(True)

    ax2.plot(mix["kWp"], mix["EAC_total"], "o-", ms=3, color="tab:green")
    ax2.plot(base["kWp"], base["EAC_total"], "r*", ms=14, label="Optimum")
    ax2.set_xlabel("Total installed PV [kWp]")
    ax2.set_ylabel("Total EAC [€/y]")
    ax2.set_title("Best achievable cost for each size")
    ax2.grid(True)
    ax2.legend()

    xs = grid.columns.to_numpy(dtype=float)
    ys = grid.index.to_numpy(dtype=float)
    mesh = ax3.pcolormesh(xs, ys, grid.to_numpy(), shading="nearest", cmap="viridis")
    fig.colorbar(mesh, ax=ax3, label="Total EAC [€/y]")
    ax3.plot(base["A_RE"] * e.eta_pv, base["A_RW"] * e.eta_pv, "r*", ms=14)
    ax3.set_xlabel("East roof [kWp]")
    ax3.set_ylabel("West roof [kWp]")
    ax3.set_title("East vs west roof (other surfaces at optimum)")

    ax4.set_title("Optimal allocation: all profiles (see the button in the controls)")
    ax4.axis("off")
    fig.tight_layout()
    show_fig(fig)


def render_all_allocation(kind, case, e):
    rows = []
    for p in available:
        df = get_df(BATT_DIR if kind == "batt" else PV_DIR, p, kind)
        if df is None:
            continue
        b = optimum_fast(df, kind, e)
        rows.append({"Profile": p, **{lbl: b[c] * e.eta_pv for c, lbl in SURFACES}})
    if not rows:
        st.warning("No RES files for this case.")
        return
    t = pd.DataFrame(rows).set_index("Profile")
    st.text(f"Optimal kWp per surface, all profiles ({case})")
    st.dataframe(t.round(2))
    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(len(t))
    for lbl in t.columns:
        ax.bar(np.arange(len(t)), t[lbl].to_numpy(), bottom=bottom, label=lbl)
        bottom += t[lbl].to_numpy()
    ax.set_xticks(np.arange(len(t)))
    ax.set_xticklabels(t.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Optimal installed PV [kWp]")
    ax.set_title(f"Optimal PV allocation by surface: {case}")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y")
    fig.tight_layout()
    show_fig(fig)


# ---------------------------------------------------------------- report graphics
def render_graphic(kind_name, prof, e, target, tol):
    if kind_name == "Allocation: all profiles":
        parts = [allocation_table({p: (get_df(PV_DIR, p, "pv"), get_df(BATT_DIR, p, "batt"))}, e)
                 for p in available]
        t = pd.concat(parts, ignore_index=True)
        if t.empty:
            st.warning("No RES files found.")
            return
        x = np.arange(len(t))
        profs = list(t["Profile"])
        fig, ax = plt.subplots(figsize=(12, 6))
        bottom = np.zeros(len(t))
        colors = ["#d9776b", "#f5b97f", "#c4c4c4", "#c99473", "#5f5f5f"]
        for (c, lbl), col in zip(SURFACES, colors):
            ax.bar(x, t[lbl].to_numpy(), bottom=bottom, color=col, edgecolor="k", lw=0.5,
                   label=lbl, width=0.6)
            bottom += t[lbl].to_numpy()
        ax.set_ylabel("PV [kWp]")
        ax.set_xticks(x)
        ax.set_xticklabels(list(t["Case"]), fontsize=8)
        # Profile name underneath each pair of bars, with a separator between profiles
        import textwrap
        starts = [i for i, p in enumerate(profs) if i == 0 or profs[i - 1] != p]
        ends = starts[1:] + [len(profs)]
        for s, en in zip(starts, ends):
            name = textwrap.fill(profs[s].replace("_", " "), 16)
            ax.text((s + en - 1) / 2, -0.2, name, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8)
        for s in starts[1:]:
            ax.plot([s - 0.5, s - 0.5], [0, -0.22], color="k", lw=0.8,
                    transform=ax.get_xaxis_transform(), clip_on=False)
        ax.grid(True, axis="y", alpha=0.4)
        ax2 = ax.twinx()
        ax2.bar(x, t["kWh"].to_numpy(), width=0.6, fill=False, linestyle="--", edgecolor="k",
                lw=1, label="E_Batt [kWh]")
        ax2.set_ylabel("Battery [kWh]")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, ncol=3, loc="upper center")
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.28)
        show_fig(fig)
        st.dataframe(t.round(2), hide_index=True)

    elif kind_name == "Battery landscape":
        df = get_df(BATT_DIR, prof, "batt")
        if df is None:
            st.warning("No battery RES file for this profile.")
            return
        kwh_v, kwp_v, Z = landscape_grid(df, e)
        Z = np.where(np.isfinite(Z), Z, np.nanmin(Z))
        KK, BB = np.meshgrid(kwp_v, kwh_v, indexing="ij")
        i, j = np.unravel_index(np.nanargmax(Z), Z.shape)
        best = Z[i, j]
        tol_eur = 0.01 * abs(best)
        fig = plt.figure(figsize=(10, 11))
        ax1 = fig.add_subplot(2, 1, 1, projection="3d")
        ax1.plot_surface(BB, KK, Z / 1000, cmap="viridis", alpha=0.85, edgecolor="none")
        ax1.scatter([kwh_v[j]], [kwp_v[i]], [best / 1000], color="r", s=120, marker="*")
        ax1.set_xlabel("Battery [kWh]")
        ax1.set_ylabel("PV [kWp]")
        ax1.set_zlabel("EAC [k€/y]")
        ax1.set_title(f"{prof}: EAC landscape, optimum {kwp_v[i]:.1f} kWp / {kwh_v[j]:g} kWh")
        ax2 = fig.add_subplot(2, 1, 2)
        cs = ax2.contourf(BB, KK, Z / 1000, 20, cmap="viridis")
        fig.colorbar(cs, ax=ax2, label="EAC [k€/y]")
        near = Z > best - tol_eur
        ax2.plot(BB[near], KK[near], "w.", ms=6)
        ax2.plot(kwh_v[j], kwp_v[i], "r*", ms=16)
        ax2.set_xlabel("Battery [kWh]")
        ax2.set_ylabel("PV [kWp]")
        ax2.set_title(f"Contour; white dots within 1% of the optimum ({tol_eur:.0f} €/y)")
        fig.tight_layout()
        show_fig(fig)
        st.code(f"{prof}: optimum {kwp_v[i]:.2f} kWp, {kwh_v[j]:g} kWh, EAC {best:,.0f} €/y. "
                f"Grid: {len(kwp_v)} PV sizes × {len(kwh_v)} battery sizes.")

    elif kind_name == "Orientation scatter":
        df = get_df(PV_DIR, prof, "pv")
        if df is None:
            st.warning("No PV-only RES file for this profile.")
            return
        t = orientation_scatter_table(df, e, target, tol)
        if t.empty:
            st.warning("No designs near this size. Increase the tolerance.")
            return
        cols = ["PV yield [kWh/y]", "Self-consumption [%]", "PV to grid [kWh/y]",
                "Grid demand [kWh/y]", "EAC [€/y]"]
        fig, axes = plt.subplots(len(cols), 1, figsize=(10, 14), sharex=True)
        for ax, col in zip(axes, cols):
            sc = ax.scatter(t["OW"], t[col], c=100 * t["Balcony share"], cmap="viridis", s=22)
            ax.set_ylabel(col, fontsize=8)
            ax.grid(True, alpha=0.4)
            fig.colorbar(sc, ax=ax, label="Balcony share [%]")
        axes[-1].set_xlabel("Roof O/W  (−1 = west, +1 = east)")
        fig.suptitle(f"{prof}: designs near {target:g} kWp (±{tol:g}), n = {len(t)}")
        fig.tight_layout()
        show_fig(fig)
        st.code("Roof-only orientation index; balcony share colour. Rows with no roof are excluded.")


# ---------------------------------------------------------------- tabs
tab_main, tab_sens, tab_orient, tab_graph = st.tabs(
    ["Results", "Sensitivity", "PV orientation", "Report graphics"])

with tab_main:
    if run_main:
        st.session_state["main_prof"] = profile
    if st.session_state.get("main_prof"):
        render_main(e, st.session_state["main_prof"])
    if run_report:
        render_report(e)

with tab_sens:
    ctrl, out = st.columns([1, 3])
    with ctrl:
        sprof = st.selectbox("Profile", available, key="sens_profile")
        case = st.radio("Case", ["PV only", "PV + battery"], key="sens_case")
        param = st.selectbox("One-parameter sweep", [s[0] for s in SWEEP_PARAMS], key="sens_param")
        _, field, scale, batt_only = next(s for s in SWEEP_PARAMS if s[0] == param)
        base = getattr(e, field) * scale
        lo = st.number_input("From", value=float(round(base * 0.5, 4)), key=f"lo_{field}")
        hi = st.number_input("To", value=float(round(base * 1.5, 4)), key=f"hi_{field}")
        steps = st.number_input("Steps", value=9, min_value=2, step=1, key="sens_steps")
        run_sweep = st.button("Run sweep")
        st.divider()
        pct = st.number_input("Change ±%", value=20.0, step=5.0, key="sens_pct")
        run_tornado = st.button("Run tornado (both cases)")

    with out:
        kind = "batt" if case == "PV + battery" else "pv"
        if run_sweep:
            df = get_df(BATT_DIR if kind == "batt" else PV_DIR, sprof, kind)
            if df is None:
                st.warning("No RES file for this profile and case.")
            elif batt_only and kind == "pv":
                st.info("This parameter only affects the PV + battery case. Choose that case.")
            else:
                table = sensitivity_1d(df, kind, e, field,
                                       np.linspace(lo / scale, hi / scale, int(steps)))
                table.insert(0, param, table["value"] * scale)
                for c, lbl in SURFACES:  # kWp per surface of each optimum
                    table[f"{lbl} [kWp]"] = table[c] * e.eta_pv
                surf_cols = [f"{lbl} [kWp]" for _, lbl in SURFACES]
                show = table[[param, "kWp", "kWh", "EAC_total", "GridDem_kWh",
                              "Incentive_PV", "Incentive_Batt"] + surf_cols]
                st.text(f"{sprof} | {case} | sweep of {param}")
                st.dataframe(show.round(2), hide_index=True)
                st.download_button("Save table as CSV",
                                   table.assign(Profile=sprof, Case=case).to_csv(index=False),
                                   file_name="sensitivity.csv", mime="text/csv")
                fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
                a1.plot(table[param], table["kWp"], "o-", label="PV [kWp]")
                if kind == "batt":
                    a1.plot(table[param], table["kWh"], "s--", label="Battery [kWh]")
                a1.set_xlabel(param)
                a1.set_ylabel("Optimal size")
                a1.grid(True)
                a1.legend()
                a2.plot(table[param], table["EAC_total"], "o-", color="tab:green")
                a2.set_xlabel(param)
                a2.set_ylabel("Optimal total EAC [€/y]")
                a2.set_title("Optimal annual cost")
                a2.grid(True)
                fig.tight_layout()
                show_fig(fig)

                # Optimal PV per surface for each sweep value, stacked like the allocation diagram
                fig2, ax3 = plt.subplots(figsize=(11, 4.5))
                x = np.arange(len(table))
                bottom = np.zeros(len(table))
                colors = ["#d9776b", "#f5b97f", "#c4c4c4", "#c99473", "#5f5f5f"]
                for (_, lbl), col in zip(SURFACES, colors):
                    vals = table[f"{lbl} [kWp]"].to_numpy()
                    ax3.bar(x, vals, bottom=bottom, color=col, edgecolor="k", lw=0.4,
                            width=0.7, label=lbl)
                    bottom += vals
                ax3.plot(x, table["kWp"].to_numpy(), "k_", ms=22, mew=2, label="Total PV [kWp]")
                ax3.set_xticks(x)
                ax3.set_xticklabels([f"{v:.3g}" for v in table[param]], rotation=45, ha="right")
                ax3.set_xlabel(param)
                ax3.set_ylabel("Optimal PV per surface [kWp]")
                ax3.set_title(f"Optimal PV allocation by surface: {sprof} | {case}")
                ax3.legend(fontsize=8, loc="upper left")
                ax3.grid(True, axis="y", alpha=0.4)
                fig2.tight_layout()
                show_fig(fig2)

        if run_tornado:
            results = []
            for case_name, k, folder in (("PV only", "pv", PV_DIR),
                                         ("PV + battery", "batt", BATT_DIR)):
                src = get_df(folder, sprof, k)
                if src is None:
                    continue
                t, base_opt = tornado(src, k, e, pct / 100)
                results.append((case_name, t, base_opt))
            if not results:
                st.warning("No RES files for this profile.")
            for case_name, t, base_opt in results:
                st.write(f"**{case_name}**: baseline EAC {base_opt['EAC_total']:,.0f} €/y, "
                         f"PV {base_opt['kWp']:.2f} kWp, battery {base_opt['kWh']:.1f} kWh")
                cols = ["Parameter", "Swing [€/y]", "Δ low [€/y]", "Δ high [€/y]", "kWp low", "kWp high"]
                if case_name == "PV + battery":
                    cols += ["kWh low", "kWh high"]
                st.dataframe(t[cols].round(1), hide_index=True)
            if results:
                fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5), squeeze=False)
                for ax, (case_name, t, _) in zip(axes[0], results):
                    y = np.arange(len(t))
                    ax.hlines(y, t["Δ low [€/y]"], t["Δ high [€/y]"], linewidth=8, color="tab:blue")
                    ax.axvline(0, color="k", lw=0.8)
                    ax.set_yticks(y)
                    ax.set_yticklabels(t["Parameter"], fontsize=8)
                    ax.invert_yaxis()
                    ax.set_xlabel(f"Change in optimal EAC [€/y] ({case_name})")
                    ax.grid(True, axis="x")
                fig.tight_layout()
                show_fig(fig)

with tab_orient:
    ctrl, out = st.columns([1, 3])
    with ctrl:
        oprof = st.selectbox("Profile", available, key="or_profile")
        ocase = st.radio("Case", ["PV only", "PV + battery"], key="or_case")
        obin = st.number_input("Size bin [kWp]", value=1.0, min_value=0.1, step=0.5, key="or_bin")
        run_or = st.button("Show orientation for profile")
        run_all = st.button("Optimal allocation: all profiles")
    with out:
        okind = "batt" if ocase == "PV + battery" else "pv"
        if run_or:
            odf = get_df(BATT_DIR if okind == "batt" else PV_DIR, oprof, okind)
            if odf is None:
                st.warning("No RES file for this profile and case.")
            else:
                render_orientation(odf, okind, oprof, ocase, obin, e)
        if run_all:
            render_all_allocation(okind, ocase, e)

with tab_graph:
    ctrl, out = st.columns([1, 3])
    with ctrl:
        gkind = st.selectbox("Graphic", ["Allocation: all profiles", "Battery landscape",
                                         "Orientation scatter"], key="g_kind")
        gprof, target, tol = available[0], 8.0, 0.25
        if gkind != "Allocation: all profiles":
            gprof = st.selectbox("Profile", available, key="g_profile")
        if gkind == "Orientation scatter":
            target = st.number_input("Orientation: total kWp", value=8.0, key="g_target")
            tol = st.number_input("Tolerance ± kWp", value=0.25, key="g_tol")
        draw = st.button("Draw")
    with out:
        descriptions = {
            "Allocation: all profiles": "Optimal PV per surface and battery size for every load profile, with and without battery.",
            "Battery landscape": "Annual cost (EAC) for every combination of PV and battery size for the chosen profile. The star marks the optimum.",
            "Orientation scatter": "Designs near the chosen total PV size: yield, self-consumption, grid feed-in, grid demand and cost against the east/west roof balance. Colour shows the balcony share.",
        }
        st.caption(descriptions[gkind])
        if draw:
            try:
                render_graphic(gkind, gprof, e, target, tol)
            except Exception as ex:
                st.error(str(ex))
