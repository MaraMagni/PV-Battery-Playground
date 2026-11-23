"""PV + battery sizing: web version (Streamlit).

Upload the RES_*.mat files in the sidebar; the economics are recomputed
from the inputs with econ_core.py, which must sit next to this file.
"""
import os
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from econ_core import (PROFILES, SWEEP_PARAMS, Economics, best_mix_vs_size, envelope,
                       evaluate, full_report, landscape_grid, load_res, optimum_fast,
                       sensitivity_1d, tornado, threshold_note, SURFACES)

st.set_page_config(page_title="PV + battery sizing", layout="wide")

DEFAULT_TIERS = [(10, 150), (20, 140), (100, 130), (1000, 120)]


@st.cache_data(show_spinner=False, max_entries=4)
def load_bytes(data: bytes, kind: str) -> pd.DataFrame:
    """Load one RES file. Only the result table is cached, and only a few profiles are kept."""
    fd, path = tempfile.mkstemp(suffix=".mat")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    try:
        return load_res(path, kind)
    finally:
        os.remove(path)


def upload_map(files):
    """Map profile name -> uploaded file, from files named RES_<profile>.mat. Nothing is loaded yet."""
    out = {}
    for f in files or []:
        name = os.path.splitext(f.name)[0]
        if name.startswith("RES_") and name[4:] in PROFILES:
            out[name[4:]] = f
    return out


def get_df(file_map, prof, kind):
    """Load the results for one profile on demand, or return None if it wasn't uploaded."""
    f = file_map.get(prof)
    return None if f is None else load_bytes(f.getvalue(), kind)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Data")
    pv_files = st.file_uploader("PV-only results (RES_*.mat)", type="mat", accept_multiple_files=True)
    batt_files = st.file_uploader("PV + battery results (RES_*.mat)", type="mat", accept_multiple_files=True)
    pv_data = upload_map(pv_files)
    batt_data = upload_map(batt_files)

    st.header("Economics")
    cost_cur = st.number_input("Electricity price [€/kWh]", value=0.33, step=0.01)
    gain_sell = st.number_input("Feed-in tariff [€/kWh]", value=0.05, step=0.01)
    r = st.number_input("Discount rate [%]", value=3.5, step=0.1)
    eta_pv = st.number_input("Module efficiency [kWp/m²]", value=0.23, step=0.01)
    inv_pv = st.number_input("PV investment [€/kWp]", value=1500.0, step=50.0)
    life_pv = st.number_input("PV lifetime [y]", value=20, step=1)
    inv_batt = st.number_input("Battery investment [€/kWh]", value=500.0, step=25.0)
    life_batt = st.number_input("Battery lifetime [y]", value=12, step=1)

    st.header("Subsidies")
    land_pv_rate = st.number_input("State PV rate [€/kWp]", value=125.0, step=5.0)
    land_pv_cap = st.number_input("State PV cap [kWp]", value=20.0, step=1.0)
    land_batt_rate = st.number_input("State battery [€/kWh]", value=0.0, step=10.0)
    bund_batt_rate = st.number_input("Federal battery [€/kWh]", value=150.0, step=10.0)
    bund_batt_cap = st.number_input("Federal battery cap [kWh]", value=50.0, step=5.0)
    mode = st.selectbox("Federal PV rule", ["whole", "marginal"])

    st.caption("Federal PV bands: upper bound [kWp] and rate [€/kWp]")
    uppers, rates = [], []
    for i, (u, rt) in enumerate(DEFAULT_TIERS):
        c1, c2 = st.columns(2)
        uppers.append(c1.number_input(f"Upper {i + 1} [kWp]", value=float(u), key=f"up{i}"))
        rates.append(c2.number_input(f"Rate {i + 1} [€/kWp]", value=float(rt), key=f"rt{i}"))

if np.any(np.diff(uppers) <= 0):
    st.error("Federal band upper bounds must be increasing.")
    st.stop()

e = Economics(
    cost_cur=cost_cur, gain_sell=gain_sell, r=r / 100, eta_pv=eta_pv,
    inv_pv=inv_pv, life_pv=int(life_pv), inv_batt=inv_batt, life_batt=int(life_batt),
    land_pv_rate=land_pv_rate, land_pv_cap_kwp=land_pv_cap,
    land_batt_rate=land_batt_rate, bund_batt_rate=bund_batt_rate,
    bund_batt_cap_kwh=bund_batt_cap,
    bund_pv_upper=tuple(uppers), bund_pv_rates=tuple(rates), bund_pv_mode=mode,
)

available = [p for p in PROFILES if p in pv_data or p in batt_data]
if not available:
    st.info("Upload at least one RES_*.mat file in the sidebar to start.")
    st.stop()

# ---------------------------------------------------------------- tabs
tab_results, tab_sens, tab_orient, tab_report = st.tabs(
    ["Results", "Sensitivity", "PV orientation", "Full report"])

with tab_results:
    prof = st.selectbox("Load profile", available, key="res_prof")
    pv_df = get_df(pv_data, prof, "pv")
    bt_df = get_df(batt_data, prof, "batt")

    cols = st.columns(2)
    for col, label, kind, df in [(cols[0], "PV only", "pv", pv_df),
                                 (cols[1], "PV + battery", "batt", bt_df)]:
        with col:
            st.subheader(label)
            if df is None:
                st.write("No file uploaded for this case.")
                continue
            b = optimum_fast(df, kind, e)
            st.metric("Optimal PV [kWp]", f"{b['kWp']:.2f}")
            st.metric("Optimal battery [kWh]", f"{b['kWh']:.1f}")
            st.metric("Total EAC [€/y]", f"{b['EAC_total']:,.0f}")
            st.write(f"Grid demand {b['GridDem_kWh']:,.0f} kWh/y · "
                     f"PV subsidy {b['Incentive_PV']:,.0f} € · "
                     f"battery subsidy {b['Incentive_Batt']:,.0f} €")
            note = threshold_note(b["kWp"], e)
            if note:
                st.warning(note)

    if pv_df is not None and bt_df is not None:
        gain = optimum_fast(bt_df, "batt", e)["EAC_total"] - optimum_fast(pv_df, "pv", e)["EAC_total"]
        st.write(f"**Gain from adding a battery: {gain:,.0f} €/y**")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if pv_df is not None:
        env = envelope(evaluate(pv_df, "pv", e), by_battery=False)
        ax.plot(env["kWp_r"], env["EAC_total"], lw=1.6, label="PV only")
    if bt_df is not None:
        envb = envelope(evaluate(bt_df, "batt", e), by_battery=True)
        for kwh, g in envb.groupby("kWh"):
            if kwh == 0:
                continue
            ax.plot(g["kWp_r"], g["EAC_total"], lw=1, alpha=0.8, label=f"{kwh:g} kWh")
    ax.set_xlabel("Installed PV [kWp]")
    ax.set_ylabel("Total EAC [€/y]")
    ax.set_title(f"EAC vs PV size: {prof}")
    ax.grid(True)
    ax.legend(fontsize=7)
    st.pyplot(fig)
    plt.close(fig)

    if bt_df is not None:
        kwh_v, kwp_v, Z = landscape_grid(bt_df, e)
        Z = np.where(np.isfinite(Z), Z, np.nanmin(Z))
        KK, BB = np.meshgrid(kwp_v, kwh_v, indexing="ij")
        i, j = np.unravel_index(np.nanargmax(Z), Z.shape)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        cs = ax.contourf(BB, KK, Z / 1000, 20, cmap="viridis")
        fig.colorbar(cs, ax=ax, label="EAC [k€/y]")
        ax.plot(kwh_v[j], kwp_v[i], "r*", ms=16)
        ax.set_xlabel("Battery [kWh]")
        ax.set_ylabel("PV [kWp]")
        ax.set_title(f"Battery landscape: optimum {kwp_v[i]:.1f} kWp / {kwh_v[j]:g} kWh")
        st.pyplot(fig)
        plt.close(fig)

with tab_sens:
    labels = [s[0] for s in SWEEP_PARAMS]
    param_label = st.selectbox("Parameter", labels)
    _, field, scale, batt_only = next(s for s in SWEEP_PARAMS if s[0] == param_label)
    case = st.radio("Case", ["PV only", "PV + battery"], horizontal=True, key="sens_case")
    kind = "batt" if case == "PV + battery" else "pv"
    sprof = st.selectbox("Load profile", available, key="sens_prof")
    df = get_df(batt_data if kind == "batt" else pv_data, sprof, kind)

    base = getattr(e, field) * scale
    c1, c2, c3 = st.columns(3)
    lo_val = c1.number_input("From", value=float(round(base * 0.5, 4)), key="sens_lo")
    hi_val = c2.number_input("To", value=float(round(base * 1.5, 4)), key="sens_hi")
    steps = c3.number_input("Steps", value=9, min_value=2, step=1, key="sens_n")

    if st.button("Run sweep"):
        if df is None:
            st.warning("No file uploaded for this profile and case.")
        elif batt_only and kind == "pv":
            st.info("This parameter only affects the PV + battery case. Choose that case.")
        else:
            table = sensitivity_1d(df, kind, e, field,
                                   np.linspace(lo_val / scale, hi_val / scale, int(steps)))
            table.insert(0, param_label, table["value"] * scale)
            st.dataframe(table[[param_label, "kWp", "kWh", "EAC_total", "GridDem_kWh",
                                "Incentive_PV", "Incentive_Batt"]].round(2), hide_index=True)
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
            a1.plot(table[param_label], table["kWp"], "o-", label="PV [kWp]")
            if kind == "batt":
                a1.plot(table[param_label], table["kWh"], "s--", label="Battery [kWh]")
            a1.set_xlabel(param_label); a1.set_ylabel("Optimal size"); a1.grid(True); a1.legend()
            a2.plot(table[param_label], table["EAC_total"], "o-", color="tab:green")
            a2.set_xlabel(param_label); a2.set_ylabel("Optimal total EAC [€/y]"); a2.grid(True)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.subheader("Tornado")
    pct = st.number_input("Change ±%", value=20.0, step=5.0)
    if st.button("Run tornado (both cases)"):
        for case_name, k, src in (("PV only", "pv", get_df(pv_data, sprof, "pv")),
                                  ("PV + battery", "batt", get_df(batt_data, sprof, "batt"))):
            if src is None:
                continue
            t, base_opt = tornado(src, k, e, pct / 100)
            st.write(f"**{case_name}**: baseline EAC {base_opt['EAC_total']:,.0f} €/y, "
                     f"PV {base_opt['kWp']:.2f} kWp, battery {base_opt['kWh']:.1f} kWh")
            st.dataframe(t[["Parameter", "Swing [€/y]", "Δ low [€/y]", "Δ high [€/y]",
                            "kWp low", "kWp high"]].round(1), hide_index=True)

with tab_orient:
    oprof = st.selectbox("Load profile", available, key="or_prof")
    ocase = st.radio("Case", ["PV only", "PV + battery"], horizontal=True, key="or_case")
    okind = "batt" if ocase == "PV + battery" else "pv"
    odf = get_df(batt_data if okind == "batt" else pv_data, oprof, okind)
    if odf is None:
        st.info("No file uploaded for this profile and case.")
    else:
        bin_kwp = st.number_input("Size bin [kWp]", value=1.0, min_value=0.1, step=0.5)
        base = optimum_fast(odf, okind, e)
        mix = best_mix_vs_size(odf, okind, e, bin_kwp)
        st.write(f"Optimum {base['kWp']:.2f} kWp, battery {base['kWh']:.1f} kWh, "
                 f"EAC {base['EAC_total']:,.0f} €/y")
        st.write("Optimal kWp per surface: " +
                 ", ".join(f"{lbl} {base[c] * e.eta_pv:.2f}" for c, lbl in SURFACES))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.stackplot(mix["kWp"], [mix[c] for c, _ in SURFACES],
                     labels=[lbl for _, lbl in SURFACES], alpha=0.85)
        ax.axvline(base["kWp"], color="k", ls="--", lw=0.8)
        ax.set_xlabel("Total installed PV [kWp]")
        ax.set_ylabel("kWp per surface")
        ax.set_title("Best surface allocation for each total size")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

with tab_report:
    st.write("Builds the report one profile at a time, so only one profile is in memory at once.")
    if st.button("Build full report (all profiles)"):
        rows = []
        bar = st.progress(0.0)
        for k, p in enumerate(available):
            rows.append(full_report(p, get_df(pv_data, p, "pv"), get_df(batt_data, p, "batt"), e))
            bar.progress((k + 1) / len(available))
        report = pd.concat(rows, ignore_index=True)
        show = report[["Profile", "Case", "PV [kWp]", "Battery [kWh]", "Total EAC [€/y]",
                       "Gain vs without PV/battery [€/y]", "PV subsidy [€]", "Battery subsidy [€]"]]
        st.dataframe(show.round(1), hide_index=True)
        st.download_button("Download full report (CSV)", report.to_csv(index=False),
                           file_name="full_report.csv", mime="text/csv")
