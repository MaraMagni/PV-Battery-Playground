"""PV- und Batterie-Dimensionierung: Web-Version der Desktop-Anwendung.

Gleicher Aufbau wie die Desktop-Anwendung: Eingaben und Aktionen in der Seitenleiste,
Ergebnisse im Hauptbereich, die übrigen Fenster als Reiter. Die Ergebnisse werden aus
den Ordnern RES_profiles_small und RES_profiles_battery_small im Repository gelesen.
"""
import re
import textwrap
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registriert die 3-D-Projektion)

from econ_core import (PROFILES, SURFACES, Economics, _scale_field, allocation_table,
                       best_mix_vs_size, envelope, evaluate, full_report, landscape_grid,
                       load_res, optimum_fast, orientation_scatter_table, pv_incentive,
                       roof_ew_grid, sensitivity_1d)

st.set_page_config(page_title="PV- und Batterie-Dimensionierung", layout="wide")

HERE = Path(__file__).parent
PV_DIR = HERE / "RES_profiles_small"
BATT_DIR = HERE / "RES_profiles_battery_small"

# Bezeichnungen der Flächen (intern englisch, Anzeige deutsch)
DE = {"Balcony": "Balkon", "Roof east": "Dach Ost", "Roof west": "Dach West",
      "Flat roof east": "Flachdach Ost", "Flat roof west": "Flachdach West"}
REGEL_DE = {"whole": "Gesamtanlage", "marginal": "anteilig"}

# Ein-Parameter-Variation (Feld, Bezeichnung, Anzeigefaktor, nur mit Batterie)
SENS = [
    ("cost_cur", "Strompreis [€/kWh]", 1.0, False),
    ("gain_sell", "Einspeisevergütung [€/kWh]", 1.0, False),
    ("r", "Diskontsatz [%]", 100.0, False),
    ("eta_pv", "PV-Wirkungsgrad [-]", 1.0, False),
    ("inv_pv", "PV-Investition [€/kWp]", 1.0, False),
    ("life_pv", "PV-Lebensdauer [a]", 1.0, False),
    ("land_pv_rate", "Landesförderung PV [€/kWp]", 1.0, False),
    ("inv_batt", "Batterie-Investition [€/kWh]", 1.0, True),
    ("life_batt", "Batterie-Lebensdauer [a]", 1.0, True),
    ("bund_batt_rate", "Bundesförderung Batterie [€/kWh]", 1.0, True),
]
# Tornado: dieselben Parameter plus alle Bundesstufen zusammen
TORN = [(f, lbl, b) for f, lbl, _, b in SENS] + [("bund_pv_rates", "Bundesförderung PV, alle Stufen [€/kWp]", False)]

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
    """Lädt eine Ergebnisdatei. Der mtime-Wert aktualisiert den Cache, wenn sich die Datei ändert."""
    return load_res(path, kind)


def get_df(folder: Path, prof: str, kind: str):
    """Ergebnisse für ein Lastprofil und einen Fall, oder None, wenn die Datei fehlt."""
    path = folder / f"RES_{prof}.mat"
    if not path.exists():
        return None
    return load_case(str(path), kind, path.stat().st_mtime)


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def de(text: str) -> str:
    """Zahlen im deutschen Format schreiben: 1,234.5 wird 1.234,5 (Kommas zwischen Wörtern bleiben)."""
    text = re.sub(r"(?<=\d),(?=\d{3})", "\x00", text)
    text = re.sub(r"(?<=\d)\.(?=\d)", ",", text)
    return text.replace("\x00", ".")


def ow_index(b):
    """Ausrichtungsindex Dach: −1 = nur West, +1 = nur Ost."""
    roof = b["A_RE"] + b["A_RW"]
    return (b["A_RE"] - b["A_RW"]) / roof if roof > 0 else float("nan")


def schwelle_hinweis(kwp, e, tol=1.0):
    """Warnung, wenn das Optimum knapp unter einer Bundes-Förderstufe liegt (nur Regel 'Gesamtanlage')."""
    if e.bund_pv_mode != "whole":
        return None
    for t in e.bund_pv_upper:
        gap = t - kwp
        if 0 <= gap <= tol:
            return (f"Das Optimum {kwp:.2f} kWp liegt {gap:.2f} kWp unter der Bundes-Förderstufe "
                    f"von {t:g} kWp. Prüfen, ob das Raster zu grob ist, um die beste Auslegung "
                    f"knapp unter oder an der Grenze zu zeigen.")
    return None


def tornado_de(df, kind, e, pct):
    """Optimum bei −pct, −pct/2, 0, +pct/2 und +pct % Änderung je Eingabe (jeweils einzeln)."""
    levels = [-pct, -pct / 2, 0.0, pct / 2, pct]
    base = optimum_fast(df, kind, e)
    rows, curves = [], {}
    for field, label, batt_only in TORN:
        if batt_only and kind == "pv":
            continue
        opts = [base if lv == 0 else optimum_fast(df, kind, _scale_field(e, field, 1 + lv / 100))
                for lv in levels]
        lo, hi = opts[0], opts[-1]
        curves[label] = [o["EAC_total"] for o in opts]
        rows.append({
            "Parameter": label,
            f"EAC bei −{pct:g} % [€/a]": lo["EAC_total"],
            f"EAC bei +{pct:g} % [€/a]": hi["EAC_total"],
            f"Δ bei −{pct:g} % [€/a]": lo["EAC_total"] - base["EAC_total"],
            f"Δ bei +{pct:g} % [€/a]": hi["EAC_total"] - base["EAC_total"],
            "Spannweite [€/a]": abs(hi["EAC_total"] - lo["EAC_total"]),
            f"PV bei −{pct:g} % [kWp]": lo["kWp"],
            f"PV bei +{pct:g} % [kWp]": hi["kWp"],
            f"Batterie bei −{pct:g} % [kWh]": lo["kWh"],
            f"Batterie bei +{pct:g} % [kWh]": hi["kWh"],
        })
    table = pd.DataFrame(rows).sort_values("Spannweite [€/a]", ascending=False, ignore_index=True)
    return base, table, levels, curves


# ---------------------------------------------------------------- Seitenleiste
with st.sidebar:
    st.header("Daten")
    st.caption("Die Ergebnisse stammen aus den Ordnern RES_profiles_small und "
               "RES_profiles_battery_small im Repository.")
    available = [p for p in PROFILES
                 if (PV_DIR / f"RES_{p}.mat").exists() or (BATT_DIR / f"RES_{p}.mat").exists()]
    if not available:
        st.error("Keine Ergebnisdateien in den Datenordnern gefunden.")
        st.stop()
    profile = st.selectbox("Lastprofil", available, key="profile")

    st.header("Wirtschaft")
    st.number_input("Strompreis [€/kWh]", step=0.01, key="cost_cur")
    st.number_input("Einspeisevergütung [€/kWh]", step=0.01, key="gain_sell")
    st.number_input("Diskontsatz [%]", step=0.1, key="r")
    st.number_input("PV-Wirkungsgrad [-]", step=0.01, key="eta_pv")
    st.number_input("PV-Investition [€/kWp]", step=50.0, key="inv_pv")
    st.number_input("PV-Lebensdauer [a]", step=1, key="life_pv")
    st.number_input("Batterie-Investition [€/kWh]", step=25.0, key="inv_batt")
    st.number_input("Batterie-Lebensdauer [a]", step=1, key="life_batt")

    st.header("Förderung")
    st.number_input("Landesförderung PV [€/kWp]", step=5.0, key="land_pv_rate")
    st.number_input("Landesförderung PV, Obergrenze [kWp]", step=1.0, key="land_pv_cap")
    st.number_input("Landesförderung Batterie [€/kWh]", step=10.0, key="land_batt_rate")
    st.number_input("Bundesförderung Batterie [€/kWh]", step=10.0, key="bund_batt_rate")
    st.number_input("Bundesförderung Batterie, Obergrenze [kWh]", step=5.0, key="bund_batt_cap")
    st.selectbox("Bundesregel PV", ["whole", "marginal"], key="mode",
                 format_func=lambda m: REGEL_DE[m])
    st.caption("Bundesförderung PV-Stufen: Obergrenze [kWp] / Satz [€/kWp]")
    for i in range(len(DEFAULT_TIERS)):
        c1, c2 = st.columns(2)
        c1.number_input(f"Stufe {i + 1}: Obergrenze [kWp]", key=f"up{i}")
        c2.number_input(f"Stufe {i + 1}: Satz [€/kWp]", key=f"rt{i}")

    st.header("Aktionen")
    run_main = st.button("Ausgewähltes Lastprofil berechnen", type="primary")
    st.button("Eingaben auf Ausgangswerte zurücksetzen", on_click=reset_inputs)
    run_report = st.button("Vollbericht: alle Lastprofile")

ss = st.session_state
uppers = [ss[f"up{i}"] for i in range(len(DEFAULT_TIERS))]
rates = [ss[f"rt{i}"] for i in range(len(DEFAULT_TIERS))]
if np.any(np.diff(uppers) <= 0):
    st.error("Die Obergrenzen der Bundesstufen müssen steigen.")
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


# ---------------------------------------------------------------- Ergebnisse
def render_main(e, prof):
    pv_df = get_df(PV_DIR, prof, "pv")
    bt_df = get_df(BATT_DIR, prof, "batt")
    lines = [f"Lastprofil: {prof}    Bundesregel PV: {REGEL_DE[e.bund_pv_mode]}", ""]
    outs, best = {}, {}
    for label, kind, df in (("Nur PV", "pv", pv_df), ("PV + Batterie", "batt", bt_df)):
        if df is None:
            continue
        outs[kind] = evaluate(df, kind, e)
        b = optimum_fast(df, kind, e)
        best[kind] = b
        demand = b["Self_kWh"] + b["GridDem_kWh"]
        lines += [
            f"{label}: PV {b['kWp']:.2f} kWp | Batterie {b['kWh']:.1f} kWh | EAC {b['EAC_total']:,.0f} €/a",
            f"  Gesamtstrombedarf {demand:,.0f} kWh/a = Eigenverbrauch {b['Self_kWh']:,.0f} "
            f"+ Netzbezug {b['GridDem_kWh']:,.0f} kWh/a",
            f"  Förderung PV {b['Incentive_PV']:,.0f} € | Förderung Batterie {b['Incentive_Batt']:,.0f} €",
            "  Ausrichtung: " + ", ".join(f"{DE[lbl]} {b[c] * e.eta_pv:.2f} kWp" for c, lbl in SURFACES)
            + f" | Ausrichtungsindex Dach O/W {ow_index(b):+.2f}",
        ]
        note = schwelle_hinweis(b["kWp"], e)
        if note:
            lines.append("  HINWEIS: " + note)
    if not best:
        st.warning(f"Keine Ergebnisdatei für {prof} gefunden.")
        return
    if "pv" in best and "batt" in best:
        gain = best["batt"]["EAC_total"] - best["pv"]["EAC_total"]
        lines.append(f"Gewinn durch die Batterie: {gain:,.0f} €/a")
    st.code(de("\n".join(lines)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    if "pv" in outs:
        env = envelope(outs["pv"], by_battery=False)
        ax1.plot(env["kWp_r"], env["EAC_total"], lw=1.6, label="Nur PV")
    if "batt" in outs:
        envb = envelope(outs["batt"], by_battery=True)
        for kwh, g in envb.groupby("kWh"):
            if kwh == 0:
                continue
            ax1.plot(g["kWp_r"], g["EAC_total"], lw=1, alpha=0.8, label=f"{kwh:g} kWh")
    ax1.set_xlabel("Installierte PV-Leistung [kWp]")
    ax1.set_ylabel("Gesamtkosten EAC [€/a]")
    ax1.set_title(f"EAC über PV-Leistung: {prof}")
    ax1.grid(True)
    ax1.legend(fontsize=7)

    x = np.linspace(0, 40, 801)
    other = "marginal" if e.bund_pv_mode == "whole" else "whole"
    e_other = replace(e, bund_pv_mode=other)
    ax2.plot(x, x * e.inv_pv - pv_incentive(x, e), label=f"Netto-PV-Investition ({REGEL_DE[e.bund_pv_mode]})")
    ax2.plot(x, x * e.inv_pv - pv_incentive(x, e_other), "--", label=f"Andere Regel ({REGEL_DE[other]})")
    ax2.set_xlabel("Installierte PV-Leistung [kWp]")
    ax2.set_ylabel("Netto-Investition [€]")
    ax2.set_title("Netto-PV-Investition nach Förderung")
    ax2.grid(True)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    show_fig(fig)


def render_report(e):
    rows = []
    bar = st.progress(0.0, text="Vollbericht wird erstellt")
    for k, p in enumerate(available):
        rows.append(full_report(p, get_df(PV_DIR, p, "pv"), get_df(BATT_DIR, p, "batt"), e))
        bar.progress((k + 1) / len(available))
    report = pd.concat(rows, ignore_index=True)
    report["Gesamtstrombedarf [kWh/y]"] = report["Self-consumed [kWh/y]"] + report["Grid demand [kWh/y]"]
    rename = {
        "Profile": "Lastprofil", "Case": "Fall", "PV [kWp]": "PV [kWp]",
        "Battery [kWh]": "Batterie [kWh]", "Total EAC [€/y]": "Gesamtkosten EAC [€/a]",
        "Gain vs without PV/battery [€/y]": "Gewinn ggü. ohne PV/Batterie [€/a]",
        "PV subsidy [€]": "Förderung PV [€]", "Battery subsidy [€]": "Förderung Batterie [€]",
        "Self-consumed [kWh/y]": "Eigenverbrauch [kWh/a]", "Grid demand [kWh/y]": "Netzbezug [kWh/a]",
        "Gesamtstrombedarf [kWh/y]": "Gesamtstrombedarf [kWh/a]",
    }
    show = report.rename(columns=rename)
    show["Fall"] = show["Fall"].map({"PV only": "Nur PV", "PV + battery": "PV + Batterie"})
    cols = ["Lastprofil", "Fall", "PV [kWp]", "Batterie [kWh]", "Gesamtkosten EAC [€/a]",
            "Gewinn ggü. ohne PV/Batterie [€/a]", "Förderung PV [€]", "Förderung Batterie [€]",
            "Eigenverbrauch [kWh/a]", "Netzbezug [kWh/a]", "Gesamtstrombedarf [kWh/a]"]
    st.dataframe(show[cols].round(1), hide_index=True)
    st.download_button("Vollbericht als CSV speichern", report.to_csv(index=False),
                       file_name="vollbericht.csv", mime="text/csv")


# ---------------------------------------------------------------- PV-Ausrichtung
def render_orientation(df, kind, prof, case, bin_kwp, e):
    base = optimum_fast(df, kind, e)
    mix = best_mix_vs_size(df, kind, e, bin_kwp)
    grid = roof_ew_grid(df, kind, e, base)
    areas = {DE[lbl]: base[c] * e.eta_pv for c, lbl in SURFACES}
    st.code(de(f"{prof} | {case} | Optimum {base['kWp']:.2f} kWp, Batterie {base['kWh']:.1f} kWh, "
               f"EAC {base['EAC_total']:,.0f} €/a\n"
               "Optimale kWp je Fläche: " + ", ".join(f"{k} {v:.2f}" for k, v in areas.items())))

    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    ax1, ax2, ax3, ax4 = axs.ravel()
    ax1.stackplot(mix["kWp"], [mix[c] for c, _ in SURFACES],
                  labels=[DE[lbl] for _, lbl in SURFACES], alpha=0.85)
    ax1.axvline(base["kWp"], color="k", ls="--", lw=0.8)
    ax1.set_xlabel("Gesamte installierte PV-Leistung [kWp]")
    ax1.set_ylabel("kWp je Fläche")
    ax1.set_title("Beste Flächenverteilung je Gesamtgröße")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(True)

    ax2.plot(mix["kWp"], mix["EAC_total"], "o-", ms=3, color="tab:green")
    ax2.plot(base["kWp"], base["EAC_total"], "r*", ms=14, label="Optimum")
    ax2.set_xlabel("Gesamte installierte PV-Leistung [kWp]")
    ax2.set_ylabel("Gesamtkosten EAC [€/a]")
    ax2.set_title("Beste erreichbare Kosten je Größe")
    ax2.grid(True)
    ax2.legend()

    xs = grid.columns.to_numpy(dtype=float)
    ys = grid.index.to_numpy(dtype=float)
    mesh = ax3.pcolormesh(xs, ys, grid.to_numpy(), shading="nearest", cmap="viridis")
    fig.colorbar(mesh, ax=ax3, label="Gesamtkosten EAC [€/a]")
    ax3.plot(base["A_RE"] * e.eta_pv, base["A_RW"] * e.eta_pv, "r*", ms=14)
    ax3.set_xlabel("Dach Ost [kWp]")
    ax3.set_ylabel("Dach West [kWp]")
    ax3.set_title("Ost- vs. West-Dach (übrige Flächen im Optimum)")

    ax4.set_title("Optimale Verteilung: alle Lastprofile (Schaltfläche in der Steuerung)")
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
        rows.append({"Lastprofil": p, **{DE[lbl]: b[c] * e.eta_pv for c, lbl in SURFACES}})
    if not rows:
        st.warning("Keine Ergebnisdateien für diesen Fall gefunden.")
        return
    t = pd.DataFrame(rows).set_index("Lastprofil")
    st.text(f"Optimale kWp je Fläche, alle Lastprofile ({case})")
    st.dataframe(t.round(2))
    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(len(t))
    for lbl in t.columns:
        ax.bar(np.arange(len(t)), t[lbl].to_numpy(), bottom=bottom, label=lbl)
        bottom += t[lbl].to_numpy()
    ax.set_xticks(np.arange(len(t)))
    ax.set_xticklabels(t.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Optimale installierte PV [kWp]")
    ax.set_title(f"Optimale PV-Verteilung nach Fläche: {case}")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y")
    fig.tight_layout()
    show_fig(fig)


# ---------------------------------------------------------------- Berichtsgrafiken
def render_graphic(kind_name, prof, e, target, tol):
    if kind_name == "Verteilung aller Lastprofile":
        parts = [allocation_table({p: (get_df(PV_DIR, p, "pv"), get_df(BATT_DIR, p, "batt"))}, e)
                 for p in available]
        t = pd.concat(parts, ignore_index=True)
        if t.empty:
            st.warning("Keine Ergebnisdateien gefunden.")
            return
        x = np.arange(len(t))
        profs = list(t["Profile"])
        fig, ax = plt.subplots(figsize=(12, 6))
        bottom = np.zeros(len(t))
        colors = ["#d9776b", "#f5b97f", "#c4c4c4", "#c99473", "#5f5f5f"]
        for (c, lbl), col in zip(SURFACES, colors):
            ax.bar(x, t[lbl].to_numpy(), bottom=bottom, color=col, edgecolor="k", lw=0.5,
                   label=DE[lbl], width=0.6)
            bottom += t[lbl].to_numpy()
        ax.set_ylabel("PV [kWp]")
        ax.set_xticks(x)
        ax.set_xticklabels(list(t["Case"]), fontsize=8)
        # Name des Lastprofils unter jedem Balkenpaar, mit Trennlinie zwischen den Profilen
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
        ax2.set_ylabel("Batterie [kWh]")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, ncol=3, loc="upper center")
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.28)
        show_fig(fig)
        st.dataframe(t.round(2), hide_index=True)

    elif kind_name == "Batterie-Landschaft":
        df = get_df(BATT_DIR, prof, "batt")
        if df is None:
            st.warning("Keine Ergebnisdatei mit Batterie für dieses Lastprofil.")
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
        ax1.set_xlabel("Batterie [kWh]")
        ax1.set_ylabel("PV [kWp]")
        ax1.set_zlabel("EAC [k€/a]")
        ax1.set_title(f"{prof}: EAC-Landschaft, Optimum {kwp_v[i]:.1f} kWp / {kwh_v[j]:g} kWh")
        ax2 = fig.add_subplot(2, 1, 2)
        cs = ax2.contourf(BB, KK, Z / 1000, 20, cmap="viridis")
        fig.colorbar(cs, ax=ax2, label="EAC [k€/a]")
        near = Z > best - tol_eur
        ax2.plot(BB[near], KK[near], "w.", ms=6)
        ax2.plot(kwh_v[j], kwp_v[i], "r*", ms=16)
        ax2.set_xlabel("Batterie [kWh]")
        ax2.set_ylabel("PV [kWp]")
        ax2.set_title(f"Höhenlinien; weiße Punkte: innerhalb 1 % des Optimums ({tol_eur:.0f} €/a)")
        fig.tight_layout()
        show_fig(fig)
        st.code(de(f"{prof}: Optimum {kwp_v[i]:.2f} kWp, {kwh_v[j]:g} kWh, EAC {best:,.0f} €/a. "
                   f"Raster: {len(kwp_v)} PV-Größen × {len(kwh_v)} Batteriegrößen."))

    else:  # Ausrichtungsstreuung
        df = get_df(PV_DIR, prof, "pv")
        if df is None:
            st.warning("Keine Ergebnisdatei ohne Batterie für dieses Lastprofil.")
            return
        t = orientation_scatter_table(df, e, target, tol)
        if t.empty:
            st.warning("Keine Auslegungen in dieser Größe. Toleranz erhöhen.")
            return
        cols = {"PV yield [kWh/y]": "PV-Ertrag [kWh/a]",
                "Self-consumption [%]": "Eigenverbrauch [%]",
                "PV to grid [kWh/y]": "Einspeisung [kWh/a]",
                "Grid demand [kWh/y]": "Netzbezug [kWh/a]",
                "EAC [€/y]": "EAC [€/a]"}
        fig, axes = plt.subplots(len(cols), 1, figsize=(10, 14), sharex=True)
        for ax, (col, label) in zip(axes, cols.items()):
            sc = ax.scatter(t["OW"], t[col], c=100 * t["Balcony share"], cmap="viridis", s=22)
            ax.set_ylabel(label, fontsize=8)
            ax.grid(True, alpha=0.4)
            fig.colorbar(sc, ax=ax, label="Balkonanteil [%]")
        axes[-1].set_xlabel("Ausrichtungsindex Dach (−1 = West, +1 = Ost)")
        fig.suptitle(f"{prof}: Auslegungen nahe {target:g} kWp (±{tol:g}), n = {len(t)}")
        fig.tight_layout()
        show_fig(fig)
        st.code("Ausrichtungsindex nur Dachflächen; Farbe = Balkonanteil. Zeilen ohne Dach sind ausgeschlossen.")


# ---------------------------------------------------------------- Reiter
tab_main, tab_sens, tab_orient, tab_graph = st.tabs(
    ["Ergebnisse", "Sensitivität", "PV-Ausrichtung", "Berichtsgrafiken"])

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
        sprof = st.selectbox("Lastprofil", available, key="sens_profile")
        case = st.radio("Fall", ["Nur PV", "PV + Batterie"], key="sens_case")
        param = st.selectbox("Ein-Parameter-Variation", [s[1] for s in SENS], key="sens_param")
        _, field, scale, batt_only = next(s for s in SENS if s[1] == param)
        base_val = getattr(e, field) * scale
        lo = st.number_input("Von", value=float(round(base_val * 0.5, 4)), key=f"lo_{field}")
        hi = st.number_input("Bis", value=float(round(base_val * 1.5, 4)), key=f"hi_{field}")
        steps = st.number_input("Schritte", value=9, min_value=2, step=1, key="sens_steps")
        run_sweep = st.button("Variation berechnen")
        st.divider()
        pct = st.number_input("Änderung ±%", value=20.0, step=5.0, key="sens_pct")
        run_tornado = st.button("Tornado berechnen (beide Fälle)")

    with out:
        kind = "batt" if case == "PV + Batterie" else "pv"
        if run_sweep:
            df = get_df(BATT_DIR if kind == "batt" else PV_DIR, sprof, kind)
            if df is None:
                st.warning("Keine Ergebnisdatei für dieses Lastprofil und diesen Fall.")
            elif batt_only and kind == "pv":
                st.info("Dieser Parameter wirkt nur auf den Fall PV + Batterie. Diesen Fall wählen.")
            else:
                table = sensitivity_1d(df, kind, e, field,
                                       np.linspace(lo / scale, hi / scale, int(steps)))
                table.insert(0, param, table["value"] * scale)
                disp = pd.DataFrame({
                    param: table[param],
                    "PV [kWp]": table["kWp"],
                    "Batterie [kWh]": table["kWh"],
                    "Gesamtkosten EAC [€/a]": table["EAC_total"],
                    "Netzbezug [kWh/a]": table["GridDem_kWh"],
                    "Gesamtstrombedarf [kWh/a]": table["Self_kWh"] + table["GridDem_kWh"],
                    "Förderung PV [€]": table["Incentive_PV"],
                    "Förderung Batterie [€]": table["Incentive_Batt"],
                })
                for c, lbl in SURFACES:
                    disp[f"{DE[lbl]} [kWp]"] = table[c].to_numpy() * e.eta_pv
                st.text(f"{sprof} | {case} | Variation von {param}")
                st.dataframe(disp.round(2), hide_index=True)
                st.download_button("Tabelle als CSV speichern",
                                   table.assign(Lastprofil=sprof, Fall=case).to_csv(index=False),
                                   file_name="sensitivitaet.csv", mime="text/csv")
                fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
                a1.plot(table[param], table["kWp"], "o-", label="PV [kWp]")
                if kind == "batt":
                    a1.plot(table[param], table["kWh"], "s--", label="Batterie [kWh]")
                a1.set_xlabel(param)
                a1.set_ylabel("Optimale Größe")
                a1.grid(True)
                a1.legend()
                a2.plot(table[param], table["EAC_total"], "o-", color="tab:green")
                a2.set_xlabel(param)
                a2.set_ylabel("Optimale Gesamtkosten EAC [€/a]")
                a2.set_title("Optimale jährliche Kosten")
                a2.grid(True)
                fig.tight_layout()
                show_fig(fig)

                # Optimale PV je Fläche für jeden Wert, gestapelt wie im Flächendiagramm
                fig2, ax3 = plt.subplots(figsize=(11, 4.5))
                x = np.arange(len(table))
                bottom = np.zeros(len(table))
                colors = ["#d9776b", "#f5b97f", "#c4c4c4", "#c99473", "#5f5f5f"]
                for (c, lbl), col in zip(SURFACES, colors):
                    vals = disp[f"{DE[lbl]} [kWp]"].to_numpy()
                    ax3.bar(x, vals, bottom=bottom, color=col, edgecolor="k", lw=0.4,
                            width=0.7, label=DE[lbl])
                    bottom += vals
                ax3.plot(x, table["kWp"].to_numpy(), "k_", ms=22, mew=2, label="PV gesamt [kWp]")
                ax3.set_xticks(x)
                ax3.set_xticklabels([f"{v:.3g}" for v in table[param]], rotation=45, ha="right")
                ax3.set_xlabel(param)
                ax3.set_ylabel("Optimale PV je Fläche [kWp]")
                ax3.set_title(f"Optimale PV-Verteilung nach Fläche: {sprof} | {case}")
                ax3.legend(fontsize=8, loc="upper left")
                ax3.grid(True, axis="y", alpha=0.4)
                fig2.tight_layout()
                show_fig(fig2)

        if run_tornado:
            results = []
            for case_name, k, folder in (("Nur PV", "pv", PV_DIR), ("PV + Batterie", "batt", BATT_DIR)):
                src = get_df(folder, sprof, k)
                if src is None:
                    continue
                base, t, levels, curves = tornado_de(src, k, e, pct)
                results.append((case_name, k, base, t, levels, curves))
            if not results:
                st.warning("Keine Ergebnisdateien für dieses Lastprofil.")
            for case_name, k, base, t, levels, curves in results:
                bat_txt = f", Batterie {base['kWh']:.1f} kWh" if k == "batt" else ""
                st.markdown(de(f"**{case_name}**: Basis-EAC {base['EAC_total']:,.0f} €/a, "
                               f"PV {base['kWp']:.2f} kWp{bat_txt}"))
                cols = ["Parameter", "Spannweite [€/a]", f"Δ bei −{pct:g} % [€/a]",
                        f"Δ bei +{pct:g} % [€/a]", f"PV bei −{pct:g} % [kWp]", f"PV bei +{pct:g} % [kWp]"]
                if k == "batt":
                    cols += [f"Batterie bei −{pct:g} % [kWh]", f"Batterie bei +{pct:g} % [kWh]"]
                st.dataframe(t[cols].round(1), hide_index=True)

                # Balkendiagramm: Änderung der EAC je Eingabe
                fig, ax = plt.subplots(figsize=(10, 5))
                y = np.arange(len(t))
                ax.hlines(y, t[f"Δ bei −{pct:g} % [€/a]"], t[f"Δ bei +{pct:g} % [€/a]"],
                          linewidth=8, color="tab:blue")
                ax.axvline(0, color="k", lw=0.8)
                ax.set_yticks(y)
                ax.set_yticklabels(t["Parameter"], fontsize=8)
                ax.invert_yaxis()
                ax.set_xlabel(f"Änderung der optimalen EAC [€/a] ({case_name})")
                ax.grid(True, axis="x")
                fig.tight_layout()
                show_fig(fig)

                # Liniendiagramm: EAC über die Änderung der Eingabe, eine Linie je Eingabe
                fig, ax = plt.subplots(figsize=(11, 5.5))
                for label in t["Parameter"]:
                    ax.plot(levels, curves[label], "o-", ms=4, label=label)
                ax.axvline(0, color="k", lw=0.8)
                ax.set_xlabel("Änderung der Eingabe [%]")
                ax.set_ylabel("Jährliche Gesamtkosten EAC [€/a]")
                ax.set_title(f"Tornado-Linien: {case_name}")
                ax.grid(True)
                ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
                fig.tight_layout()
                show_fig(fig)

with tab_orient:
    ctrl, out = st.columns([1, 3])
    with ctrl:
        oprof = st.selectbox("Lastprofil", available, key="or_profile")
        ocase = st.radio("Fall", ["Nur PV", "PV + Batterie"], key="or_case")
        obin = st.number_input("Größenintervall [kWp]", value=1.0, min_value=0.1, step=0.5, key="or_bin")
        run_or = st.button("Ausrichtung für Lastprofil anzeigen")
        run_all = st.button("Optimale Verteilung: alle Lastprofile")
    with out:
        okind = "batt" if ocase == "PV + Batterie" else "pv"
        if run_or:
            odf = get_df(BATT_DIR if okind == "batt" else PV_DIR, oprof, okind)
            if odf is None:
                st.warning("Keine Ergebnisdatei für dieses Lastprofil und diesen Fall.")
            else:
                render_orientation(odf, okind, oprof, ocase, obin, e)
        if run_all:
            render_all_allocation(okind, ocase, e)

with tab_graph:
    ctrl, out = st.columns([1, 3])
    with ctrl:
        gkind = st.selectbox("Grafik", ["Verteilung aller Lastprofile", "Batterie-Landschaft",
                                        "Ausrichtungsstreuung"], key="g_kind")
        gprof, target, tol = available[0], 8.0, 0.25
        if gkind != "Verteilung aller Lastprofile":
            gprof = st.selectbox("Lastprofil", available, key="g_profile")
        if gkind == "Ausrichtungsstreuung":
            target = st.number_input("Ziel-PV-Leistung [kWp]", value=8.0, key="g_target")
            tol = st.number_input("Toleranz ± kWp", value=0.25, key="g_tol")
        draw = st.button("Zeichnen")
    with out:
        descriptions = {
            "Verteilung aller Lastprofile": "Optimale PV je Fläche und Batteriegröße für jedes Lastprofil, mit und ohne Batterie.",
            "Batterie-Landschaft": "Jährliche Kosten (EAC) für jede Kombination aus PV- und Batteriegröße des gewählten Lastprofils. Der Stern markiert das Optimum.",
            "Ausrichtungsstreuung": "Auslegungen nahe der gewählten Gesamtleistung: Ertrag, Eigenverbrauch, Einspeisung, Netzbezug und Kosten über das Ost-West-Verhältnis der Dächer. Die Farbe zeigt den Balkonanteil.",
        }
        st.caption(descriptions[gkind])
        if draw:
            try:
                render_graphic(gkind, gprof, e, target, tol)
            except Exception as ex:
                st.error(str(ex))
