"""
Economic core for PV and PV+battery sizing.

Reads the physical results saved by the MATLAB optimizers (RES_*.mat) and
recomputes annualised costs for any set of economic inputs and subsidy rules.
No simulation is run here, so changing prices, costs or subsidies is instant.

Mirrors recompute_econ_pv, recompute_econ_batt, pv_incentive and
batt_incentive in postproc_v3_2.m.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.io import loadmat

AREA_COLS = ["A_S", "A_RE", "A_RW", "A_ReF", "A_RwF"]
PV_PHYS_COLS = AREA_COLS + ["PVYield_kWh", "Self_kWh", "PV2grid_kWh", "GridDem_kWh"]
BATT_PHYS_COLS = AREA_COLS + [
    "E_batt_kWh", "PVYield_kWh", "SelfPV_kWh", "SelfBatt_kWh",
    "PV2grid_kWh", "GridDem_kWh", "Throughput_kWh",
]

# PV surfaces, as stored in the RES files. Labels are inferred from the MATLAB names.
SURFACES = [
    ("A_S", "Balcony"),
    ("A_RE", "Roof east"),
    ("A_RW", "Roof west"),
    ("A_ReF", "Flat roof east"),
    ("A_RwF", "Flat roof west"),
]

PROFILES = [
    "Appl", "Appl_EV",
    "Appl_HP_Bestand", "Appl_HP_Waende", "Appl_HP_WaendeUFenster", "Appl_HP_WaendeUFensterUWRG",
    "Appl_HP_EV_Bestand", "Appl_HP_EV_Waende", "Appl_HP_EV_WaendeUFenster",
    "Appl_HP_EV_WaendeUFensterUWRG",
]


@dataclass
class Economics:
    # Energy prices and PV technical
    cost_cur: float = 0.33          # €/kWh purchase
    gain_sell: float = 0.05         # €/kWh feed-in
    eta_pv: float = 0.23            # kWp per m²
    r: float = 0.035                # discount rate

    # PV investment
    inv_pv: float = 1500.0          # €/kWp
    life_pv: int = 20               # years

    # Battery investment
    inv_batt: float = 500.0         # €/kWh
    life_batt: int = 12             # years

    # State (Landesförderung) PV subsidy: flat rate up to a cap
    land_pv_rate: float = 125.0     # €/kWp
    land_pv_cap_kwp: float = 20.0   # kWp

    # Federal (Bundesförderung) PV subsidy: band rates by system size
    bund_pv_upper: tuple = (10.0, 20.0, 100.0, 1000.0)   # upper bounds [kWp]
    bund_pv_rates: tuple = (150.0, 140.0, 130.0, 120.0)  # €/kWp
    bund_pv_mode: str = "whole"     # "whole" or "marginal"

    # Battery subsidies
    land_batt_rate: float = 0.0     # €/kWh
    bund_batt_rate: float = 150.0   # €/kWh
    bund_batt_cap_kwh: float = 50.0 # kWh

    def annuity(self, life: int) -> float:
        if self.r == 0:
            return 1.0 / life
        return 1.0 / ((1.0 - (1.0 + self.r) ** (-life)) / self.r)


# ----------------------------------------------------------------------
# Subsidies
# ----------------------------------------------------------------------
def pv_incentive(kwp, e: Economics) -> np.ndarray:
    """Total PV subsidy [€] for installed kWp (vectorised)."""
    kwp = np.atleast_1d(np.asarray(kwp, dtype=float))
    land = e.land_pv_rate * np.minimum(kwp, e.land_pv_cap_kwp)

    upper = np.asarray(e.bund_pv_upper, dtype=float)
    rates = np.asarray(e.bund_pv_rates, dtype=float)
    lower = np.concatenate([[0.0], upper[:-1]])

    if e.bund_pv_mode == "marginal":
        # Each band pays only for the kWp that fall inside it
        in_band = np.clip(np.minimum(kwp[:, None], upper[None, :]) - lower[None, :], 0, None)
        bund = in_band @ rates
        bund = np.where(kwp > upper[-1], 0.0, bund)
    else:
        # Whole-system: the band rate applies to all installed kWp
        mask = (kwp[:, None] > lower[None, :]) & (kwp[:, None] <= upper[None, :])
        bund = (mask @ rates) * kwp

    return bund + land


def batt_incentive(kwh, e: Economics) -> np.ndarray:
    """Total battery subsidy [€] for installed kWh (vectorised)."""
    kwh = np.atleast_1d(np.asarray(kwh, dtype=float))
    land = e.land_batt_rate * kwh
    bund = e.bund_batt_rate * np.minimum(kwh, e.bund_batt_cap_kwh)
    return bund + land


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------
def load_res(path: str, kind: str) -> pd.DataFrame:
    """Load a RES_*.mat file and keep only the physical columns.

    kind: "pv" or "batt". Economic columns saved by the optimizer are ignored,
    since they are recomputed here from the current inputs.
    """
    try:
        res = loadmat(path)["RES"]
    except NotImplementedError:  # MATLAB v7.3 (HDF5) file
        import h5py
        with h5py.File(path, "r") as f:
            res = np.array(f["RES"]).T
    res = np.asarray(res, dtype=float)
    cols = PV_PHYS_COLS if kind == "pv" else BATT_PHYS_COLS
    return pd.DataFrame(res[:, : len(cols)], columns=cols)


def evaluate(df: pd.DataFrame, kind: str, e: Economics) -> pd.DataFrame:
    """Annualised costs for every design in the sweep."""
    kwp = df[AREA_COLS].to_numpy().sum(axis=1) * e.eta_pv
    inc_pv = pv_incentive(kwp, e)
    grid = df["GridDem_kWh"].to_numpy()
    pv2grid = df["PV2grid_kWh"].to_numpy()

    out = pd.DataFrame({
        "kWp": kwp,
        "GridDem_kWh": grid,
        "PV2grid_kWh": pv2grid,
        "Incentive_PV": inc_pv,
        "EAC_PV": -(kwp * e.inv_pv - inc_pv) * e.annuity(e.life_pv),
        "EAC_op": -grid * e.cost_cur,
        "EAC_sell": pv2grid * e.gain_sell,
    })

    out["PVYield_kWh"] = df["PVYield_kWh"].to_numpy()

    if kind == "pv":
        out["kWh"] = 0.0
        out["Self_kWh"] = df["Self_kWh"].to_numpy()
        out["Incentive_Batt"] = 0.0
        out["EAC_Batt"] = 0.0
    else:
        kwh = df["E_batt_kWh"].to_numpy()
        inc_b = batt_incentive(kwh, e)
        out["kWh"] = kwh
        out["Self_kWh"] = (df["SelfPV_kWh"] + df["SelfBatt_kWh"]).to_numpy()
        out["Incentive_Batt"] = inc_b
        out["EAC_Batt"] = -(kwh * e.inv_batt - inc_b) * e.annuity(e.life_batt)

    out["EAC_total"] = out[["EAC_PV", "EAC_Batt", "EAC_op", "EAC_sell"]].sum(axis=1)
    return out


def optimum(out: pd.DataFrame) -> pd.Series:
    """Design with the highest total EAC (the sweep's best grid point)."""
    return out.loc[out["EAC_total"].idxmax()]


def threshold_note(opt_kwp: float, e: Economics, tol: float = 1.0) -> str | None:
    """Warn if the optimum sits just below a federal cliff (whole-system mode only)."""
    if e.bund_pv_mode != "whole":
        return None
    for t in e.bund_pv_upper:
        gap = t - opt_kwp
        if 0 <= gap <= tol:
            return (f"Optimum {opt_kwp:.2f} kWp is {gap:.2f} kWp below the "
                    f"{t:g} kWp federal threshold. Check whether the grid is too coarse "
                    f"to show the best design at or just below the cliff.")
    return None


def envelope(out: pd.DataFrame, by_battery: bool) -> pd.DataFrame:
    """Best EAC per kWp (and per battery size if by_battery), for plotting."""
    tmp = out.assign(kWp_r=out["kWp"].round(2))
    keys = ["kWh", "kWp_r"] if by_battery else ["kWp_r"]
    return tmp.groupby(keys, as_index=False)["EAC_total"].max()


# ----------------------------------------------------------------------
# Memory-light optimum, full reports and sensitivity analysis
# ----------------------------------------------------------------------
def _kwp_kwh_eac(df: pd.DataFrame, kind: str, e: Economics):
    """Installed kWp, battery kWh and total EAC for every design (three arrays only)."""
    kwp = df[AREA_COLS].to_numpy().sum(axis=1) * e.eta_pv
    grid = df["GridDem_kWh"].to_numpy()
    pv2 = df["PV2grid_kWh"].to_numpy()
    inc_pv = pv_incentive(kwp, e)
    eac = -(kwp * e.inv_pv - inc_pv) * e.annuity(e.life_pv) - grid * e.cost_cur + pv2 * e.gain_sell
    if kind == "batt":
        kwh = df["E_batt_kWh"].to_numpy()
        eac = eac - (kwh * e.inv_batt - batt_incentive(kwh, e)) * e.annuity(e.life_batt)
    else:
        kwh = np.zeros(len(df))
    return kwp, kwh, eac


def optimum_fast(df: pd.DataFrame, kind: str, e: Economics) -> dict:
    """Best design and its full cost breakdown, without building a large table."""
    _, _, eac = _kwp_kwh_eac(df, kind, e)
    i = int(np.argmax(eac))
    out = evaluate(df.iloc[[i]], kind, e).iloc[0].to_dict()
    out.update({c: float(df[c].iloc[i]) for c in AREA_COLS})  # surface areas [m²]
    return out


def _scale_field(base: Economics, field: str, factor: float) -> Economics:
    v = getattr(base, field)
    new = tuple(x * factor for x in v) if isinstance(v, tuple) else v * factor
    if field in ("life_pv", "life_batt"):
        new = max(1, int(round(new)))
    return replace(base, **{field: new})


# (label, field name, display scale, battery-only)
SWEEP_PARAMS = [
    ("Electricity price [€/kWh]", "cost_cur", 1.0, False),
    ("Feed-in tariff [€/kWh]", "gain_sell", 1.0, False),
    ("Discount rate [%]", "r", 100.0, False),
    ("Module efficiency [kWp/m²]", "eta_pv", 1.0, False),
    ("PV investment [€/kWp]", "inv_pv", 1.0, False),
    ("PV lifetime [y]", "life_pv", 1.0, False),
    ("State PV rate [€/kWp]", "land_pv_rate", 1.0, False),
    ("Battery investment [€/kWh]", "inv_batt", 1.0, True),
    ("Battery lifetime [y]", "life_batt", 1.0, True),
    ("Federal battery rate [€/kWh]", "bund_batt_rate", 1.0, True),
]

# (field name, label, battery-only) for the tornado
TORNADO_PARAMS = [(f, lbl, b) for lbl, f, _, b in SWEEP_PARAMS] + [
    ("bund_pv_rates", "Federal PV rates (all bands)", False),
]


def sensitivity_1d(df: pd.DataFrame, kind: str, base: Economics, field: str, values) -> pd.DataFrame:
    """Re-optimise for each value of one parameter."""
    rows = []
    for v in values:
        v = int(round(v)) if field in ("life_pv", "life_batt") else float(v)
        b = optimum_fast(df, kind, replace(base, **{field: v}))
        rows.append({"value": v, **b})
    return pd.DataFrame(rows)


def tornado(df: pd.DataFrame, kind: str, base: Economics, pct: float = 0.2):
    """Change each parameter by -pct and +pct and re-optimise. Returns (table, baseline optimum)."""
    base_opt = optimum_fast(df, kind, base)
    rows = []
    for field, label, batt_only in TORNADO_PARAMS:
        if batt_only and kind == "pv":
            continue
        lo = optimum_fast(df, kind, _scale_field(base, field, 1 - pct))
        hi = optimum_fast(df, kind, _scale_field(base, field, 1 + pct))
        rows.append({
            "Parameter": label,
            "Base value": str(getattr(base, field)),
            "EAC low [€/y]": lo["EAC_total"], "EAC high [€/y]": hi["EAC_total"],
            "Δ low [€/y]": lo["EAC_total"] - base_opt["EAC_total"],
            "Δ high [€/y]": hi["EAC_total"] - base_opt["EAC_total"],
            "kWp low": lo["kWp"], "kWp high": hi["kWp"],
            "kWh low": lo["kWh"], "kWh high": hi["kWh"],
            "Swing [€/y]": abs(hi["EAC_total"] - lo["EAC_total"]),
        })
    table = pd.DataFrame(rows).sort_values("Swing [€/y]", ascending=False, ignore_index=True)
    return table, base_opt


def full_report(profile: str, df_pv, df_batt, e: Economics) -> pd.DataFrame:
    """One row per case (PV only, PV + battery) with the optimum and its breakdown."""
    rows = []
    for case, df, kind in (("PV only", df_pv, "pv"), ("PV + battery", df_batt, "batt")):
        if df is None:
            continue
        b = optimum_fast(df, kind, e)
        kwp, kwh, eac = _kwp_kwh_eac(df, kind, e)
        z = np.flatnonzero((kwp == 0) & (kwh == 0))
        eac0 = float(eac[z[0]]) if z.size else np.nan
        pvy = max(b["PVYield_kWh"], 1e-9)
        selfc = b["Self_kWh"]
        rows.append({
            "Profile": profile, "Case": case,
            "PV [kWp]": b["kWp"], "Battery [kWh]": b["kWh"],
            "Total EAC [€/y]": b["EAC_total"],
            "EAC PV [€/y]": b["EAC_PV"], "EAC battery [€/y]": b["EAC_Batt"],
            "EAC grid purchase [€/y]": b["EAC_op"], "EAC feed-in [€/y]": b["EAC_sell"],
            "EAC without PV/battery [€/y]": eac0,
            "Gain vs without PV/battery [€/y]": b["EAC_total"] - eac0,
            "PV subsidy [€]": b["Incentive_PV"], "Battery subsidy [€]": b["Incentive_Batt"],
            "PV yield [kWh/y]": b["PVYield_kWh"], "Self-consumed [kWh/y]": selfc,
            "Grid demand [kWh/y]": b["GridDem_kWh"], "PV to grid [kWh/y]": b["PV2grid_kWh"],
            "Self-consumption ratio": selfc / pvy,
            "Self-sufficiency ratio": selfc / max(selfc + b["GridDem_kWh"], 1e-9),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Orientation (surface allocation) analysis
# ----------------------------------------------------------------------
def best_mix_vs_size(df: pd.DataFrame, kind: str, e: Economics, bin_kwp: float = 1.0) -> pd.DataFrame:
    """For each total PV size, the best surface allocation and its cost.

    For the battery case the optimal battery size is held fixed, so the curve
    shows allocation and cost for the best battery in the sweep.
    """
    kwp, kwh, eac = _kwp_kwh_eac(df, kind, e)
    if kind == "batt":
        keep = kwh == kwh[int(np.argmax(eac))]
    else:
        keep = np.ones(len(df), dtype=bool)

    data = {"kWp": kwp[keep], "EAC_total": eac[keep]}
    for c, _ in SURFACES:
        data[c] = df[c].to_numpy()[keep] * e.eta_pv   # kWp per surface
    tab = pd.DataFrame(data)
    tab["bin"] = np.round(tab["kWp"] / bin_kwp) * bin_kwp
    best = tab.sort_values("EAC_total", ascending=False).drop_duplicates("bin")
    cols = ["bin", "EAC_total"] + [c for c, _ in SURFACES]
    return best[cols].sort_values("bin", ignore_index=True).rename(columns={"bin": "kWp"})


def roof_ew_grid(df: pd.DataFrame, kind: str, e: Economics, base: dict) -> pd.DataFrame:
    """EAC over east vs west roof area, all other inputs held at the base optimum.

    Returns a pivot: index = west roof [kWp], columns = east roof [kWp], values = EAC [€/y].
    """
    mask = (df["A_S"].to_numpy() == base["A_S"]) \
        & (df["A_ReF"].to_numpy() == base["A_ReF"]) \
        & (df["A_RwF"].to_numpy() == base["A_RwF"])
    if kind == "batt":
        mask &= df["E_batt_kWh"].to_numpy() == base["kWh"]
    sub = df.loc[mask, ["A_RE", "A_RW"]].copy()
    _, _, eac = _kwp_kwh_eac(df.loc[mask], kind, e)
    sub["EAC_total"] = eac
    sub["roof_east_kWp"] = sub["A_RE"] * e.eta_pv
    sub["roof_west_kWp"] = sub["A_RW"] * e.eta_pv
    return sub.pivot_table(index="roof_west_kWp", columns="roof_east_kWp",
                           values="EAC_total", aggfunc="max")


# ----------------------------------------------------------------------
# Graphics data: allocation bars, battery landscape, orientation scatter, daily profile
# ----------------------------------------------------------------------
def allocation_table(profile_dfs: dict, e: Economics) -> pd.DataFrame:
    """Optimal kWp per surface and battery kWh, for PV and PV + battery, all profiles.

    profile_dfs: {profile: (df_pv or None, df_batt or None)}
    """
    rows = []
    for prof in PROFILES:
        if prof not in profile_dfs:
            continue
        df_pv, df_batt = profile_dfs[prof]
        for case, df, kind in (("PV", df_pv, "pv"), ("PV+Batt", df_batt, "batt")):
            if df is None:
                continue
            b = optimum_fast(df, kind, e)
            row = {"Profile": prof, "Case": case, "kWp": b["kWp"], "kWh": b["kWh"],
                   "EAC_total": b["EAC_total"]}
            row.update({lbl: b[c] * e.eta_pv for c, lbl in SURFACES})
            rows.append(row)
    return pd.DataFrame(rows)


def landscape_grid(df_batt: pd.DataFrame, e: Economics):
    """Best EAC over battery sizes for each (PV kWp, battery kWh) pair.

    Returns (kwh_values, kwp_values, Z) with Z[i, j] = best EAC at kwp_values[i], kwh_values[j].
    """
    kwp, kwh, eac = _kwp_kwh_eac(df_batt, "batt", e)
    tmp = pd.DataFrame({"kWp": np.round(kwp, 2), "kWh": kwh, "EAC": eac})
    g = tmp.groupby(["kWp", "kWh"], as_index=False)["EAC"].max()
    piv = g.pivot(index="kWp", columns="kWh", values="EAC")
    return piv.columns.to_numpy(float), piv.index.to_numpy(float), piv.to_numpy()


def orientation_scatter_table(df_pv: pd.DataFrame, e: Economics, kwp_target: float,
                              tol: float = 0.25) -> pd.DataFrame:
    """Designs with total kWp near kwp_target, with roof orientation index and KPIs.

    OW index: -1 = all roof west, +1 = all roof east (roof surfaces only).
    Balcony share: balcony kWp / total kWp.
    """
    kwp, _, eac = _kwp_kwh_eac(df_pv, "pv", e)
    m = np.abs(kwp - kwp_target) <= tol
    sub = df_pv.loc[m]
    re, rw = sub["A_RE"].to_numpy(), sub["A_RW"].to_numpy()
    roof = re + rw
    with np.errstate(invalid="ignore", divide="ignore"):
        ow = np.where(roof > 0, (re - rw) / roof, np.nan)
    total_area = sub[AREA_COLS].to_numpy().sum(axis=1)
    balcony_share = np.where(total_area > 0, sub["A_S"].to_numpy() / total_area, np.nan)
    pvy = sub["PVYield_kWh"].to_numpy()
    self_kwh = sub["Self_kWh"].to_numpy()
    return pd.DataFrame({
        "OW": ow,
        "Balcony share": balcony_share,
        "PV yield [kWh/y]": pvy,
        "Self-consumption [%]": 100 * self_kwh / np.maximum(pvy, 1e-9),
        "PV to grid [kWh/y]": sub["PV2grid_kWh"].to_numpy(),
        "Grid demand [kWh/y]": sub["GridDem_kWh"].to_numpy(),
        "EAC [€/y]": eac[m],
    }).dropna(subset=["OW"]).reset_index(drop=True)


def load_appl_profile(path: str, name: str) -> np.ndarray:
    """Household load for one profile in kW, 15-minute steps (from Appl.mat, stored in W)."""
    return np.asarray(loadmat(path)[name], dtype=float).ravel() / 1000.0


def daily_profile(pv_per_m2: np.ndarray, areas_m2: dict, load_kw: np.ndarray,
                  start: int, n_steps: int) -> dict:
    """PV output (kW) and load (kW) for a window of 15-minute steps.

    pv_per_m2: array (T, 5) with columns S, RE, RW, RfE, RfW in kW per m² (inverter included).
    areas_m2: {"S": m², "RE": m², ...}
    """
    order = ["S", "RE", "RW", "RfE", "RfW"]
    sl = slice(start, start + n_steps)
    pv = sum(pv_per_m2[sl, i] * areas_m2[k] for i, k in enumerate(order))
    return {"pv_kw": pv, "load_kw": load_kw[sl]}
