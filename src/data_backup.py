"""Loading and schema handling for IndPenSim.

IndPenSim column names are long and contain units, e.g. 'Penicillin concentration(P:g/L)'.
Rather than hard-coding them, we resolve canonical names with regex patterns and
FAIL LOUDLY (printing the actual columns) if something can't be found.
Run `python scripts/00_inspect_data.py` first and adjust ALIASES if needed.
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd

# canonical name -> regex (case-insensitive, matched with re.search on the raw column name)
ALIASES = {
    "batch_id":   r"^batch id",
    "time":       r"^time\s*\(h\)|^time",
    "penicillin": r"^penicillin concentration",
    "biomass":    r"^biomass concentration",
    "substrate":  r"^substrate concentration",
    "ph":         r"^ph\(|^ph$",
    "temp":       r"^temperature",
    "do2":        r"^dissolved oxygen concentration",
    "our":        r"^oxygen uptake rate",
    "cer":        r"^carbon evolution rate",
    "aeration":   r"^aeration rate",
    "agitation":  r"^agitator rpm",
    "sugar_feed": r"^sugar feed rate",
    "viscosity":  r"^viscosity",
    "volume":     r"^vessel volume",
    "co2_off":    r"^carbon dioxide percent in off-gas",
    "o2_off":     r"^oxygen in percent in off-gas",
    "acid":       r"^acid flow",
    "base":       r"^base flow",
    "cooling":    r"^heating/cooling water flow",
    "heating":    r"^heating water flow",
    "water_inj":  r"^water for injection",
    "pressure":   r"^air head pressure",
    "dumped":     r"^dumped broth flow",
    "weight":     r"^vessel weight",
    "heat_gen":   r"^generated heat",
    "paa_flow":   r"^paa flow",
    "oil":        r"^oil flow",
    "nh3_shots":  r"^ammonia shots",
    "fault_flag": r"^fault flag",
    "fault_ref":  r"^fault reference",
    # real header: '2-PAT control(PAT_ref:PAT ref)' (1 = PAT/APC) and
    #              '0 - Recipe driven 1 - Operator controlled(Control_ref:Control ref)' (1 = operator)
    "pat_ctrl":   r"^2-pat control",
    "op_ctrl":    r"recipe driven.*operator controlled",
    "raman_flag": r"^1-\s*raman spec recorded",
    "batch_ref":  r"^batch reference",
}
# Process variables used as model / monitoring features (resolved subset of ALIASES keys)
# NB: the real file has only OFFLINE biomass/viscosity/PAA/NH3 (sparse lab assays) -> excluded by design.
PROCESS_VARS = ["substrate", "ph", "temp", "do2", "our", "cer", "aeration", "agitation", "sugar_feed",
                "volume", "weight", "co2_off", "o2_off", "acid", "base", "cooling", "heating", "water_inj",
                "pressure", "dumped", "heat_gen", "paa_flow", "oil", "nh3_shots"]

# Documented IndPenSim batch layout (used ONLY as a fallback; verify against the data).
FALLBACK_STRATEGY = {range(1, 31): "recipe", range(31, 61): "operator",
                     range(61, 91): "apc_raman", range(91, 101): "fault"}


def is_spectral(col: str) -> bool:
    try:
        float(str(col).strip()); return True
    except ValueError:
        return False


def resolve_columns(columns) -> dict:
    cols = [c for c in columns if not is_spectral(c)]
    found = {}
    for key, pat in ALIASES.items():
        hits = [c for c in cols if re.search(pat, c.strip(), flags=re.I)]
        hits = [h for h in hits if "offline" not in h.lower()]  # never pick offline/lab columns
        if hits:
            found[key] = sorted(hits, key=len)[0]
    return found


def load_indpensim(path, keep_raman=False) -> tuple[pd.DataFrame, dict]:
    """Returns (df with canonical column names, mapping canonical->raw name).
    Raman columns (~2200 of them) are dropped unless keep_raman=True."""
    path = Path(path)
    header = pd.read_csv(path, nrows=0).columns
    spectral = [c for c in header if is_spectral(c)]
    mapping = resolve_columns(header)
    required = ["batch_id", "time", "penicillin"]
    missing = [k for k in required if k not in mapping]
    if missing:
        raise KeyError(f"Could not resolve {missing}. Non-spectral columns seen: "
                       f"{[c for c in header if not is_spectral(c)][:60]} ... edit ALIASES in src/data.py")
    usecols = list(mapping.values()) + (spectral if keep_raman else [])
    df = pd.read_csv(path, usecols=usecols)
    df = df.rename(columns={v: k for k, v in mapping.items()})
    # alignment guards: rows with a different field count than the header would shift columns silently
    bid = pd.to_numeric(df["batch_id"], errors="coerce")
    if bid.isna().any() or (bid % 1 != 0).any():
        raise ValueError("'Batch ID' has missing/non-integer values: data rows probably have a different number of "
                         "fields than the header (shifted columns). Run scripts/00_inspect_data.py")
    if "fault_flag" in df and not set(pd.unique(df["fault_flag"].dropna())) <= {0, 1}:
        raise ValueError("'Fault flag' is not 0/1 - columns look shifted. Run scripts/00_inspect_data.py")
    df["batch_id"] = bid.astype(int)
    df = df.sort_values(["batch_id", "time"]).reset_index(drop=True)
    return df, mapping


def available_process_vars(df) -> list[str]:
    return [v for v in PROCESS_VARS if v in df.columns]


def _fallback(bid):
    for r, name in FALLBACK_STRATEGY.items():
        if bid in r: return name
    return "unknown"


def batch_labels(df) -> pd.DataFrame:
    """One row per batch: control strategy and fault status.
    Strategy: any fault flag -> 'fault'; else operator if op_ctrl==1; else apc_raman if pat_ctrl==1;
    else recipe. Falls back to documented ID ranges if the flag columns are missing."""
    g = df.groupby("batch_id")
    lab = pd.DataFrame(index=g.size().index)
    if "op_ctrl" in df and "pat_ctrl" in df:
        op = g["op_ctrl"].max() > 0
        pat = g["pat_ctrl"].max() > 0
        lab["strategy"] = np.select([lab.has_fault, op, pat], ["fault", "operator", "apc_raman"], "recipe")
        lab["has_fault"] = (g["fault_ref"].max() > 0) if "fault_ref" in df.columns else False                                                                                                                 
        expected = pd.Series([_fallback(b) for b in lab.index], index=lab.index)
      o  n_diff = int((lab.strategy != expected).sum())
        if n_diff:
            print(f"WARNING: {n_diff} batches have a strategy label that differs from the documented ID ranges "
                  f"(1-30 recipe, 31-60 operator, 61-90 APC, 91-100 fault). Inspect results/tables/eda_batch_overview.csv.")
    else:
        lab["strategy"] = [_fallback(b) for b in lab.index]
    return lab


def load_yields(path) -> pd.DataFrame:
    """100_Batches_IndPenSim_Statistics.csv -> per-batch total penicillin yield (tonnes) and fault flag."""
    d = pd.read_csv(path)
    col = lambda pat: next(c for c in d.columns if re.search(pat, c, flags=re.I))
    out = pd.DataFrame({"batch_id": d[col(r"^batch")].astype(int),
                        "total_yield_t": d[col(r"yield_total")] / 1e3,
                        "fault_stats": d[col(r"^fault")].astype(int)}).set_index("batch_id")
    return out


def batch_summary(df, horizon_h=None, yields=None) -> pd.DataFrame:
    """Batch-level feature table. If horizon_h is given, features use ONLY data with time<=horizon_h
    (early prediction of final yield). Target is always the final-sample penicillin concentration."""
    pv = available_process_vars(df)
    target = df.groupby("batch_id")["penicillin"].last().rename("final_penicillin")
    sub = df if horizon_h is None else df[df["time"] <= horizon_h]
    g = sub.groupby("batch_id")[pv]
    feats = pd.concat([g.mean().add_suffix("_mean"), g.std().add_suffix("_std"),
                       g.min().add_suffix("_min"), g.max().add_suffix("_max")], axis=1)
    out = feats.join(target)
    out["duration_h"] = df.groupby("batch_id")["time"].max()
    out = out.join(batch_labels(df))
    if yields is not None: out = out.join(yields[["total_yield_t"]])
    return out
