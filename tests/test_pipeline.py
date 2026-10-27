import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from src import synthetic
from src.data import resolve_columns, batch_summary, batch_labels, load_indpensim
from src.models import batch_models


def test_resolution_and_no_offline_leak(tmp_path):
    p = tmp_path / "s.csv"; synthetic.make(n_batches=100, dt=2)[0].to_csv(p, index=False)
    df, m = load_indpensim(p)
    assert {"batch_id", "time", "penicillin", "ph", "do2", "co2_off", "acid", "paa_flow", "fault_flag"} <= set(m)
    # regression tests for the two real-header traps
    assert m["op_ctrl"].startswith("0 - Recipe driven") and m["pat_ctrl"].startswith("2-PAT control")
    assert "biomass" not in m and "viscosity" not in m   # only offline versions exist
    assert all("offline" not in v.lower() for v in m.values())          # offline assays never used
    assert not any(str(c).isdigit() for c in df.columns)                # Raman dropped


def _loaded(tmp_path, dt=2):
    p = tmp_path / "s.csv"; synthetic.make(n_batches=100, dt=dt)[0].to_csv(p, index=False); return load_indpensim(p)[0]


def test_labels_and_early_features_do_not_peek(tmp_path):
    df = _loaded(tmp_path); lab = batch_labels(df)
    assert set(lab.strategy) == {"recipe", "operator", "apc_raman", "fault"}
    a = batch_summary(df, 48); df2 = df.copy(); df2.loc[df2.time > 48, "ph"] = 999   # corrupt the future
    b = batch_summary(df2, 48); pd.testing.assert_frame_equal(a.drop(columns="final_penicillin"), b.drop(columns="final_penicillin"))


def test_models_run(tmp_path):
    df = _loaded(tmp_path); bs = batch_summary(df, 96)
    X = bs[[c for c in bs.columns if c.endswith("_mean")]]; y = bs.final_penicillin
    for m in batch_models().values(): m.fit(X, y)


def test_alignment_guard_rejects_shifted_columns(tmp_path):
    import pytest
    d = synthetic.make(n_batches=100, dt=2)[0]; d["Fault flag"] = 2.5   # impossible flag value -> shifted columns
    p = tmp_path / "bad.csv"; d.to_csv(p, index=False)
    with pytest.raises(ValueError): load_indpensim(p)
