"""Stage 1 - exploratory analysis."""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from . import viz
from .data import available_process_vars, batch_labels

LABELS = {"penicillin": "Penicillin (g/L)", "biomass": "Biomass (g/L)", "substrate": "Substrate (g/L)",
          "ph": "pH", "temp": "Temperature (K)", "do2": "Dissolved O2 (mg/L)", "our": "OUR",
          "cer": "CER", "aeration": "Aeration (L/h)", "agitation": "Agitation (RPM)",
          "sugar_feed": "Sugar feed (L/h)", "viscosity": "Viscosity", "volume": "Volume (L)",
          "co2_off": "Off-gas CO2"}


def overview(df) -> pd.DataFrame:
    lab = batch_labels(df)
    g = df.groupby("batch_id")
    ov = pd.DataFrame({"n_samples": g.size(), "duration_h": g["time"].max(),
                       "final_penicillin": g["penicillin"].last(), "max_penicillin": g["penicillin"].max()}).join(lab)
    summ = ov.groupby("strategy").agg(n_batches=("n_samples", "size"), duration_h=("duration_h", "mean"),
                                      final_P_mean=("final_penicillin", "mean"),
                                      final_P_sd=("final_penicillin", "std"))
    miss = df.isna().mean().sort_values(ascending=False).rename("frac_missing").to_frame()
    summ.to_csv(viz.TAB / "eda_strategy_overview.csv"); ov.to_csv(viz.TAB / "eda_batch_overview.csv")
    miss.to_csv(viz.TAB / "eda_missingness.csv")
    return ov


def _binned(df, var, grp, width=1.0):
    d = df[["time", var]].copy(); d["grp"] = grp; d["t"] = (d["time"] / width).round() * width
    return d.groupby(["grp", "t"])[var].agg(["median", lambda s: s.quantile(.25), lambda s: s.quantile(.75)]).reset_index().set_axis(
        ["grp", "t", "med", "q1", "q3"], axis=1)


def plot_trajectories(df):
    viz.setup()
    lab = batch_labels(df); strat = df["batch_id"].map(lab["strategy"])
    vars_ = ["penicillin"] + available_process_vars(df)
    n = len(vars_); ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.2 * nrow), squeeze=False)
    for ax, v in zip(axes.ravel(), vars_):
        b = _binned(df, v, strat)
        for s, d in b.groupby("grp"):
            c = viz.STRAT_COLORS.get(s, "k")
            ax.plot(d.t, d.med, color=c, label=s, lw=1.5); ax.fill_between(d.t, d.q1, d.q3, color=c, alpha=.15)
        ax.set_title(LABELS.get(v, v)); ax.set_xlabel("Time (h)")
    for ax in axes.ravel()[n:]: ax.axis("off")
    axes[0, 0].legend(title="strategy")
    fig.suptitle("Median and IQR of each variable over time, by control strategy", y=1.002)
    viz.save(fig, "01_trajectories_by_strategy")


def plot_penicillin_batches(df):
    viz.setup(); lab = batch_labels(df)
    fig, ax = plt.subplots(figsize=(9, 5))
    for b, d in df.groupby("batch_id"):
        s = lab.loc[b, "strategy"]; ax.plot(d.time, d.penicillin, color=viz.STRAT_COLORS.get(s, "k"), alpha=.35, lw=.8)
    for s, c in viz.STRAT_COLORS.items():
        if s in set(lab.strategy): ax.plot([], [], color=c, label=s)
    ax.set_xlabel("Time (h)"); ax.set_ylabel("Penicillin (g/L)"); ax.legend(title="strategy")
    ax.set_title("All batches: penicillin concentration")
    viz.save(fig, "02_penicillin_all_batches")


def plot_yield_by_strategy(ov):
    viz.setup(); fig, ax = plt.subplots(figsize=(7, 5))
    order = [s for s in ["recipe", "operator", "apc_raman", "fault"] if s in set(ov.strategy)] or None
    sns.boxplot(data=ov, x="strategy", y="final_penicillin", order=order, hue="strategy", palette=viz.STRAT_COLORS, legend=False, ax=ax, showfliers=False)
    sns.stripplot(data=ov, x="strategy", y="final_penicillin", order=order, color="k", alpha=.5, size=3, ax=ax)
    ax.set_ylabel("Final penicillin (g/L)"); ax.set_title("Final yield by control strategy")
    viz.save(fig, "03_final_yield_by_strategy")


def plot_fault_vs_normal(df):
    if "fault_flag" not in df: return
    viz.setup(); lab = batch_labels(df); fb = lab.index[lab.has_fault]
    if len(fb) == 0: return
    vars_ = [v for v in ["ph", "do2", "our", "temp", "biomass", "penicillin"] if v in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(15, 6.5)); 
    for ax, v in zip(axes.ravel(), vars_):
        for b, d in df.groupby("batch_id"):
            f = b in fb; ax.plot(d.time, d[v], color="#C44E52" if f else "#999", alpha=.7 if f else .15, lw=1 if f else .6)
        ax.set_title(LABELS.get(v, v)); ax.set_xlabel("Time (h)")
    fig.suptitle("Fault batches (red) vs fault-free batches (grey)")
    viz.save(fig, "04_fault_vs_normal")


def plot_correlations(df):
    viz.setup(); pv = available_process_vars(df) + ["penicillin"]
    c = df[pv].corr(method="spearman"); c.to_csv(viz.TAB / "eda_spearman_pooled.csv")
    fig, ax = plt.subplots(figsize=(8, 7)); sns.heatmap(c, cmap="vlag", center=0, annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax)
    ax.set_title("Pooled Spearman correlation (confounded by time - see Stage 2 for within-batch)")
    viz.save(fig, "05_correlation_pooled")


def run(df):
    ov = overview(df); plot_trajectories(df); plot_penicillin_batches(df)
    plot_yield_by_strategy(ov); plot_fault_vs_normal(df); plot_correlations(df)
    return ov
