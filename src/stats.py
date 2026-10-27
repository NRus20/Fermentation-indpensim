"""Stage 2 - statistical analysis. All p-value families are multiplicity-corrected."""
import itertools, numpy as np, pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from . import viz
from .data import available_process_vars, batch_summary


def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    ranked = p[o] * n / (np.arange(n) + 1); q[o] = np.minimum.accumulate(ranked[::-1])[::-1]
    return np.clip(q, 0, 1)


def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); adj = np.empty(n)
    run = 0
    for i, idx in enumerate(o):
        run = max(run, (n - i) * p[idx]); adj[idx] = min(run, 1)
    return adj


def rank_biserial(a, b):
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return 2 * u / (len(a) * len(b)) - 1


def strategy_comparison(bs, target="final_penicillin"):
    groups = {s: d[target].dropna().values for s, d in bs.groupby("strategy") if len(d) >= 3}
    kw = stats.kruskal(*groups.values()) if len(groups) > 1 else None
    rows = []
    for a, b in itertools.combinations(groups, 2):
        p = stats.mannwhitneyu(groups[a], groups[b], alternative="two-sided").pvalue
        rows.append(dict(group_a=a, group_b=b, median_a=np.median(groups[a]), median_b=np.median(groups[b]),
                         rank_biserial=rank_biserial(groups[a], groups[b]), p_raw=p))
    t = pd.DataFrame(rows)
    if len(t): t["p_holm"] = holm(t.p_raw)
    t.to_csv(viz.TAB / "stats_strategy_pairwise.csv", index=False)
    return {"kruskal_H": getattr(kw, "statistic", None), "kruskal_p": getattr(kw, "pvalue", None)}, t


def batch_feature_correlations(bs, exclude_fault=True, target="final_penicillin"):
    d = bs[~bs.has_fault] if exclude_fault else bs
    feats = [c for c in d.columns if c.endswith(("_mean", "_std", "_min", "_max"))]
    rows = []
    for f in feats:
        x = d[f]; ok = x.notna() & d[target].notna()
        if ok.sum() < 10 or x[ok].nunique() < 3: continue
        r, p = stats.spearmanr(x[ok], d[target][ok]); rows.append((f, r, p, ok.sum()))
    t = pd.DataFrame(rows, columns=["feature", "spearman_rho", "p_raw", "n"]).sort_values("p_raw")
    t["q_bh"] = bh_fdr(t.p_raw); t.to_csv(viz.TAB / "stats_batch_feature_vs_yield.csv", index=False)
    viz.setup(); top = t.reindex(t.spearman_rho.abs().sort_values(ascending=False).index).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top.feature, top.spearman_rho, color=np.where(top.q_bh < .05, "#4C72B0", "#bbb"))
    ax.set_xlabel(f"Spearman rho with {target}"); ax.set_title("Batch-level features vs yield\n(blue: BH-FDR q<0.05; fault-free batches)")
    viz.save(fig, "06_batch_feature_correlations"); return t


def within_batch_correlations(df, step=5):
    """Spearman between each variable and (a) penicillin level, (b) penicillin production rate,
    computed WITHIN each batch then summarised across batches -> removes between-batch confounding."""
    pv = available_process_vars(df); rows = []
    for b, d in df.groupby("batch_id"):
        d = d.iloc[::step].copy(); d["rate"] = d.penicillin.diff() / d.time.diff(); d = d.dropna(subset=["rate"])
        for v in pv:
            if d[v].nunique() < 3: continue
            rows.append((b, v, stats.spearmanr(d[v], d.penicillin)[0], stats.spearmanr(d[v], d.rate)[0]))
    r = pd.DataFrame(rows, columns=["batch_id", "var", "rho_level", "rho_rate"])
    out = []
    for v, d in r.groupby("var"):
        for col in ["rho_level", "rho_rate"]:
            x = d[col].dropna(); w = stats.wilcoxon(x) if len(x) > 5 and (x != 0).any() else None
            out.append(dict(var=v, target=col, median_rho=x.median(), frac_positive=(x > 0).mean(), n=len(x),
                            p_raw=getattr(w, "pvalue", np.nan)))
    o = pd.DataFrame(out); o["q_bh"] = bh_fdr(o.p_raw.fillna(1)); o.to_csv(viz.TAB / "stats_within_batch.csv", index=False)
    viz.setup(); fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, col, t in zip(axes, ["rho_level", "rho_rate"], ["vs penicillin level", "vs production rate (dP/dt)"]):
        order = o[o.target == col].sort_values("median_rho").iloc[:, 0].tolist()
        ax.boxplot([r[(r["var"] == v)][col].dropna() for v in order], vert=False, tick_labels=order, showfliers=False)
        ax.axvline(0, color="k", lw=.8); ax.set_title(f"Within-batch Spearman {t}"); ax.set_xlabel("rho (one value per batch)")
    viz.save(fig, "07_within_batch_correlations"); return o


def mixed_model(df, step=10):
    """Random-intercept model: P ~ standardised process vars + (1|batch). Skipped if statsmodels missing."""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("statsmodels not installed - skipping mixed model"); return None
    pv = [v for v in available_process_vars(df) if df[v].nunique() > 3]
    d = df.iloc[::step][["batch_id", "time", "penicillin"] + pv].dropna().copy()
    d[pv] = (d[pv] - d[pv].mean()) / d[pv].std()
    m = smf.mixedlm("penicillin ~ " + " + ".join(pv) + " + time", d, groups=d["batch_id"]).fit(method="lbfgs")
    t = pd.DataFrame({"coef": m.params, "se": m.bse, "p": m.pvalues}).drop(index="Group Var", errors="ignore")
    t.to_csv(viz.TAB / "stats_mixed_model.csv"); return t


def fault_vs_normal(bs, target="final_penicillin"):
    f = bs[bs.has_fault][target].dropna(); n = bs[~bs.has_fault][target].dropna()
    if len(f) < 3: return {}
    res = dict(n_fault=len(f), n_normal=len(n), median_fault=f.median(), median_normal=n.median(),
               p=stats.mannwhitneyu(f, n).pvalue, rank_biserial=rank_biserial(f, n))
    pd.Series(res).to_csv(viz.TAB / "stats_fault_vs_normal.csv"); return res


def run(df, yields=None, target="final_penicillin"):
    bs = batch_summary(df, None, yields); out = {"target": target}
    out["strategy_test"], _ = strategy_comparison(bs, target)
    out["fault_vs_normal"] = fault_vs_normal(bs, target)
    batch_feature_correlations(bs, target=target); within_batch_correlations(df); mixed_model(df)
    return out
