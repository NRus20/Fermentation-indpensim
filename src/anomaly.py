"""Stage 4 - anomaly detection / process monitoring.

Approach: (1) remove the nominal trajectory (mean of FAULT-FREE TRAINING batches at each hour) because
batch processes are strongly non-stationary; (2) fit PCA (Hotelling T2 + SPE/Q) and Isolation Forest on the
training residuals; (3) alarm when the statistic exceeds its training 99th percentile for k consecutive samples.
Train/test are split by batch; every fault batch is test-only. Repeated over random splits of the fault-free batches.
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from . import viz
from .data import available_process_vars, batch_labels

SEED = 42


class Monitor:
    def __init__(self, feats, n_var=.9, q=.99, k=5):
        self.feats, self.n_var, self.q, self.k = feats, n_var, q, k

    def _resid(self, d):
        bins = d.time.round().clip(0, self.max_bin).astype(int)
        mu = self.traj.reindex(bins.values).ffill().bfill().values
        return (d[self.feats].values - mu) / self.sd

    def fit(self, tr):
        self.max_bin = int(tr.time.round().max()); b = tr.time.round().astype(int)
        self.traj = tr.groupby(b)[self.feats].mean().reindex(range(self.max_bin + 1)).interpolate().ffill().bfill()
        self.sd = 1.0; R = self._resid(tr); self.sd = np.nanstd(R, 0) + 1e-9; R = self._resid(tr)
        self.imp = np.nanmedian(R, 0)
        R = np.where(np.isnan(R), self.imp, R)
        self.pca = PCA(n_components=self.n_var, svd_solver="full").fit(R)
        t2, spe = self._stats(R); self.lim = {"t2": np.quantile(t2, self.q), "spe": np.quantile(spe, self.q)}
        self.iso = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1).fit(R[::2])
        self.lim["iso"] = np.quantile(-self.iso.score_samples(R), self.q); return self

    def _stats(self, R):
        Z = self.pca.transform(R); t2 = ((Z ** 2) / self.pca.explained_variance_).sum(1)
        spe = ((R - self.pca.inverse_transform(Z)) ** 2).sum(1); return t2, spe

    def score(self, d):
        R = self._resid(d); R = np.where(np.isnan(R), self.imp, R); t2, spe = self._stats(R)
        return pd.DataFrame({"t2": t2, "spe": spe, "iso": -self.iso.score_samples(R)}, index=d.index), R

    def alarm(self, s, method):
        ex = ((s.t2 > self.lim["t2"]) | (s.spe > self.lim["spe"])) if method == "pca" else (s.iso > self.lim["iso"])
        return ex.astype(int).rolling(self.k).sum() >= self.k  # k consecutive exceedances

    def combined(self, s, method):
        """Continuous anomaly score (for AUC): limit-normalised T2+SPE, or isolation score."""
        return s.t2 / self.lim["t2"] + s.spe / self.lim["spe"] if method == "pca" else s.iso


def _onset(d):
    f = d.index[d["fault_ref"] > 0] if "fault_ref" in d else []
    return d.time.loc[f[0]] if len(f) else np.nan


def evaluate(df, n_splits=10, train_frac=.7, k=5):
    lab = batch_labels(df); feats = [f for f in available_process_vars(df) if df[f].nunique() > 3]
    normal = lab.index[~lab.has_fault].values; faulty = lab.index[lab.has_fault].values
    if len(faulty) == 0:
        print("No fault batches found; skipping anomaly evaluation"); return None, None
    rows = []; rng = np.random.RandomState(SEED); mon = None
    for rep in range(n_splits):
        tr_b = rng.choice(normal, int(len(normal) * train_frac), replace=False)
        te_norm = np.setdiff1d(normal, tr_b)                       # held-out fault-free batches
        mon = Monitor(feats, k=k).fit(df[df.batch_id.isin(tr_b)])  # training = fault-free batches only
        cache = {}
        for b in np.concatenate([te_norm, faulty]):
            d = df[df.batch_id == b].reset_index(drop=True); cache[b] = (d, mon.score(d)[0])
        for method in ["pca", "iso"]:
            false_alarm = pre_onset = det = 0; delays = []; ys = []; sc = []
            for b in np.concatenate([te_norm, faulty]):
                d, s = cache[b]
                a = mon.alarm(s, method).values; sc += list(mon.combined(s, method))
                if b in faulty:
                    on = _onset(d); ys += list((d["fault_ref"] > 0).astype(int))
                    pre = a & (d.time.values < on); post = a & (d.time.values >= on)
                    pre_onset += int(pre.any())
                    if post.any(): det += 1; delays.append(d.time.values[post][0] - on)
                else:
                    ys += [0] * len(d); false_alarm += int(a.any())
            rows.append(dict(rep=rep, method=method, batch_FPR=false_alarm / len(te_norm), batch_TPR=det / len(faulty),
                             prefault_alarm_rate=pre_onset / len(faulty),
                             median_delay_h=np.median(delays) if delays else np.nan,
                             sample_AUC=roc_auc_score(ys, sc) if 0 < np.mean(ys) < 1 else np.nan))
    r = pd.DataFrame(rows); r.to_csv(viz.TAB / "anomaly_repeats.csv", index=False)
    summ = r.drop(columns="rep").groupby("method").agg(["mean", "std"]); summ.to_csv(viz.TAB / "anomaly_summary.csv")
    _plots(df, mon, faulty, normal, feats); return r, summ


def _plots(df, mon, faulty, normal, feats):
    viz.setup(); fb = faulty[0]; nb = normal[0]
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex="col")
    for j, b in enumerate([nb, fb]):
        d = df[df.batch_id == b].reset_index(drop=True); s, R = mon.score(d)
        axes[0, j].plot(d.time, s.t2, label="T2"); axes[0, j].axhline(mon.lim["t2"], color="C0", ls="--")
        axes[1, j].plot(d.time, s.spe, "C1", label="SPE"); axes[1, j].axhline(mon.lim["spe"], color="C1", ls="--")
        on = _onset(d) if b in faulty else np.nan
        for ax in axes[:, j]:
            if not np.isnan(on): ax.axvline(on, color="red", lw=1.5, label="fault onset")
        axes[0, j].set_title(f"Batch {b} ({'FAULT' if b in faulty else 'fault-free'})"); axes[1, j].set_xlabel("Time (h)")
        axes[0, j].set_yscale("log"); axes[1, j].set_yscale("log")
    axes[0, 0].set_ylabel("Hotelling T2"); axes[1, 0].set_ylabel("SPE (Q)"); axes[0, 1].legend()
    fig.suptitle("PCA monitoring charts (dashed = 99% training limit)"); viz.save(fig, "10_monitoring_charts")
    # variable contributions to SPE after fault onset
    d = df[df.batch_id == fb].reset_index(drop=True); on = _onset(d); _, R = mon.score(d)
    Z = mon.pca.transform(R); E = (R - mon.pca.inverse_transform(Z)) ** 2; m = d.time >= on
    c = pd.Series(E[m.values].mean(0), index=feats).sort_values(); c.to_csv(viz.TAB / "anomaly_spe_contributions.csv", header=["mean_sq_resid"])
    fig, ax = plt.subplots(figsize=(6, 5)); c.plot.barh(ax=ax, color="#C44E52"); ax.set_xlabel("Mean SPE contribution after fault onset")
    ax.set_title(f"Which variables flag batch {fb}?"); viz.save(fig, "11_fault_contributions")
