import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

from src.data import load_indpensim, batch_labels, available_process_vars
from src.anomaly import Monitor

DATA = "data/raw/100_Batches_IndPenSim_V3_reconstructed.csv"
OUT = Path("results/tables")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
N_SPLITS = 10
TRAIN_FRAC = 0.7

QUANTILES = [0.95, 0.97, 0.99, 0.995, 0.999]
K_VALUES = [3, 5, 8, 12]

print("Loading dataset...", flush=True)
df, _ = load_indpensim(DATA)

labels = batch_labels(df)
normal = labels.index[~labels["has_fault"]].to_numpy()
faulty = labels.index[labels["has_fault"]].to_numpy()

features = [
    f for f in available_process_vars(df)
    if df[f].nunique() > 3
]

batches = {
    b: d.reset_index(drop=True)
    for b, d in df.groupby("batch_id")
}

rng = np.random.RandomState(SEED)
rows = []

for split in range(N_SPLITS):

    train_ids = rng.choice(
        normal,
        size=int(len(normal) * TRAIN_FRAC),
        replace=False
    )

    test_normal = np.setdiff1d(normal, train_ids)

    train_df = df[df["batch_id"].isin(train_ids)]

    # Fit the global PCA once per split.
    # Thresholds will be recalibrated from training scores.
    model = Monitor(features, q=0.99, k=5).fit(train_df)

    # Training residual scores, never test scores.
    R = model._resid(train_df)
    R = np.where(np.isnan(R), model.imp, R)

    t2_train, spe_train = model._stats(R)

    # Score held-out batches once per split.
    test_ids = np.concatenate([test_normal, faulty])

    scores = {
        b: model.score(batches[b])[0]
        for b in test_ids
    }

    for q in QUANTILES:

        model.lim["t2"] = np.quantile(t2_train, q)
        model.lim["spe"] = np.quantile(spe_train, q)

        for k in K_VALUES:

            model.k = k

            false_alarms = 0
            detected_faults = 0
            delays = []
            pre_fault_alarms = 0

            for b in test_normal:
                alarms = model.alarm(
                    scores[b], "pca"
                ).to_numpy(dtype=bool)

                false_alarms += int(alarms.any())

            for b in faulty:

                d = batches[b]
                alarms = model.alarm(
                    scores[b], "pca"
                ).to_numpy(dtype=bool)

                times = d["time"].to_numpy()

                onset = d.loc[
                    d["fault_ref"] > 0, "time"
                ].iloc[0]

                pre = alarms & (times < onset)
                post = alarms & (times >= onset)

                pre_fault_alarms += int(pre.any())

                if post.any():
                    detected_faults += 1
                    delays.append(
                        times[post][0] - onset
                    )

            rows.append({
                "split": split,
                "threshold_percentile": q * 100,
                "k": k,
                "batch_FPR": false_alarms / len(test_normal),
                "fault_TPR": detected_faults / len(faulty),
                "prefault_alarm_rate": pre_fault_alarms / len(faulty),
                "median_delay_h": (
                    np.median(delays)
                    if delays else np.nan
                ),
            })

    print(
        f"Completed split {split + 1}/{N_SPLITS}",
        flush=True
    )

results = pd.DataFrame(rows)

summary = (
    results.groupby(["threshold_percentile", "k"])
    .agg(
        mean_FPR=("batch_FPR", "mean"),
        std_FPR=("batch_FPR", "std"),
        mean_TPR=("fault_TPR", "mean"),
        mean_prefault_rate=("prefault_alarm_rate", "mean"),
        median_delay_h=("median_delay_h", "median")
    )
    .reset_index()
)

# Rank by low false-positive rate, then high detection.
# Inspect the trade-off rather than automatically declaring a winner.
ranked = summary.sort_values(
    ["mean_FPR", "mean_TPR", "median_delay_h"],
    ascending=[True, False, True]
)

results.to_csv(
    OUT / "pca_threshold_optimization_splits.csv",
    index=False
)

ranked.to_csv(
    OUT / "pca_threshold_optimization_summary.csv",
    index=False
)

print("\n=== ALL CONFIGURATIONS ===")
print(
    summary.sort_values(
        ["threshold_percentile", "k"]
    ).round(3).to_string(index=False)
)

print("\n=== LOWEST FALSE-POSITIVE CONFIGURATIONS ===")
print(
    ranked.head(10).round(3).to_string(index=False)
)

print("\nSaved optimization tables.")
