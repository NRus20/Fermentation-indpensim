from pathlib import Path
import numpy as np
import pandas as pd

from src.data import load_indpensim, batch_labels, available_process_vars
from src.anomaly import Monitor

DATA = "data/raw/100_Batches_IndPenSim_V3_reconstructed.csv"
OUT = Path("results/tables")
OUT.mkdir(parents=True, exist_ok=True)

N_SPLITS = 20
TRAIN_FRAC = 0.70
STRATEGIES = ["recipe", "operator", "apc_raman"]

print("Loading dataset...", flush=True)
df, _ = load_indpensim(DATA)
labels = batch_labels(df)

features = [
    f for f in available_process_vars(df)
    if df[f].nunique() > 3
]

normal_ids = labels.index[~labels["has_fault"]].to_numpy()
batches = {
    int(b): d.reset_index(drop=True)
    for b, d in df.groupby("batch_id")
    if b in normal_ids
}

rows = []

for rep in range(N_SPLITS):
    rng = np.random.RandomState(42 + rep)

    # Stratified batch-level splitting ensures that
    # every strategy appears in training and testing.
    train_ids = []
    test_ids = []

    for strategy in STRATEGIES:
        ids = np.array([
            b for b in normal_ids
            if labels.loc[b, "strategy"] == strategy
        ])

        rng.shuffle(ids)
        n_train = int(len(ids) * TRAIN_FRAC)

        train_ids.extend(ids[:n_train])
        test_ids.extend(ids[n_train:])

    train_ids = np.array(train_ids)
    test_ids = np.array(test_ids)

    global_model = Monitor(features, k=5).fit(
        df[df["batch_id"].isin(train_ids)]
    )

    specific_models = {}

    for strategy in STRATEGIES:
        ids = [
            b for b in train_ids
            if labels.loc[b, "strategy"] == strategy
        ]

        specific_models[strategy] = Monitor(
            features, k=5
        ).fit(df[df["batch_id"].isin(ids)])

    for batch in test_ids:
        strategy = labels.loc[batch, "strategy"]
        d = batches[int(batch)]

        for name, model in [
            ("global", global_model),
            ("strategy_specific", specific_models[strategy])
        ]:
            scores, _ = model.score(d)
            alarm = model.alarm(scores, "pca")

            rows.append({
                "split": rep,
                "batch_id": int(batch),
                "strategy": strategy,
                "model": name,
                "false_alarm": int(alarm.any())
            })

    print(
        f"Completed split {rep + 1}/{N_SPLITS}",
        flush=True
    )

results = pd.DataFrame(rows)

split_summary = (
    results.groupby(["split", "model"])
    ["false_alarm"]
    .mean()
    .unstack("model")
)

strategy_summary = (
    results.groupby(["split", "strategy", "model"])
    ["false_alarm"]
    .mean()
    .reset_index()
)

print("\n=== REPEATED VALIDATION ===")
print(
    split_summary.agg(["mean", "std", "min", "max"])
    .to_string()
)

print("\n=== MEAN FALSE-POSITIVE RATE BY STRATEGY ===")
print(
    strategy_summary.groupby(["strategy", "model"])
    ["false_alarm"]
    .agg(["mean", "std"])
    .to_string()
)

difference = (
    split_summary["strategy_specific"]
    - split_summary["global"]
)

print("\n=== PAIRED DIFFERENCE ===")
print(f"Mean difference: {difference.mean():.4f}")
print(f"Std difference:  {difference.std():.4f}")
print(
    "Splits where global PCA performed better:",
    int((difference > 0).sum()),
    "of",
    N_SPLITS
)
print(
    "Splits tied:",
    int((difference == 0).sum())
)

results.to_csv(
    OUT / "pca_repeated_batch_results.csv",
    index=False
)

split_summary.to_csv(
    OUT / "pca_repeated_split_summary.csv"
)

strategy_summary.to_csv(
    OUT / "pca_repeated_strategy_summary.csv",
    index=False
)

print("\nSaved all validation tables.")
