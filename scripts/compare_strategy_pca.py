import numpy as np
import pandas as pd

from src.data import load_indpensim, batch_labels, available_process_vars
from src.anomaly import Monitor

PATH = "data/raw/100_Batches_IndPenSim_V3_reconstructed.csv"
SEED = 42

df, _ = load_indpensim(PATH)
labels = batch_labels(df)
features = [
    f for f in available_process_vars(df)
    if df[f].nunique() > 3
]

normal = labels.index[~labels["has_fault"]].to_numpy()
rng = np.random.RandomState(SEED)

# Same 70/30 split as our previous diagnostic.
train_ids = rng.choice(
    normal, int(len(normal) * 0.7), replace=False
)
test_ids = np.setdiff1d(normal, train_ids)

global_model = Monitor(features, k=5).fit(
    df[df["batch_id"].isin(train_ids)]
)

strategies = ["recipe", "operator", "apc_raman"]
strategy_models = {}

for strategy in strategies:
    ids = [
        b for b in train_ids
        if labels.loc[b, "strategy"] == strategy
    ]
    strategy_models[strategy] = Monitor(features, k=5).fit(
        df[df["batch_id"].isin(ids)]
    )
    print(f"Trained {strategy} model on {len(ids)} batches")

rows = []

for batch in test_ids:
    strategy = labels.loc[batch, "strategy"]
    d = df[df["batch_id"] == batch].reset_index(drop=True)

    for model_name, model in [
        ("global", global_model),
        ("strategy_specific", strategy_models[strategy])
    ]:
        scores, _ = model.score(d)
        alarm = model.alarm(scores, "pca")

        rows.append({
            "batch_id": int(batch),
            "strategy": strategy,
            "model": model_name,
            "false_alarm": int(alarm.any())
        })

results = pd.DataFrame(rows)

summary = (
    results.groupby(["model", "strategy"])
    .agg(
        batches=("batch_id", "count"),
        false_alarms=("false_alarm", "sum"),
        false_positive_rate=("false_alarm", "mean")
    )
    .reset_index()
)

overall = (
    results.groupby("model")
    .agg(
        batches=("batch_id", "count"),
        false_alarms=("false_alarm", "sum"),
        false_positive_rate=("false_alarm", "mean")
    )
)

print("\n=== FALSE ALARMS BY STRATEGY ===")
print(summary.to_string(index=False))

print("\n=== OVERALL COMPARISON ===")
print(overall.to_string())

print("\n=== BATCH-LEVEL COMPARISON ===")
print(
    results.pivot(
        index=["batch_id", "strategy"],
        columns="model",
        values="false_alarm"
    ).to_string()
)

from pathlib import Path
out = Path("results/tables")
out.mkdir(parents=True, exist_ok=True)

summary.to_csv(out / "strategy_pca_comparison.csv", index=False)
results.to_csv(out / "strategy_pca_batch_results.csv", index=False)

print("\nSaved comparison tables.")
