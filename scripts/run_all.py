"""python scripts/run_all.py [path/to/IndPenSim.csv]  -> runs stages 1-4, writes results/ figures + tables + summary.json"""
import sys, json, time; sys.path.insert(0, ".")
import os
from src.data import load_indpensim, load_yields
from src import eda, stats, models, anomaly, viz

path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/100_Batches_IndPenSim_V3.csv"
ypath = sys.argv[2] if len(sys.argv) > 2 else "data/raw/100_Batches_IndPenSim_Statistics.csv"
yields = load_yields(ypath) if os.path.exists(ypath) else None
target = "total_yield_t" if yields is not None else "final_penicillin"
print("Prediction target:", target)
viz.setup(); t0 = time.time()
df, mapping = load_indpensim(path)
print(f"Loaded {df.batch_id.nunique()} batches, {len(df):,} rows"); json.dump(mapping, open("results/column_mapping.json", "w"), indent=1)
summary = {}
print("Stage 1: EDA");        ov = eda.run(df); summary["strategy_counts"] = ov.strategy.value_counts().to_dict()
print("Stage 2: statistics"); summary["stats"] = stats.run(df, yields, target)
print("Stage 3: modelling")
res, _ = models.early_prediction(df, yields=yields, target=target); models.leave_strategy_out(df, yields=yields, target=target); sl, imp = models.sample_level(df)
summary["early_prediction_full"] = res[res.horizon_h == "full"].set_index("model")["r2"].to_dict()
summary["sample_level_r2"] = sl.set_index("model")["r2"].to_dict()
print("Stage 4: anomaly detection"); r, s = anomaly.evaluate(df)
if r is not None: summary["anomaly"] = r.groupby("method")[["batch_FPR", "batch_TPR", "median_delay_h", "sample_AUC"]].mean().to_dict()
json.dump(summary, open("results/summary.json", "w"), indent=1, default=float)
print(f"Done in {time.time() - t0:.0f}s. See results/")
