# Pre-specified hypotheses (write results next to each one after running)
H1. Final yield differs by control strategy (APC/Raman-guided > recipe-driven; operator-controlled most variable). *Test:* Kruskal-Wallis, Holm pairwise, rank-biserial.
H2. Within batches, substrate, DO and pH excursions relate to production rate (dP/dt) more consistently than to P level (which mostly tracks time). *Test:* stats_within_batch.csv.
H3. Final yield is partly predictable from the first 48-96 h, and prediction improves with horizon. *Test:* model_batch_early_prediction.csv vs mean baseline.
H4. Sample-level models beat a time-only baseline on held-out batches. *Test:* model_sample_level.csv. If not, the "prediction" is just the time course.
H5. PCA T2/SPE trained on fault-free batches detects faults with low batch-level false-alarm rate and short delay; Isolation Forest is weaker/needs tuning. *Test:* anomaly_summary.csv.
H6. Models trained on some strategies transfer imperfectly to unseen strategies. *Test:* model_leave_strategy_out.csv (look at bias).
