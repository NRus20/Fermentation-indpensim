# Publication checklist — IndPenSim portfolio

This checklist distinguishes documentation improvements from checks that require the user's local repository/data. Do not mark a check as passed without evidence.

## Scientific checks (must resolve before publishing numerical claims)

- [ ] **Yield units:** inspect `100_Batches_IndPenSim_Statistics.csv` headers, raw values, and `src/data.py::load_yields`. The previous README described group means around 2,913–3,484 **tonnes**, which is likely a conversion/label error. Determine whether raw numbers are kg, g, or another unit, then regenerate Stage 2/3 results and report with consistent units.
- [ ] **Fault onset:** compare `fault_flag` and `fault_ref` time courses for batches 91–100. The original `src/anomaly.py::_onset` uses `fault_flag > 0`; `scripts/validate_pca_final.py` uses `fault_ref > 0`. Confirm definitions against dataset documentation and re-run metrics if needed.
- [ ] **Potential control-strategy confounding:** check that the 10 fault batches' operating regimes are comparable with the 90 fault-free training batches; a fault-associated batch being flagged could partly reflect a strategy difference.
- [ ] **Predictive-model metrics:** read `results/tables/model_batch_early_prediction.csv`, `model_sample_level.csv`, and `model_leave_strategy_out.csv`. Include only actual R², RMSE, and uncertainty, not generic claims of accuracy.
- [ ] **Statistical conclusions:** inspect `results/tables/stats_*.csv` and check multiple-testing correction before claiming significant effects. Resolve yield units first.
- [ ] **Alarm definition:** verify whether an alarm after fault onset could reflect an earlier exceedance streak; explicitly document the rule and its limitations.

## Reproducibility

- [ ] **Repair script:** add the actual deterministic raw CSV alignment / batch-ID reconstruction code as a script (for example `scripts/reconstruct_batches.py`), or provide a verified step-by-step recipe. Test from a clean original download.
- [ ] **Check input names:** ensure scripts reference `data/raw/100_Batches_IndPenSim_V3_reconstructed.csv` where appropriate.
- [ ] **Run commands:** test every command in README from the repository root, using a clean virtual environment.
- [ ] **Tests:** run `python -m pytest tests`; check that tests exercise the repaired dataset format as well as synthetic data.
- [ ] **Versioning:** record package versions and random seeds. The final PCA validation uses seed 20261019, whereas older modules use 42.
- [ ] **Source/README consistency:** the backup README embedded source code from an older version; maintain implementation only in source files and link to them from README.
- [ ] **Results:** verify three PNGs and four PCA validation CSVs exist, render correctly, and correspond to the final script.
- [ ] **Data redistribution:** verify original Mendeley Data license, dataset version, citation, and whether original or reconstructed data can be uploaded.
- [ ] **Git hygiene:** add `.gitignore` for `.venv/`, `__pycache__/`, `.pytest_cache/`, large `data/raw/` files, and other temporary files. Keep appropriate reproducible figures/tables.
- [ ] **License:** choose and add an actual `LICENSE` file for original project code; do not assume this licenses IndPenSim data.

## Optional presentation improvements

- [ ] Add 1–2 verified predictive-model findings and a corresponding figure.
- [ ] Add a short table summarizing the six original hypotheses and their evidence status.
- [ ] Add a brief note about what threshold tuning means operationally for false alarms versus missed faults.
- [ ] Include the data source and licensing details in a separate `DATA_AVAILABILITY.md` if raw data cannot be committed.
