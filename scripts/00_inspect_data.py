"""Run FIRST on the real file. Prints how columns were resolved so you can fix src/data.py ALIASES if needed."""
import sys; sys.path.insert(0, ".")
import pandas as pd
from src.data import resolve_columns, is_spectral, ALIASES

path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/100_Batches_IndPenSim_V3.csv"
header = pd.read_csv(path, nrows=0).columns
nonspec = [c for c in header if not is_spectral(c)]
print(f"{len(header)} columns total, {len(header) - len(nonspec)} spectral (Raman), {len(nonspec)} non-spectral\n")
m = resolve_columns(header)
for k in ALIASES: print(f"{k:12s} -> {m.get(k, '!! NOT FOUND !!')}")
print("\nAll non-spectral columns:"); [print("  ", c) for c in nonspec]

# --- row/header alignment check: count fields in the first data row vs header ---
import csv
with open(path, newline="") as f:
    r = csv.reader(f); h = next(r); row = next(r)
n_spec = sum(is_spectral(c) for c in h)
print(f"\nheader fields: {len(h)} | first data row fields: {len(row)}", "-> OK" if len(h) == len(row) else "-> MISMATCH! columns will shift on load")
vals = dict(zip(h, row))
for key in ["Batch ID", "Fault flag", "Time (h)"]:
    print(f"  first-row {key!r}: {vals.get(key)}")
