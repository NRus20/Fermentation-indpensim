"""Shrink the 2,200-column IndPenSim CSV for sharing/upload: drops all numeric-named (Raman) columns,
keeps every Nth sample per batch (default 5 -> 1 h resolution), rounds values.
usage: python scripts/make_slim.py 100_Batches_IndPenSim_V3.csv IndPenSim_slim.csv [step]"""
import sys, pandas as pd
src, dst = sys.argv[1], sys.argv[2]; step = int(sys.argv[3]) if len(sys.argv) > 3 else 5
def is_num(c):
    try: float(c); return True
    except ValueError: return False
cols = [c for c in pd.read_csv(src, nrows=0).columns if not is_num(c)]
df = pd.read_csv(src, usecols=cols)
df = df[df.groupby("Batch ID").cumcount() % step == 0].round(5)
df.to_csv(dst, index=False); print(df.shape, "->", dst)
