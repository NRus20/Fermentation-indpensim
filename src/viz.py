import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

FIG = Path("results/figures"); TAB = Path("results/tables")
STRAT_COLORS = {"recipe": "#4C72B0", "operator": "#DD8452", "apc_raman": "#55A868", "fault": "#C44E52",
                "unknown": "#999999"}

def setup():
    sns.set_theme(style="whitegrid", context="notebook")
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)

def save(fig, name):
    setup(); fig.tight_layout(); fig.savefig(FIG / f"{name}.png", dpi=200); plt.close(fig)
