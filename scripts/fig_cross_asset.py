"""Cross-asset saturation curve overlay: 10 assets on one CVaR-95 chart."""
import json, glob
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = []
for f in sorted(glob.glob(str(ROOT / "results_per_asset" / "portfolio_*.json"))):
    asset = f.split("portfolio_")[1].replace(".json", "")
    d = json.load(open(f))
    df = pd.DataFrame(d["mean_curve"])
    for r in df.to_dict(orient="records"):
        r["asset"] = asset
        rows.append(r)
data = pd.DataFrame(rows)
lw = data[data["method"] == "lw"].copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
palette = plt.colormaps["tab10"]
assets = sorted(lw["asset"].unique())

for i, a in enumerate(assets):
    sub = lw[lw["asset"] == a].sort_values("n_universe")
    axes[0].plot(sub["n_universe"], sub["cvar95_oos"], "o-",
                 color=palette(i % 10), linewidth=1.5, alpha=0.9, label=a)
    axes[1].plot(sub["n_universe"], sub["vol_oos"], "o-",
                 color=palette(i % 10), linewidth=1.5, alpha=0.9, label=a)

axes[0].set_xscale("log")
axes[0].set_xlabel("Number of strategies in portfolio")
axes[0].set_ylabel("OOS CVaR-95 (more negative = worse)")
axes[0].set_title("Cross-asset saturation: tail risk")
axes[0].grid(True, alpha=0.3, which="both")
axes[0].legend(loc="lower right", fontsize=8, ncol=2, framealpha=0.85)

axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("Number of strategies in portfolio")
axes[1].set_ylabel("OOS annualised vol (log)")
axes[1].set_title("Cross-asset saturation: volatility")
axes[1].grid(True, alpha=0.3, which="both")

fig.suptitle("Diversification on 10 deep-WFO crypto assets — Ledoit-Wolf min-var",
             fontsize=12)
fig.tight_layout()
out = ROOT / "figures" / "fig_cross_asset_saturation.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
