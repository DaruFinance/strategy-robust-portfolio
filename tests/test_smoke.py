"""Smoke test for the analysis machinery itself.

Uses `--synthetic` (factor-model toy returns) to exercise the cov →
solver → saturation-curve pipeline deterministically. Real-corpus runs go
through the orchestrator (and depend on the pnl_daily ETL).
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_synthetic_demo():
    res = subprocess.run(
        [sys.executable, "scripts/robust_portfolio.py", "--synthetic"],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    assert res.returncode == 0, res.stderr
    out = json.loads((ROOT / "portfolio.json").read_text())
    assert out["mode"] == "synthetic"
    assert out["n_strategies_universe"] == 200
    # Sanity: vol should monotonically decrease in n for the LW method
    rows = [r for r in out["mean_curve"] if r["method"] == "lw"]
    rows.sort(key=lambda r: r["n_universe"])
    vols = [r["vol_oos"] for r in rows]
    assert vols == sorted(vols, reverse=True), \
        f"LW vol_oos not monotonic in n: {vols}"
    for fname in ["fig_saturation_cvar.png", "fig_saturation_vol.png",
                  "fig_method_comparison.png"]:
        assert (ROOT / "figures" / fname).is_file(), fname


if __name__ == "__main__":
    test_synthetic_demo()
    print("OK")
