"""Robust portfolio construction over a strategy universe.

Three formulations and a saturation curve:

    sample-min-var      : argmin w' Sigma_sample w        s.t. 1' w = 1, w >= 0
    lw-min-var          : argmin w' Sigma_lw w            (Ledoit-Wolf shrinkage)
    robust-min-var      : argmin max_{Sigma in U}  w' Sigma w
                          where U = { Sigma_lw + Delta : ||Delta||_F <= rho }
                          (Ben-Tal/Nemirovski-style box on the covariance)

For n strategies in {1, 5, 25, 100, 500, 2k, 10k} (capped at corpus size),
the saturation curve plots OOS volatility, max-DD, and CVaR-95 of each
selected portfolio. The elbow is the n at which adding more strategies
stops reducing tail risk.

Usage:
    python scripts/robust_portfolio.py                 # synthetic demo
    python scripts/robust_portfolio.py --from-data \\
        --returns-parquet /mnt/d/strategies_parquet/pnl_daily \\
        --asset BTC_30m_27W
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf

OUT_FIG = Path(__file__).resolve().parent.parent / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)
RESULTS_JSON = OUT_FIG.parent / "portfolio.json"


# -------------------------------------------------------------------
# Covariance estimators
# -------------------------------------------------------------------

def cov_sample(R: np.ndarray) -> np.ndarray:
    return np.cov(R, rowvar=False)


def cov_ledoit_wolf(R: np.ndarray) -> tuple[np.ndarray, float]:
    lw = LedoitWolf().fit(R)
    return lw.covariance_, float(lw.shrinkage_)


# -------------------------------------------------------------------
# Portfolio solvers
# -------------------------------------------------------------------

def solve_min_var(Sigma: np.ndarray, long_only: bool = True) -> np.ndarray:
    """Closed-form unconstrained min-variance is Sigma^{-1} 1 / (1' Sigma^{-1} 1)
    but for long-only we use cvxpy.
    """
    import cvxpy as cp
    n = Sigma.shape[0]
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1]
    if long_only:
        cons.append(w >= 0)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma))), cons)
    prob.solve(solver=cp.SCS)
    return np.array(w.value).flatten()


def solve_robust_min_var(Sigma: np.ndarray, rho: float = 0.05,
                         long_only: bool = True) -> np.ndarray:
    """Min over w of max over Delta of w'(Sigma+Delta)w with ||Delta||_F <= rho.
    Closed form: max-norm perturbation aligned with w gives
        max term = w'Sigma w + rho * ||w||_2^2
    so the robust problem becomes  min  w'Sigma w + rho * ||w||_2^2  s.t. simplex.
    """
    import cvxpy as cp
    n = Sigma.shape[0]
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1]
    if long_only:
        cons.append(w >= 0)
    obj = cp.quad_form(w, cp.psd_wrap(Sigma)) + rho * cp.sum_squares(w)
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.SCS)
    return np.array(w.value).flatten()


# -------------------------------------------------------------------
# Saturation curve
# -------------------------------------------------------------------

def saturation_curve(R_train: np.ndarray, R_test: np.ndarray,
                     ns: list[int], rho: float = 0.05,
                     n_repeats: int = 5, seed: int = 42) -> pd.DataFrame:
    """For each n: pick n random strategies, build three portfolios on R_train,
    compute their OOS metrics on R_test. Repeat n_repeats times.
    """
    rng = np.random.default_rng(seed)
    N = R_train.shape[1]
    rows = []
    for n in ns:
        n = min(n, N)
        for rep in range(n_repeats):
            idx = rng.choice(N, size=n, replace=False)
            Rtr = R_train[:, idx]
            Rte = R_test[:, idx]
            for label, Sigma_fn, weight_fn in [
                ("sample", cov_sample, lambda S: solve_min_var(S)),
                ("lw", lambda r: cov_ledoit_wolf(r)[0],
                 lambda S: solve_min_var(S)),
                ("robust", lambda r: cov_ledoit_wolf(r)[0],
                 lambda S: solve_robust_min_var(S, rho=rho)),
            ]:
                try:
                    Sigma = Sigma_fn(Rtr)
                    w = weight_fn(Sigma)
                    if w is None or np.any(np.isnan(w)):
                        continue
                except Exception:
                    continue
                pnl_oos = Rte @ w
                vol = float(pnl_oos.std() * np.sqrt(252))
                cum = np.cumsum(pnl_oos)
                mdd = float((cum - np.maximum.accumulate(cum)).min())
                cvar = float(np.sort(pnl_oos)[:max(1, int(0.05 * len(pnl_oos)))].mean())
                eff_n = int((np.abs(w) > 1e-4).sum())
                rows.append({
                    "n_universe": n, "rep": rep, "method": label,
                    "vol_oos": vol, "max_dd_oos": mdd, "cvar95_oos": cvar,
                    "effective_n": eff_n,
                    "weight_concentration": float(np.sum(w**2)),
                })
    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# Synthetic demo
# -------------------------------------------------------------------

def load_real_returns(parquet_root: str, asset: str | None = None,
                      min_active_days: int = 60
                      ) -> tuple[np.ndarray, list[str]]:
    """Load the pnl_daily/ Parquet and pivot to a (T, N) returns matrix.

    Drops strategies with fewer than `min_active_days` non-zero-PnL days
    (otherwise sample covariance is dominated by sparsity). Fills missing
    (strategy, date) cells with zero — a strategy that didn't trade that day
    is genuinely contributing zero PnL to a portfolio.

    Returns (R, strat_index) where R[t, i] is strategy_index[i]'s PnL on day t.
    """
    import pyarrow.dataset as ds
    d = ds.dataset(parquet_root, format="parquet", partitioning="hive")
    flt = None
    if asset:
        flt = ds.field("asset") == asset
    df = d.to_table(filter=flt,
                    columns=["asset", "strategy_name", "date",
                             "pnl_sum", "n_trades"]).to_pandas()
    if df.empty:
        raise RuntimeError(f"no rows in {parquet_root} (asset={asset!r})")
    # Filter active strategies
    counts = df.groupby(["asset", "strategy_name"])["date"].nunique()
    active = counts[counts >= min_active_days].index
    df = df.set_index(["asset", "strategy_name"]).loc[active].reset_index()
    df["sid"] = df["asset"] + "/" + df["strategy_name"]
    pivot = df.pivot_table(index="date", columns="sid", values="pnl_sum",
                           aggfunc="sum", fill_value=0.0)
    pivot = pivot.sort_index()
    return pivot.to_numpy(dtype=np.float64), list(pivot.columns)


def synthetic_returns(N: int = 200, T: int = 1500, n_factors: int = 4,
                      seed: int = 42) -> np.ndarray:
    """N strategies, T daily returns. Each strategy = small loading on
    a few latent factors plus per-strategy idio noise.
    """
    rng = np.random.default_rng(seed)
    F = rng.normal(0, 0.01, size=(T, n_factors))
    L = rng.normal(0, 0.5, size=(N, n_factors)) * \
        rng.choice([0, 1], size=(N, n_factors), p=[0.5, 0.5])
    eps = rng.normal(0, 0.005, size=(T, N))
    R = F @ L.T + eps
    return R


# -------------------------------------------------------------------
# Plots
# -------------------------------------------------------------------

def plot_saturation(curve: pd.DataFrame, out_path: Path,
                    metric: str = "cvar95_oos"):
    agg = (curve.groupby(["n_universe", "method"])[metric]
           .agg(["mean", "std"]).reset_index())
    fig, ax = plt.subplots(figsize=(9, 5.5))
    palette = {"sample": "#999999", "lw": "#4c8acc", "robust": "#cc4c4c"}
    labels = {"sample": "sample cov + min-var",
              "lw": "Ledoit-Wolf + min-var",
              "robust": "Ledoit-Wolf + robust min-var (rho)"}
    for m in ("sample", "lw", "robust"):
        sub = agg[agg["method"] == m]
        if sub.empty:
            continue
        ax.errorbar(sub["n_universe"], sub["mean"], yerr=sub["std"],
                    marker="o", linestyle="-", color=palette[m],
                    label=labels[m], capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("Number of strategies in portfolio")
    if metric == "cvar95_oos":
        ax.set_ylabel("Out-of-sample CVaR-95 (more negative = worse)")
    elif metric == "vol_oos":
        ax.set_ylabel("Out-of-sample annualised vol")
    else:
        ax.set_ylabel(metric)
    ax.set_title(f"Saturation curve: {metric}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_method_comparison(curve: pd.DataFrame, out_path: Path):
    """Pairwise relative comparison: lw vs sample, robust vs lw."""
    agg = curve.groupby(["n_universe", "method"])[
        ["vol_oos", "max_dd_oos", "cvar95_oos"]].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    for ax, metric, title in zip(axes,
                                 ["vol_oos", "cvar95_oos", "max_dd_oos"],
                                 ["OOS volatility (annualised)",
                                  "OOS CVaR-95",
                                  "OOS max drawdown"]):
        for m in ("sample", "lw", "robust"):
            sub = agg[agg["method"] == m]
            if sub.empty:
                continue
            ax.plot(sub["n_universe"], sub[metric], "o-", label=m)
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("n strategies")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def split_train_test(R: np.ndarray, train_frac: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    n = int(R.shape[0] * train_frac)
    return R[:n], R[n:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--returns-parquet",
                    default="/mnt/d/strategies_parquet/pnl_daily",
                    help="path to the pnl_daily/ Parquet substrate")
    ap.add_argument("--asset", default=None)
    ap.add_argument("--rho", type=float, default=0.0001,
                    help="Frobenius-norm uncertainty radius for robust min-var")
    ap.add_argument("--ns", nargs="+", type=int,
                    default=[5, 10, 25, 50, 100, 200])
    ap.add_argument("--synthetic", action="store_true",
                    help="(rare) factor-model synthetic returns; only for "
                         "testing the analysis machinery.")
    args = ap.parse_args()

    summary: dict = {"mode": "synthetic" if args.synthetic else "data",
                     "rho": args.rho}
    if not args.synthetic:
        R, strat_index = load_real_returns(args.returns_parquet,
                                           asset=args.asset,
                                           min_active_days=60)
        summary["asset"] = args.asset
        print(f"loaded R: {R.shape} (T x N), {len(strat_index):,} strategies "
              f"with >= 60 active days")
    else:
        R = synthetic_returns()
    R_train, R_test = split_train_test(R)
    summary["n_strategies_universe"] = int(R.shape[1])
    summary["n_train_bars"] = int(R_train.shape[0])
    summary["n_test_bars"] = int(R_test.shape[0])
    print(f"R: {R.shape}, train {R_train.shape}, test {R_test.shape}")

    t0 = time.time()
    curve = saturation_curve(R_train, R_test, ns=args.ns, rho=args.rho)
    print(f"saturation_curve done in {time.time()-t0:.1f}s, "
          f"{len(curve)} rows")

    plot_saturation(curve, OUT_FIG / "fig_saturation_cvar.png", metric="cvar95_oos")
    plot_saturation(curve, OUT_FIG / "fig_saturation_vol.png", metric="vol_oos")
    plot_method_comparison(curve, OUT_FIG / "fig_method_comparison.png")

    # Mean curves
    mean_curve = (curve.groupby(["n_universe", "method"])[
        ["vol_oos", "max_dd_oos", "cvar95_oos"]].mean().reset_index())
    summary["mean_curve"] = mean_curve.to_dict(orient="records")
    with open(RESULTS_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nresults summary -> {RESULTS_JSON}")


if __name__ == "__main__":
    main()
