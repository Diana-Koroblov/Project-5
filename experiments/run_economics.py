"""Entry point for the break-even and OPEX analysis.

Computes On-Prem, API (no-cache), API (cached), and Cloud GPU cost-per-request
curves over the configured monthly request range, identifies the break-even
point, and saves a figure + JSON summary.

Usage:
    uv run python experiments/run_economics.py

Output: figures/break_even.png, results/economics_<timestamp>.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ex05.config import EconomicsConfig
from ex05.economics import (
    compute_api_cost,
    compute_cloud_gpu_cost,
    compute_onprem_cost,
    find_breakeven,
)

_ROOT = Path(__file__).resolve().parents[1]


def _build_request_range(cfg: EconomicsConfig) -> np.ndarray:
    """Return integer array of monthly request counts for the cost curves."""
    linear_max = min(100, cfg.monthly_request_range_max) + 1
    return np.unique(
        np.concatenate([
            np.arange(cfg.monthly_request_range_min, linear_max),
            np.geomspace(
                max(1, cfg.monthly_request_range_min),
                cfg.monthly_request_range_max,
                num=300,
            ).astype(int),
        ])
    )


def _generate_figure(
    x: np.ndarray,
    onprem: np.ndarray,
    api_normal: np.ndarray,
    api_cached: np.ndarray,
    cloud_gpu: np.ndarray,
    n_star: float,
    n_star_cached: float,
    out_path: Path,
) -> None:
    """Save the four-curve break-even figure."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, onprem, label="On-Prem (amortized)", color="#2ca02c", linewidth=2)
    ax.plot(x, api_normal, label="API (no cache)", color="#d62728",
            linewidth=2, linestyle="--")
    ax.plot(x, api_cached, label="API (cached)", color="#ff7f0e",
            linewidth=2, linestyle="-.")
    ax.plot(x, cloud_gpu, label="Cloud GPU", color="#9467bd",
            linewidth=2, linestyle=":")

    # Annotate break-even points
    for n, color, tag in [
        (n_star, "#d62728", f"N*={int(n_star):,}"),
        (n_star_cached, "#ff7f0e", f"N*(cached)={int(n_star_cached):,}"),
    ]:
        ax.axvline(n, color=color, alpha=0.5, linewidth=1)
        ax.text(n * 1.02, ax.get_ylim()[1] * 0.9, tag,
                color=color, fontsize=8, rotation=90, va="top")

    ax.set_xlabel("Monthly Requests")
    ax.set_ylabel("Cost per Request (ILS)")
    ax.set_title("Break-Even Analysis: On-Prem vs Cloud vs API")
    ax.set_xscale("log")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def main() -> None:
    """Run the economics analysis and persist results."""
    cfg = EconomicsConfig.load()

    x = _build_request_range(cfg)
    onprem = np.array([compute_onprem_cost(int(n), cfg) for n in x])
    api_normal_val = compute_api_cost(cfg, use_cache=False)
    api_cached_val = compute_api_cost(cfg, use_cache=True)
    cloud_val = compute_cloud_gpu_cost(cfg)

    api_normal = np.full_like(x, api_normal_val, dtype=float)
    api_cached = np.full_like(x, api_cached_val, dtype=float)
    cloud_gpu = np.full_like(x, cloud_val, dtype=float)

    n_star = find_breakeven(cfg, use_cache=False)
    n_star_cached = find_breakeven(cfg, use_cache=True)

    print("=" * 60)
    print("ECONOMICS ANALYSIS")
    print("=" * 60)
    print(f"  API cost/req (no cache) : {api_normal_val:.6f} ILS")
    print(f"  API cost/req (cached)   : {api_cached_val:.6f} ILS")
    print(f"  Cloud GPU cost/req      : {cloud_val:.6f} ILS")
    print(f"  Break-even (no cache)   : {int(n_star):,} req/month")
    print(f"  Break-even (cached)     : {int(n_star_cached):,} req/month")

    # Save figure
    figures_dir = _ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)
    _generate_figure(
        x, onprem, api_normal, api_cached, cloud_gpu,
        n_star, n_star_cached,
        figures_dir / "break_even.png",
    )

    # Save JSON summary
    results_dir = _ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "api_cost_per_request_ils": api_normal_val,
        "api_cost_per_request_cached_ils": api_cached_val,
        "cloud_gpu_cost_per_request_ils": cloud_val,
        "breakeven_requests_no_cache": int(n_star),
        "breakeven_requests_cached": int(n_star_cached),
        "config": {
            "hardware_cost_ils": cfg.hardware_cost_ils,
            "amortization_years": cfg.amortization_years,
            "maintenance_cost_annual_ils": cfg.maintenance_cost_annual_ils,
            "electricity_kwh_ils": cfg.electricity_kwh_ils,
            "avg_power_watts": cfg.avg_power_watts,
        },
    }
    out_path = results_dir / f"economics_{ts}.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out_path.name}")


if __name__ == "__main__":
    main()
