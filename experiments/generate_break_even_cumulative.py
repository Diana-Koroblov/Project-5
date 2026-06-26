"""Cumulative-cost break-even figure (total monthly spend vs usage volume).

Complements `run_economics.py` (which plots *cost per request*). The
assignment asks specifically for a "graph of cumulative cost vs usage
volume", which is the classic break-even view: a flat On-Prem line crossed
by rising API / Cloud lines. Two panels:

  A (linear)  — On-Prem vs API, with shaded decision regions and the
                caching-dependent band between the two break-even points.
  B (log-log) — all four options across the full range, showing both the
                Cloud×On-Prem and API×On-Prem crossings.

Usage:
    uv run python experiments/generate_break_even_cumulative.py

Output: figures/break_even_cumulative.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from ex05.config import EconomicsConfig
from ex05.economics import (
    compute_api_cost,
    compute_cloud_gpu_cost,
    compute_onprem_cost,
    find_breakeven,
)

_ROOT = Path(__file__).resolve().parents[1]

_GREEN = "#2ca02c"
_RED = "#d62728"
_ORANGE = "#ff7f0e"
_PURPLE = "#9467bd"


def _thousands(value: float, _pos: int) -> str:
    """Axis tick formatter: 525000 -> '525k', 1000000 -> '1.0M'."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def _panel_linear(ax, x, fixed, api_nc, api_c, n_nc, n_c) -> None:
    """Linear cumulative view: On-Prem (flat) vs API, with decision regions."""
    ax.axvspan(0, n_nc, color=_RED, alpha=0.06)
    ax.axvspan(n_nc, n_c, color=_ORANGE, alpha=0.18)
    ax.axvspan(n_c, x[-1], color=_GREEN, alpha=0.06)

    ax.axhline(fixed, color=_GREEN, linewidth=2, label="On-Prem (fixed/mo)")
    ax.plot(x, api_nc * x, color=_RED, linewidth=2, linestyle="--",
            label="API (no cache)")
    ax.plot(x, api_c * x, color=_ORANGE, linewidth=2, linestyle="-.",
            label="API (cached)")

    for n, color in [(n_nc, _RED), (n_c, _ORANGE)]:
        ax.plot(n, fixed, "o", color=color, markersize=7, zorder=5)
    ax.annotate(
        f"Break-even N*\n{int(n_nc):,} (no cache)\n{int(n_c):,} (cached)",
        xy=(n_nc, fixed), xytext=(n_nc * 0.34, fixed * 1.95),
        color="#333333", fontsize=8,
        arrowprops={"arrowstyle": "->", "color": "#333333", "alpha": 0.7})

    ax.text(n_nc * 0.48, fixed * 0.45, "API cheaper", color=_RED,
            fontsize=9, ha="center")
    ax.text((n_c + x[-1]) / 2, fixed * 0.45, "Self-host cheaper", color=_GREEN,
            fontsize=9, ha="center")
    ax.text(n_c * 1.02, fixed * 2.45, "← caching\n   band", color="#b8860b",
            fontsize=7.5, ha="left", va="top")

    ax.set_xlim(0, x[-1])
    ax.set_ylim(0, fixed * 2.6)
    ax.set_xlabel("Monthly requests")
    ax.set_ylabel("Total monthly cost (ILS)")
    ax.set_title("A — Cumulative cost: On-Prem vs API")
    ax.xaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)


def _panel_log(ax, fixed, api_nc, api_c, cloud, n_nc, n_cloud) -> None:
    """Log-log cumulative view: all four options across the full range."""
    xl = np.geomspace(1, 1_000_000, 400)
    ax.axhline(fixed, color=_GREEN, linewidth=2, label="On-Prem (fixed/mo)")
    ax.plot(xl, api_nc * xl, color=_RED, linewidth=2, linestyle="--",
            label="API (no cache)")
    ax.plot(xl, api_c * xl, color=_ORANGE, linewidth=2, linestyle="-.",
            label="API (cached)")
    ax.plot(xl, cloud * xl, color=_PURPLE, linewidth=2, linestyle=":",
            label="Cloud GPU")

    for n, color, tag in [
        (n_cloud, _PURPLE, f"Cloud N*={int(n_cloud):,}"),
        (n_nc, _RED, f"API N*={int(n_nc):,}"),
    ]:
        ax.axvline(n, color=color, alpha=0.5, linewidth=1)
        ax.text(n * 1.1, fixed * 0.3, tag, color=color, fontsize=7.5,
                rotation=90, va="bottom")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Monthly requests")
    ax.set_ylabel("Total monthly cost (ILS)")
    ax.set_title("B — Full range incl. Cloud GPU (log-log)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3, which="both")


def main() -> None:
    """Build and save the cumulative-cost break-even figure."""
    cfg = EconomicsConfig.load()

    fixed = compute_onprem_cost(1, cfg)  # per-req at N=1 == total fixed/mo
    api_nc = compute_api_cost(cfg, use_cache=False)
    api_c = compute_api_cost(cfg, use_cache=True)
    cloud = compute_cloud_gpu_cost(cfg)

    n_nc = find_breakeven(cfg, use_cache=False)
    n_c = find_breakeven(cfg, use_cache=True)
    n_cloud = fixed / cloud

    print("=" * 60)
    print("CUMULATIVE BREAK-EVEN")
    print("=" * 60)
    print(f"  On-Prem fixed/mo        : {fixed:.2f} ILS")
    print(f"  Break-even API (no cache): {int(n_nc):,} req/mo")
    print(f"  Break-even API (cached)  : {int(n_c):,} req/mo")
    print(f"  Break-even Cloud GPU     : {int(n_cloud):,} req/mo")

    x = np.linspace(0, 700_000, 500)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6))
    _panel_linear(ax_a, x, fixed, api_nc, api_c, n_nc, n_c)
    _panel_log(ax_b, fixed, api_nc, api_c, cloud, n_nc, n_cloud)
    fig.tight_layout()

    out = _ROOT / "figures" / "break_even_cumulative.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


if __name__ == "__main__":
    main()
