"""Roofline-model analysis (Extension / Original Initiative).

Builds a log-log Roofline plot for this CPU-only box and overlays the measured
decode operating points (Baseline direct-FP16 vs AirLLM-FP16). The plot makes
the core finding visual: token-by-token Decode is a GEMV with arithmetic
intensity ~1 FLOP/byte, so it lands far left of every ridge point and is pinned
to memory bandwidth — DDR5 for the baseline, the much lower NVMe roof for AirLLM.

Usage:
    uv run python experiments/generate_roofline.py

Output: figures/roofline.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_FIGURES_DIR = _ROOT / "figures"

# --- Hardware roofs (documented, from PRD §3) -----------------------------
# CPU FP32 peak: 8 cores × ~5.0 GHz × 64 FLOP/cycle (2× AVX-512 FMA units,
# Zen 5) ≈ 2560 GFLOP/s. Theoretical; real GEMM efficiency is ~50-70%.
COMPUTE_ROOF_GFLOPS = 2560.0
DDR5_BW_GBS = 50.0   # DDR5-5200 dual-channel, effective (~83 GB/s theoretical)
NVME_BW_GBS = 7.0    # PCIe 4.0 x4 sequential read

# --- Workload constants ----------------------------------------------------
PARAMS = 8.03e9                 # Meta-Llama-3.1-8B
FLOP_PER_TOKEN = 2 * PARAMS     # 1 multiply + 1 add per weight, used once (GEMV)
DECODE_AI = 1.0                 # GEMV: 2 FLOP per 2-byte FP16 weight = 1 FLOP/byte
PREFILL_AI = 11.0              # GEMM, ~11-token prompt reuses each weight ~11×

# --- Measured decode points (from results/*.json) -------------------------
BASELINE_TPOT_S = 0.62         # estimated: 10 tok ÷ 6.24 s baseline runtime
AIRLLM_TPOT_S = 15.09          # results/airllm_fp16_*.json


def _perf_at(ai: float, bw_gbs: float) -> float:
    """Attainable GFLOP/s at arithmetic intensity `ai` under memory roof `bw`."""
    return min(COMPUTE_ROOF_GFLOPS, ai * bw_gbs)


def _plot_roof(ax, ai: np.ndarray, bw_gbs: float, label: str, color: str) -> None:
    """Draw one memory roof (diagonal until it meets the compute ceiling)."""
    perf = np.minimum(COMPUTE_ROOF_GFLOPS, ai * bw_gbs)
    ax.plot(ai, perf, color=color, linewidth=2, label=label)
    ridge = COMPUTE_ROOF_GFLOPS / bw_gbs
    ax.plot(ridge, COMPUTE_ROOF_GFLOPS, "o", color=color, markersize=5)
    ax.annotate(f"ridge ≈ {ridge:.0f}", (ridge, COMPUTE_ROOF_GFLOPS),
                textcoords="offset points", xytext=(4, -12),
                fontsize=7, color=color)


def main() -> None:
    """Generate figures/roofline.png."""
    _FIGURES_DIR.mkdir(exist_ok=True)
    ai = np.geomspace(0.1, 1000, 400)

    fig, ax = plt.subplots(figsize=(9, 6))

    # Compute ceiling.
    ax.axhline(COMPUTE_ROOF_GFLOPS, color="black", linestyle="--", linewidth=1.5,
               label=f"CPU compute roof ≈ {COMPUTE_ROOF_GFLOPS/1000:.1f} TFLOP/s")

    # Memory roofs.
    _plot_roof(ax, ai, DDR5_BW_GBS, f"DDR5 roof ({DDR5_BW_GBS:.0f} GB/s)", "#1f77b4")
    _plot_roof(ax, ai, NVME_BW_GBS, f"NVMe roof ({NVME_BW_GBS:.0f} GB/s)", "#9467bd")

    # Measured decode operating points (perf = FLOP/token ÷ TPOT).
    base_perf = FLOP_PER_TOKEN / BASELINE_TPOT_S / 1e9
    air_perf = FLOP_PER_TOKEN / AIRLLM_TPOT_S / 1e9
    ax.plot(DECODE_AI, base_perf, "s", color="#d62728", markersize=11,
            label=f"Baseline decode ({base_perf:.1f} GFLOP/s)")
    ax.plot(DECODE_AI, air_perf, "D", color="#2ca02c", markersize=11,
            label=f"AirLLM decode ({air_perf:.2f} GFLOP/s)")

    # Decode / Prefill intensity guides.
    ax.axvline(DECODE_AI, color="grey", linestyle=":", linewidth=1)
    ax.annotate("Decode (GEMV)\nAI ≈ 1 FLOP/byte\n→ memory-bound",
                (DECODE_AI, 0.2), fontsize=8, color="grey",
                ha="left", va="bottom")
    ax.axvline(PREFILL_AI, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.annotate("Prefill (GEMM)\nAI ≈ 11", (PREFILL_AI, 0.2),
                fontsize=8, color="grey", ha="left", va="bottom", alpha=0.8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP / byte)")
    ax.set_ylabel("Attainable Performance (GFLOP/s)")
    ax.set_title("Roofline Model — CPU-only Llama-3.1-8B Inference\n"
                 "Decode sits left of every ridge point: pinned to memory bandwidth")
    ax.set_ylim(0.1, COMPUTE_ROOF_GFLOPS * 3)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    out = _FIGURES_DIR / "roofline.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")
    print(f"  Baseline decode : {base_perf:.1f} GFLOP/s "
          f"({base_perf / DDR5_BW_GBS * 100:.0f}% of DDR5 roof at AI=1)")
    print(f"  AirLLM decode   : {air_perf:.2f} GFLOP/s "
          f"({air_perf / NVME_BW_GBS * 100:.0f}% of NVMe roof at AI=1)")


if __name__ == "__main__":
    main()
