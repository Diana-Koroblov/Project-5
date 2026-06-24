"""Plot the Ollama CPU-quantization gradient (real quantization results).

Reads results/ollama_<level>_*.json and produces a two-panel figure showing how
peak RAM falls and throughput rises as precision drops FP16 -> Q8 -> Q4 -> Q2.
This is the figure for the "real quantization" stage (unlike AirLLM, GGUF/llama.cpp
quantization runs on CPU, so these are genuine quantized data points).

Usage:
    uv run python experiments/generate_ollama_graphs.py

Output: figures/ollama_quant_comparison.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIR = _ROOT / "results"
_FIGURES_DIR = _ROOT / "figures"

_ORDER = ["fp16", "q8", "q4", "q2"]
_LABELS = {"fp16": "FP16", "q8": "Q8", "q4": "Q4", "q2": "Q2"}
# Precision-ordered gradient (high precision -> aggressive).
_COLORS = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd"]


def _load_latest() -> dict[str, dict]:
    """Return the newest result per Ollama quant level, keyed by level."""
    found: dict[str, dict] = {}
    for level in _ORDER:
        matches = sorted(_RESULTS_DIR.glob(f"ollama_{level}_*.json"))
        if matches:
            with matches[-1].open(encoding="utf-8") as f:
                found[level] = json.load(f)
    return found


def _bars(ax, levels, values, colors, ylabel, title, fmt) -> None:
    """Draw one labelled bar panel."""
    bars = ax.bar([_LABELS[lvl] for lvl in levels], values,
                  color=colors, edgecolor="black", alpha=0.85)
    ax.bar_label(bars, fmt=fmt, padding=3, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlabel("Quantization level (precision decreasing →)")
    ax.grid(axis="y", alpha=0.3)


def main() -> None:
    """Generate figures/ollama_quant_comparison.png."""
    _FIGURES_DIR.mkdir(exist_ok=True)
    data = _load_latest()
    if not data:
        print("No ollama_*.json results found. Run experiments/run_ollama.py first.")
        return

    levels = [lvl for lvl in _ORDER if lvl in data]
    colors = [_COLORS[_ORDER.index(lvl)] for lvl in levels]
    ram = [data[lvl]["peak_ram_gb"] for lvl in levels]
    tput = [data[lvl]["throughput_tokens_per_sec"] for lvl in levels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    _bars(ax1, levels, ram, colors, "Peak RAM (GB)",
          "Memory shrinks with quantization", "%.2f")
    _bars(ax2, levels, tput, colors, "Throughput (tokens/sec)",
          "Throughput rises with quantization", "%.2f")
    fig.suptitle(
        "Real CPU Quantization via Ollama (GGUF) — Llama-3.1-8B, "
        "deterministic greedy decode (temp=0)",
        fontsize=12,
    )
    fig.tight_layout()

    out = _FIGURES_DIR / "ollama_quant_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")
    for lvl in levels:
        d = data[lvl]
        print(f"  {_LABELS[lvl]:5s} RAM={d['peak_ram_gb']:.2f} GB  "
              f"throughput={d['throughput_tokens_per_sec']:.2f} tok/s  "
              f"TPOT={d['tpot_seconds']:.3f} s/tok")


if __name__ == "__main__":
    main()
