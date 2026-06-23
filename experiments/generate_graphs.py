"""Generate KPI comparison graphs from all experiment result JSON files.

Reads every non-economics JSON in results/, orders scenarios from Baseline
through each AirLLM quantization level, and produces three bar charts.

Usage:
    uv run python experiments/generate_graphs.py

Output:
    figures/ttft_comparison.png
    figures/ram_comparison.png
    figures/throughput_comparison.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIR = _ROOT / "results"
_FIGURES_DIR = _ROOT / "figures"

_ORDER = ["baseline_fp16", "airllm_fp16", "airllm_4bit", "airllm_8bit", "airllm_2bit"]
_LABELS = {
    "baseline_fp16": "Baseline\n(FP16 direct)",
    "airllm_fp16": "AirLLM\nFP16",
    "airllm_4bit": "AirLLM\nQ4",
    "airllm_8bit": "AirLLM\nQ8",
    "airllm_2bit": "AirLLM\nQ2",
}
_COLORS = {
    "baseline_fp16": "#d62728",
    "airllm_fp16": "#1f77b4",
    "airllm_4bit": "#2ca02c",
    "airllm_8bit": "#ff7f0e",
    "airllm_2bit": "#9467bd",
}


def _load_results() -> dict[str, dict[str, Any]]:
    """Load all result JSON files, keyed by scenario name."""
    results: dict[str, dict[str, Any]] = {}
    for path in sorted(_RESULTS_DIR.glob("*.json")):
        if "economics" in path.name:
            continue
        with path.open() as f:
            data = json.load(f)
        scenario = data.get("scenario", path.stem)
        results[scenario] = data
    return results


def _bar_chart(
    results: dict,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    note: str = "",
) -> None:
    """Save a colour-coded bar chart comparing one metric across all scenarios."""
    scenarios = [s for s in _ORDER if s in results]
    values = [results[s].get(metric) or 0.0 for s in scenarios]
    labels = [_LABELS.get(s, s) for s in scenarios]
    colors = [_COLORS.get(s, "steelblue") for s in scenarios]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", alpha=0.85)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if note:
        ax.text(0.01, 0.97, note, transform=ax.transAxes,
                fontsize=8, va="top", color="grey")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def main() -> None:
    """Generate all three comparison figures."""
    _FIGURES_DIR.mkdir(exist_ok=True)
    results = _load_results()

    if not results:
        print("No result files found in results/. Run experiments first.")
        return

    print(f"Loaded scenarios: {list(results.keys())}")

    _bar_chart(results, "ttft_seconds", "Time to First Token (s)",
               "TTFT — Baseline vs AirLLM Quantization Levels",
               _FIGURES_DIR / "ttft_comparison.png",
               note="Lower is better. Baseline may show partial/crash value.")

    _bar_chart(results, "peak_ram_gb", "Peak RAM (GB)",
               "Peak RAM — Baseline vs AirLLM Quantization Levels",
               _FIGURES_DIR / "ram_comparison.png",
               note="Lower is better.")

    _bar_chart(results, "throughput_tokens_per_sec", "Throughput (tokens/sec)",
               "Throughput — Baseline vs AirLLM Quantization Levels",
               _FIGURES_DIR / "throughput_comparison.png",
               note="Higher is better.")

    print(f"\nAll figures saved to {_FIGURES_DIR}")


if __name__ == "__main__":
    main()
