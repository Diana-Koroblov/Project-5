"""Entry point for the baseline (FP16 direct) experiment.

Usage:
    uv run python experiments/run_baseline.py

Expected outcome: OOM crash or 10-minute timeout, demonstrating that
16 GB RAM is insufficient for FP16 inference without AirLLM.
Results are saved to results/baseline_<timestamp>.json.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from ex05.baseline import run_baseline
from ex05.config import ExperimentConfig

_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Run the baseline experiment and persist results."""
    cfg = ExperimentConfig.load()

    print("=" * 60)
    print("BASELINE EXPERIMENT — Direct FP16 Loading")
    print("=" * 60)
    print(f"Model  : {cfg.model_id}")
    print(f"Prompt : {cfg.prompt[:60]}...")
    print("WARNING: Expected to OOM or time out. Monitor system RAM.")
    print("=" * 60)

    result = run_baseline(cfg)

    results_dir = _ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"baseline_{ts}.json"
    with out_path.open("w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print("\n--- Results ---")
    print(f"Peak RAM    : {result.peak_ram_gb:.2f} GB")
    print(f"Runtime     : {result.total_runtime_seconds:.1f}s")
    print(f"Power (est) : {result.estimated_power_wh:.4f} Wh")
    print(f"Error       : {result.error or 'None (model ran)'}")
    print(f"Saved to    : {out_path}")


if __name__ == "__main__":
    main()
