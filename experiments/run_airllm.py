"""Entry point for the AirLLM quantization sweep.

Iterates over all quantization_levels in experiment_config.json and runs
one inference per level, saving results to results/airllm_<level>_<ts>.json.

Usage:
    uv run python experiments/run_airllm.py

Tip: Clear RAM to ≤ 4 GB before running. Each level takes 30–90 minutes
depending on hardware and quantization aggressiveness.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from ex05.airllm_runner import _COMPRESSION_MAP, run_airllm
from ex05.config import ExperimentConfig

_ROOT = Path(__file__).resolve().parents[1]


def _save(result_dict: dict, label: str) -> Path:
    """Persist a result dict to results/ and return the path."""
    results_dir = _ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"airllm_{label}_{ts}.json"
    with out_path.open("w") as f:
        json.dump(result_dict, f, indent=2)
    return out_path


def main() -> None:
    """Run the full quantization sweep."""
    cfg = ExperimentConfig.load()

    print("=" * 60)
    print("AIRLLM QUANTIZATION SWEEP")
    print("=" * 60)
    print(f"Model  : {cfg.model_id}")
    print(f"Levels : {cfg.quantization_levels}")
    print(f"Tokens : {cfg.max_new_tokens_experiment} (experiment)")
    print("=" * 60)

    for level in cfg.quantization_levels:
        compression = _COMPRESSION_MAP.get(level)
        print(f"\n--- Level: {level} (compression={compression!r}) ---")

        result = run_airllm(cfg, compression=compression)
        out_path = _save(result.to_dict(), label=level)

        print(f"  TTFT        : {result.ttft_seconds:.2f} s")
        print(f"  TPOT        : {result.tpot_seconds:.3f} s/token")
        print(f"  Throughput  : {result.throughput_tokens_per_sec:.4f} tok/s")
        print(f"  Peak RAM    : {result.peak_ram_gb:.2f} GB")
        print(f"  Power (est) : {result.estimated_power_wh:.4f} Wh")
        print(f"  Tokens gen  : {result.generated_tokens}")
        print(f"  Error       : {result.error or 'None'}")
        print(f"  Saved to    : {out_path}")


if __name__ == "__main__":
    main()
