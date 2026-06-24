"""Entry point for the Ollama CPU-quantization sweep (GGUF / llama.cpp).

Iterates over the quant levels in experiment_config.json -> ollama.quant_levels,
runs the standardized prompt against each on CPU, and writes one JSON result per
level to results/ollama_<level>_<ts>.json.

Usage:
    uv run python experiments/run_ollama.py            # all levels
    uv run python experiments/run_ollama.py q4 q2      # subset of levels

Prereq: an Ollama server is running (`ollama serve`) and the models are pulled
(`ollama pull llama3.1:8b-instruct-q4_K_M`, etc.).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from ex05.config import OllamaConfig
from ex05.ollama_runner import run_ollama

_ROOT = Path(__file__).resolve().parents[1]


def _save(result_dict: dict, level: str) -> Path:
    """Persist one result dict to results/ and return the path."""
    results_dir = _ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"ollama_{level}_{ts}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)
    return out_path


def main() -> None:
    """Run the Ollama quantization sweep (optionally a subset via argv)."""
    cfg = OllamaConfig.load()
    requested = [a.lower() for a in sys.argv[1:]]
    levels = {
        k: v for k, v in cfg.quant_levels.items()
        if not requested or k in requested
    }

    print("=" * 60)
    print("OLLAMA CPU-QUANTIZATION SWEEP (GGUF / llama.cpp)")
    print("=" * 60)
    print(f"Host       : {cfg.host}")
    print(f"Levels     : {list(levels)}")
    print(f"num_predict: {cfg.num_predict} | CPU-only: {cfg.force_cpu}")
    print("=" * 60)

    for level, model in levels.items():
        print(f"\n--- {level} ({model}) ---")
        result = run_ollama(cfg, level, model)
        out = _save(result.to_dict(), level)
        print(f"  TTFT       : {result.ttft_seconds:.2f} s")
        print(f"  TPOT       : {result.tpot_seconds:.3f} s/token")
        print(f"  Throughput : {result.throughput_tokens_per_sec:.2f} tok/s")
        print(f"  Peak RAM   : {result.peak_ram_gb:.2f} GB")
        print(f"  Power (est): {result.estimated_power_wh:.4f} Wh")
        print(f"  Tokens     : {result.generated_tokens}")
        print(f"  Error      : {result.error or 'None'}")
        print(f"  Saved      : {out.name}")


if __name__ == "__main__":
    main()
