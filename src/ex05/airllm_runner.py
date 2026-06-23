"""AirLLM inference runner with configurable quantization.

TTFT measurement strategy: two separate generate() calls are made —
  1. max_new_tokens=1  → measures Time To First Token (prefill + 1 decode).
  2. max_new_tokens=N  → measures total generation for TPOT / throughput.
This double-loads all layers twice, which is expensive but accurate.
For AirLLM, TTFT ≈ TPOT because weights are reloaded from disk for every
token regardless of KV-cache state — the disk I/O is the dominant cost.
"""

from __future__ import annotations

import time
from typing import Optional

from ex05.config import ExperimentConfig, get_hf_token
from ex05.metrics import MetricsResult, RamMonitor

_COMPRESSION_MAP: dict[str, Optional[str]] = {
    "fp16": None,
    "4bit": "4bit",
    "8bit": "8bit",
    "2bit": "2bit",
}


def _make_model(model_id: str, token: str, compression: Optional[str], shards_path: str):
    """Instantiate the AirLLM AutoModel (deferred import)."""
    from airllm import AutoModel  # noqa: PLC0415
    return AutoModel.from_pretrained(
        model_id,
        token=token,
        compression=compression,
        layer_shards_saving_path=shards_path,
    )


def run_airllm(cfg: ExperimentConfig, compression: Optional[str]) -> MetricsResult:
    """Run AirLLM inference and return a full MetricsResult.

    Args:
        cfg: Loaded experiment configuration.
        compression: AirLLM compression level ("4bit", "8bit", "2bit") or
                     None for full FP16 precision via AirLLM.

    Returns:
        MetricsResult with TTFT, TPOT, throughput, RAM, and power metrics.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415

    token = get_hf_token()
    scenario = f"airllm_{compression or 'fp16'}"
    monitor = RamMonitor()
    error_msg: Optional[str] = None
    output_text = ""
    ttft = 0.0
    token_timestamps: list[float] = []
    total_rt = 0.0

    try:
        model = _make_model(cfg.model_id, token, compression, cfg.layer_shards_path)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token=token)
        input_ids = tokenizer(
            cfg.prompt,
            return_tensors="pt",
            return_attention_mask=False,
        )["input_ids"]

        monitor.start()

        # --- Pass 1: TTFT (1 token) ---
        t0 = time.perf_counter()
        out1 = model.generate(
            input_ids,
            max_new_tokens=1,
            use_cache=True,
            return_dict_in_generate=True,
        )
        ttft = time.perf_counter() - t0
        token_timestamps.append(t0 + ttft)

        # --- Pass 2: Full generation (N tokens) ---
        t_gen_start = time.perf_counter()
        out2 = model.generate(
            input_ids,
            max_new_tokens=cfg.max_new_tokens_experiment,
            use_cache=True,
            return_dict_in_generate=True,
        )
        t_gen_end = time.perf_counter()

        n_tokens = out2.sequences.shape[-1] - input_ids.shape[-1]
        step = (t_gen_end - t_gen_start) / max(n_tokens, 1)
        for i in range(1, n_tokens):
            token_timestamps.append(t_gen_start + i * step)

        total_rt = ttft + (t_gen_end - t_gen_start)
        output_text = tokenizer.decode(out2.sequences[0], skip_special_tokens=True)

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"

    finally:
        monitor.stop()
        monitor.join(timeout=2.0)

    return MetricsResult(
        scenario=scenario,
        prompt_tokens=len(cfg.prompt.split()),
        generated_tokens=len(token_timestamps),
        ttft_seconds=ttft,
        token_timestamps=token_timestamps,
        peak_ram_gb=monitor.peak_gb,
        total_runtime_seconds=total_rt,
        cpu_tdp_watts=cfg.cpu_tdp_watts,
        output_text=output_text,
        error=error_msg,
    )
