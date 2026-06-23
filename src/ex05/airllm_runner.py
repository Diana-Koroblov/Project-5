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
from pathlib import Path

from ex05.config import ExperimentConfig, get_hf_token
from ex05.metrics import MetricsResult, RamMonitor

_COMPRESSION_MAP: dict[str, str | None] = {
    "fp16": None,
    "4bit": "4bit",
    "8bit": "8bit",
    "2bit": "2bit",
}


def _resolve_model_source(cfg: ExperimentConfig) -> str:
    """Return the local model directory if the full model was already
    downloaded there, otherwise fall back to the HF repo id.

    Using the local download avoids re-fetching ~16 GB into the HF hub cache
    and sidesteps the Windows symlink-privilege error (WinError 1314) that the
    hub cache triggers without Developer Mode / admin rights.
    """
    local = Path(cfg.layer_shards_path)
    if (local / "model.safetensors.index.json").exists():
        return str(local)
    return cfg.model_id


def _make_model(
    model_source: str, token: str, compression: str | None, shards_path: str
):
    """Instantiate the AirLLM AutoModel (deferred import).

    device="cpu" is forced because this experiment runs CPU-only (no usable
    CUDA GPU); AirLLM otherwise defaults to "cuda:0". The token kwarg is
    `hf_token` in AirLLM's API, not `token`. When model_source is a local
    directory, AirLLM splits it in place (no download); split shards are
    written to shards_path/splitted_model[.<compression>].
    """
    from airllm import AutoModel  # noqa: PLC0415
    model = AutoModel.from_pretrained(
        model_source,
        device="cpu",
        hf_token=token,
        compression=compression,
        layer_shards_saving_path=shards_path,
    )
    # transformers >= ~4.45 GenerationMixin.generate() reads the class attr
    # `_is_stateful` (defined on PreTrainedModel). AirLLM's model inherits only
    # GenerationMixin, so it is missing. AirLLM is not a stateful model, so
    # False is the correct value.
    if not hasattr(type(model), "_is_stateful"):
        type(model)._is_stateful = False
    return model


def run_airllm(cfg: ExperimentConfig, compression: str | None) -> MetricsResult:
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
    model_source = _resolve_model_source(cfg)
    monitor = RamMonitor()
    error_msg: str | None = None
    output_text = ""
    ttft = 0.0
    token_timestamps: list[float] = []
    total_rt = 0.0

    try:
        model = _make_model(model_source, token, compression, cfg.layer_shards_path)
        tokenizer = AutoTokenizer.from_pretrained(model_source, token=token)
        input_ids = tokenizer(
            cfg.prompt,
            return_tensors="pt",
            return_attention_mask=False,
        )["input_ids"]

        monitor.start()

        # --- Pass 1: TTFT (1 token) ---
        # use_cache=False is required: AirLLM does not support transformers'
        # new Cache class and forces it off internally; passing True makes
        # generate() build a cache object that breaks prepare_inputs_for_generation.
        t0 = time.perf_counter()
        model.generate(
            input_ids,
            max_new_tokens=1,
            use_cache=False,
            return_dict_in_generate=True,
        )
        ttft = time.perf_counter() - t0
        token_timestamps.append(t0 + ttft)

        # --- Pass 2: Full generation (N tokens) ---
        t_gen_start = time.perf_counter()
        out2 = model.generate(
            input_ids,
            max_new_tokens=cfg.max_new_tokens_experiment,
            use_cache=False,
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
        if monitor.is_alive():
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
