"""Ollama CPU-quantization runner (GGUF / llama.cpp backend).

This is the experiment's *real* quantization path. AirLLM's bitsandbytes route
needs CUDA and cannot quantize on this AMD/CPU box, so Q4/Q8 there only ever
crashed. llama.cpp's GGUF quantization runs natively on the CPU, so this runner
actually exercises Q8/Q4/Q2 and measures the memory/speed/quality gradient the
assignment asks for.

We force CPU execution (`num_gpu=0`) for an apples-to-apples comparison with the
baseline/AirLLM runs, and stream tokens to measure wall-clock TTFT and per-token
timestamps. Peak RAM comes from Ollama's own `/api/ps` report: Ollama mmaps the
GGUF file, so its file-backed pages do not appear in the runner process's RSS
(especially on Windows). Ollama's `size` field is the authoritative resident
footprint, and with `num_gpu=0` the whole model sits in system RAM (size_vram=0).
"""

from __future__ import annotations

import json
import time
import urllib.request

from ex05.config import OllamaConfig
from ex05.metrics import MetricsResult


def _build_payload(cfg: OllamaConfig, model: str) -> dict:
    """Build the /api/generate request body (CPU-forced, deterministic)."""
    options: dict = {
        "temperature": cfg.temperature,
        "seed": cfg.seed,
        "num_predict": cfg.num_predict,
    }
    if cfg.force_cpu:
        options["num_gpu"] = 0  # keep all layers on CPU
    return {"model": model, "prompt": cfg.prompt, "stream": True, "options": options}


def _model_ram_gb(cfg: OllamaConfig, model: str) -> float:
    """Return the loaded model's CPU memory footprint (GB) via /api/ps."""
    try:
        with urllib.request.urlopen(f"{cfg.host}/api/ps", timeout=5) as resp:
            data = json.loads(resp.read())
    except Exception:
        return 0.0
    for m in data.get("models", []):
        if model in (m.get("model"), m.get("name")):
            return (m.get("size", 0) - m.get("size_vram", 0)) / 1024**3
    return 0.0


def _assemble(
    cfg: OllamaConfig,
    level: str,
    t0: float,
    events: list[tuple[float, dict]],
    peak_gb: float,
    error: str | None,
) -> MetricsResult:
    """Build a MetricsResult from streamed (timestamp, chunk) events (pure)."""
    token_timestamps = [ts for ts, c in events if c.get("response")]
    output_text = "".join(c["response"] for _, c in events if c.get("response"))
    ttft = (token_timestamps[0] - t0) if token_timestamps else 0.0
    final = next((c for _, c in events if c.get("done")), {})
    done_ts = next((ts for ts, c in events if c.get("done")), None)
    total_rt = (done_ts - t0) if done_ts is not None else 0.0
    prompt_tokens = final.get("prompt_eval_count") or len(cfg.prompt.split())
    return MetricsResult(
        scenario=f"ollama_{level}",
        prompt_tokens=prompt_tokens,
        generated_tokens=final.get("eval_count") or len(token_timestamps),
        ttft_seconds=ttft,
        token_timestamps=token_timestamps,
        peak_ram_gb=peak_gb,
        total_runtime_seconds=total_rt,
        cpu_tdp_watts=cfg.cpu_tdp_watts,
        output_text=output_text,
        error=error,
    )


def _stream_generate(cfg: OllamaConfig, model: str):
    """Yield (timestamp, chunk) for each newline-delimited JSON line."""
    data = json.dumps(_build_payload(cfg, model)).encode()
    req = urllib.request.Request(
        f"{cfg.host}/api/generate", data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:  # localhost, trusted
        for line in resp:
            stripped = line.strip()
            if stripped:
                yield time.perf_counter(), json.loads(stripped)


def run_ollama(cfg: OllamaConfig, level: str, model: str) -> MetricsResult:
    """Run one quantization level via Ollama and return a MetricsResult."""
    events: list[tuple[float, dict]] = []
    error_msg: str | None = None

    t0 = time.perf_counter()
    try:
        for ts, chunk in _stream_generate(cfg, model):
            events.append((ts, chunk))
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"

    peak_gb = _model_ram_gb(cfg, model) if error_msg is None else 0.0
    return _assemble(cfg, level, t0, events, peak_gb, error_msg)
