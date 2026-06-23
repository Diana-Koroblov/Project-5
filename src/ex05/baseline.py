"""Baseline experiment: direct FP16 model loading without AirLLM.

Expected outcome: OOM crash or severe slowdown on systems with < 19 GB
free RAM, demonstrating why AirLLM and quantization are necessary.

Timeout: 10 minutes hard limit via SIGALRM on POSIX. On Windows no
timeout is enforced — press Ctrl+C manually if the process hangs.
"""

from __future__ import annotations

import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ex05.config import ExperimentConfig, get_hf_token
from ex05.metrics import InferenceTimer, MetricsResult, RamMonitor

_TIMEOUT_SEC = 600  # 10 minutes


def _arm_timeout() -> None:
    """Install a SIGALRM timeout on POSIX; silent no-op on Windows."""
    if sys.platform == "win32":
        return
    import signal

    def _handler(signum, frame):  # noqa: ARG001
        raise TimeoutError(f"Baseline exceeded {_TIMEOUT_SEC}s hard limit.")

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(_TIMEOUT_SEC)


def _disarm_timeout() -> None:
    """Cancel SIGALRM if on POSIX; silent no-op on Windows."""
    if sys.platform == "win32":
        return
    import signal
    signal.alarm(0)


def run_baseline(cfg: ExperimentConfig) -> MetricsResult:
    """Attempt direct FP16 inference and capture the failure mode.

    Args:
        cfg: Loaded experiment configuration.

    Returns:
        MetricsResult with peak RAM, elapsed time, and error details.
    """
    token = get_hf_token()
    monitor = RamMonitor()
    monitor.start()
    _arm_timeout()
    timer = InferenceTimer()
    error_msg: str | None = None
    output_text = ""

    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            torch_dtype=torch.float16,
            device_map="cpu",
            token=token,
        )
        inputs = tokenizer(cfg.prompt, return_tensors="pt")
        with timer:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=False,
            )
            timer.record_token()
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"

    finally:
        _disarm_timeout()
        monitor.stop()
        monitor.join(timeout=2.0)

    total_rt = timer.total_runtime or (time.perf_counter() - timer.t_start)
    return MetricsResult(
        scenario="baseline_fp16",
        prompt_tokens=len(cfg.prompt.split()),
        generated_tokens=len(timer.token_timestamps),
        ttft_seconds=timer.ttft,
        token_timestamps=timer.token_timestamps,
        peak_ram_gb=monitor.peak_gb,
        total_runtime_seconds=total_rt,
        cpu_tdp_watts=cfg.cpu_tdp_watts,
        output_text=output_text,
        error=error_msg,
    )
