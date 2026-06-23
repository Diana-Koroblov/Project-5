"""Performance metrics collection for LLM inference experiments.

Provides:
  - RamMonitor  : background thread that samples and records peak RSS memory.
  - InferenceTimer : context manager that records wall-clock timing and
                     per-token timestamps for TTFT / TPOT calculation.
  - MetricsResult  : dataclass that aggregates all KPIs and serialises to dict.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import psutil


@dataclass
class MetricsResult:
    """All measured performance metrics for one inference run."""

    scenario: str
    prompt_tokens: int
    generated_tokens: int
    ttft_seconds: float
    token_timestamps: list[float] = field(default_factory=list)
    peak_ram_gb: float = 0.0
    total_runtime_seconds: float = 0.0
    cpu_tdp_watts: float = 45.0
    output_text: str = ""
    error: str | None = None

    @property
    def tpot_seconds(self) -> float:
        """Mean inter-token latency after the first token (seconds/token)."""
        if len(self.token_timestamps) < 2:
            return 0.0
        gaps = [
            self.token_timestamps[i] - self.token_timestamps[i - 1]
            for i in range(1, len(self.token_timestamps))
        ]
        return sum(gaps) / len(gaps)

    @property
    def throughput_tokens_per_sec(self) -> float:
        """Total tokens generated divided by total elapsed time."""
        if self.total_runtime_seconds <= 0:
            return 0.0
        return self.generated_tokens / self.total_runtime_seconds

    @property
    def estimated_power_wh(self) -> float:
        """Estimated energy consumed: (runtime_hours) × cpu_tdp_watts."""
        return (self.total_runtime_seconds / 3600) * self.cpu_tdp_watts

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "scenario": self.scenario,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "ttft_seconds": self.ttft_seconds,
            "tpot_seconds": self.tpot_seconds,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "peak_ram_gb": self.peak_ram_gb,
            "total_runtime_seconds": self.total_runtime_seconds,
            "estimated_power_wh": self.estimated_power_wh,
            "output_text": self.output_text,
            "error": self.error,
        }


class RamMonitor(threading.Thread):
    """Daemon thread that polls RSS memory and records the peak value."""

    def __init__(self, interval_sec: float = 0.5) -> None:
        super().__init__(daemon=True)
        self._interval = interval_sec
        self._stop_event = threading.Event()
        self.peak_gb: float = 0.0

    def run(self) -> None:
        """Sample process RSS every interval until stop() is called."""
        process = psutil.Process()
        while not self._stop_event.is_set():
            rss_gb = process.memory_info().rss / (1024**3)
            if rss_gb > self.peak_gb:
                self.peak_gb = rss_gb
            time.sleep(self._interval)

    def stop(self) -> None:
        """Signal the thread to exit on its next poll cycle."""
        self._stop_event.set()


class InferenceTimer:
    """Context manager that records wall-clock timing and token timestamps."""

    def __init__(self) -> None:
        self.t_start: float = 0.0
        self.t_first_token: float = 0.0
        self.t_end: float = 0.0
        self.token_timestamps: list[float] = []

    def __enter__(self) -> InferenceTimer:
        self.t_start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.t_end = time.perf_counter()

    def record_token(self) -> None:
        """Call once per generated token to capture its timestamp."""
        ts = time.perf_counter()
        self.token_timestamps.append(ts)
        if len(self.token_timestamps) == 1:
            self.t_first_token = ts

    @property
    def ttft(self) -> float:
        """Seconds from start to first token."""
        if not self.t_first_token:
            return self.total_runtime
        return self.t_first_token - self.t_start

    @property
    def total_runtime(self) -> float:
        """Total elapsed seconds."""
        return self.t_end - self.t_start
