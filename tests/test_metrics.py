"""Tests for src/ex05/metrics.py.

All tests are fully offline — no model, no network, no GPU required.
psutil is mocked where needed to avoid measuring the test process itself.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from ex05.metrics import InferenceTimer, MetricsResult, RamMonitor

# ---------------------------------------------------------------------------
# MetricsResult
# ---------------------------------------------------------------------------

class TestMetricsResult:
    def _make(self, **kwargs) -> MetricsResult:
        defaults = {
            "scenario": "test",
            "prompt_tokens": 10,
            "generated_tokens": 5,
            "ttft_seconds": 1.0,
            "token_timestamps": [1.0, 1.5, 2.5, 4.0],
            "peak_ram_gb": 4.0,
            "total_runtime_seconds": 10.0,
            "cpu_tdp_watts": 45.0,
            "output_text": "hello",
        }
        defaults.update(kwargs)
        return MetricsResult(**defaults)

    def test_tpot_mean_of_gaps(self):
        r = self._make(token_timestamps=[1.0, 1.5, 2.5, 4.0])
        # gaps: 0.5, 1.0, 1.5 → mean = 1.0
        assert abs(r.tpot_seconds - 1.0) < 1e-9

    def test_tpot_single_token_is_zero(self):
        assert self._make(token_timestamps=[1.0]).tpot_seconds == 0.0

    def test_tpot_empty_is_zero(self):
        assert self._make(token_timestamps=[]).tpot_seconds == 0.0

    def test_throughput_basic(self):
        r = self._make(generated_tokens=20, total_runtime_seconds=10.0)
        assert abs(r.throughput_tokens_per_sec - 2.0) < 1e-9

    def test_throughput_zero_runtime_is_zero(self):
        assert self._make(total_runtime_seconds=0.0).throughput_tokens_per_sec == 0.0

    def test_estimated_power_one_hour(self):
        r = self._make(total_runtime_seconds=3600.0, cpu_tdp_watts=45.0)
        assert abs(r.estimated_power_wh - 45.0) < 1e-9

    def test_estimated_power_half_hour(self):
        r = self._make(total_runtime_seconds=1800.0, cpu_tdp_watts=60.0)
        assert abs(r.estimated_power_wh - 30.0) < 1e-9

    def test_to_dict_has_all_keys(self):
        keys = set(self._make().to_dict().keys())
        required = {
            "scenario", "prompt_tokens", "generated_tokens", "ttft_seconds",
            "tpot_seconds", "throughput_tokens_per_sec", "peak_ram_gb",
            "total_runtime_seconds", "estimated_power_wh", "output_text", "error",
        }
        assert required.issubset(keys)

    def test_error_field_serialised(self):
        r = self._make(error="MemoryError: foo")
        assert r.to_dict()["error"] == "MemoryError: foo"


# ---------------------------------------------------------------------------
# RamMonitor
# ---------------------------------------------------------------------------

class TestRamMonitor:
    def test_peak_tracks_maximum(self):
        rss_sequence = [2.0e9, 4.0e9, 3.0e9, 4.5e9, 1.0e9]
        call_idx = {"n": 0}

        def fake_memory_info():
            m = MagicMock()
            idx = min(call_idx["n"], len(rss_sequence) - 1)
            m.rss = rss_sequence[idx]
            call_idx["n"] += 1
            return m

        with patch("psutil.Process") as mock_proc:
            mock_proc.return_value.memory_info.side_effect = fake_memory_info
            monitor = RamMonitor(interval_sec=0.02)
            monitor.start()
            time.sleep(0.15)
            monitor.stop()
            monitor.join(timeout=1.0)

        assert monitor.peak_gb >= 4.5 / 1024**3 * 1e9 / 1e9  # ≥ 4.5 GB


# ---------------------------------------------------------------------------
# InferenceTimer
# ---------------------------------------------------------------------------

class TestInferenceTimer:
    def test_ttft_reflects_first_token_timing(self):
        timer = InferenceTimer()
        with timer:
            time.sleep(0.05)
            timer.record_token()
            time.sleep(0.05)

        assert timer.ttft >= 0.04
        assert timer.total_runtime >= timer.ttft

    def test_multiple_tokens_recorded(self):
        timer = InferenceTimer()
        with timer:
            for _ in range(5):
                timer.record_token()
                time.sleep(0.01)

        assert len(timer.token_timestamps) == 5

    def test_no_tokens_ttft_equals_runtime(self):
        timer = InferenceTimer()
        with timer:
            time.sleep(0.02)

        assert timer.ttft == timer.total_runtime
