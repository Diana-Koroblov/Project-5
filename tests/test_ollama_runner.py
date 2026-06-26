"""Tests for src/ex05/ollama_runner.py.

Covers the pure logic: request-payload construction, /api/ps RAM parsing (with
a mocked HTTP call), and MetricsResult assembly from streamed events. Fully
offline — the live streaming path (`_stream_generate`) needs a running Ollama
server and is exercised by `experiments/run_ollama.py`, not the unit suite.
"""

from __future__ import annotations

import dataclasses
import json
from unittest.mock import MagicMock, patch

from ex05.ollama_runner import (
    _assemble,
    _build_payload,
    _model_ram_gb,
    _stream_generate,
    run_ollama,
)

# ---------------------------------------------------------------------------
# _build_payload
# ---------------------------------------------------------------------------

class TestBuildPayload:
    def test_forces_cpu_when_enabled(self, ollama_cfg):
        payload = _build_payload(ollama_cfg, "m")
        assert payload["options"]["num_gpu"] == 0
        assert payload["stream"] is True
        assert payload["model"] == "m"

    def test_no_num_gpu_when_cpu_not_forced(self, ollama_cfg):
        cfg = dataclasses.replace(ollama_cfg, force_cpu=False)
        assert "num_gpu" not in _build_payload(cfg, "m")["options"]

    def test_passes_deterministic_options(self, ollama_cfg):
        opts = _build_payload(ollama_cfg, "m")["options"]
        assert opts["seed"] == 42
        assert opts["temperature"] == 0.0
        assert opts["num_predict"] == 50


# ---------------------------------------------------------------------------
# _model_ram_gb
# ---------------------------------------------------------------------------

class TestModelRamGb:
    def _mock_ps(self, payload: dict):
        cm = MagicMock()
        cm.__enter__.return_value.read.return_value = json.dumps(payload).encode()
        return cm

    def test_parses_cpu_footprint(self, ollama_cfg):
        body = {"models": [{"model": "m", "size": 5 * 1024**3, "size_vram": 0}]}
        with patch("ex05.ollama_runner.urllib.request.urlopen",
                   return_value=self._mock_ps(body)):
            assert abs(_model_ram_gb(ollama_cfg, "m") - 5.0) < 1e-6

    def test_subtracts_vram(self, ollama_cfg):
        body = {"models": [{"name": "m", "size": 6 * 1024**3,
                            "size_vram": 2 * 1024**3}]}
        with patch("ex05.ollama_runner.urllib.request.urlopen",
                   return_value=self._mock_ps(body)):
            assert abs(_model_ram_gb(ollama_cfg, "m") - 4.0) < 1e-6

    def test_missing_model_returns_zero(self, ollama_cfg):
        with patch("ex05.ollama_runner.urllib.request.urlopen",
                   return_value=self._mock_ps({"models": []})):
            assert _model_ram_gb(ollama_cfg, "m") == 0.0

    def test_request_error_returns_zero(self, ollama_cfg):
        with patch("ex05.ollama_runner.urllib.request.urlopen",
                   side_effect=OSError("refused")):
            assert _model_ram_gb(ollama_cfg, "m") == 0.0


# ---------------------------------------------------------------------------
# _assemble
# ---------------------------------------------------------------------------

class TestAssemble:
    def _events(self):
        # Three token chunks at t=1.0, 1.5, 2.0 then a done chunk at t=2.2.
        return [
            (1.0, {"response": "Hello"}),
            (1.5, {"response": " world"}),
            (2.0, {"response": "!"}),
            (2.2, {"done": True, "eval_count": 3, "prompt_eval_count": 7}),
        ]

    def test_scenario_and_text(self, ollama_cfg):
        r = _assemble(ollama_cfg, "q4", 0.5, self._events(), 5.19, None)
        assert r.scenario == "ollama_q4"
        assert r.output_text == "Hello world!"
        assert r.generated_tokens == 3
        assert r.prompt_tokens == 7
        assert r.peak_ram_gb == 5.19

    def test_ttft_and_total_runtime(self, ollama_cfg):
        r = _assemble(ollama_cfg, "q4", 0.5, self._events(), 5.19, None)
        assert abs(r.ttft_seconds - 0.5) < 1e-9       # 1.0 - 0.5
        assert abs(r.total_runtime_seconds - 1.7) < 1e-9  # 2.2 - 0.5

    def test_tpot_from_timestamps(self, ollama_cfg):
        r = _assemble(ollama_cfg, "q4", 0.5, self._events(), 5.19, None)
        assert abs(r.tpot_seconds - 0.5) < 1e-9  # gaps (0.5, 0.5) mean

    def test_empty_events(self, ollama_cfg):
        r = _assemble(ollama_cfg, "q4", 0.5, [], 0.0, "boom")
        assert r.generated_tokens == 0
        assert r.ttft_seconds == 0.0
        assert r.error == "boom"


# ---------------------------------------------------------------------------
# _stream_generate
# ---------------------------------------------------------------------------

class TestStreamGenerate:
    def _mock_response(self, lines):
        resp = MagicMock()
        resp.__iter__.return_value = iter(lines)
        cm = MagicMock()
        cm.__enter__.return_value = resp
        cm.__exit__.return_value = False
        return cm

    def test_yields_parsed_json_and_skips_blank_lines(self, ollama_cfg):
        lines = [b'{"response": "Hi"}\n', b'   \n', b'{"done": true}\n']
        with patch("ex05.ollama_runner.urllib.request.urlopen",
                   return_value=self._mock_response(lines)):
            out = list(_stream_generate(ollama_cfg, "m"))
        assert [chunk for _, chunk in out] == [{"response": "Hi"}, {"done": True}]
        assert all(isinstance(ts, float) for ts, _ in out)


# ---------------------------------------------------------------------------
# run_ollama
# ---------------------------------------------------------------------------

class TestRunOllama:
    def test_happy_path(self, ollama_cfg):
        events = [
            (1.0, {"response": "A"}),
            (1.5, {"response": "B"}),
            (2.0, {"done": True, "eval_count": 2, "prompt_eval_count": 5}),
        ]
        with (
            patch("ex05.ollama_runner._stream_generate", return_value=iter(events)),
            patch("ex05.ollama_runner._model_ram_gb", return_value=5.19),
        ):
            r = run_ollama(ollama_cfg, "q4", "m")
        assert r.scenario == "ollama_q4"
        assert r.output_text == "AB"
        assert r.generated_tokens == 2
        assert r.peak_ram_gb == 5.19
        assert r.error is None

    def test_stream_error_sets_error_and_zero_ram(self, ollama_cfg):
        with patch("ex05.ollama_runner._stream_generate",
                   side_effect=OSError("connection refused")):
            r = run_ollama(ollama_cfg, "q4", "m")
        assert r.error.startswith("OSError")
        assert r.peak_ram_gb == 0.0
        assert r.generated_tokens == 0
