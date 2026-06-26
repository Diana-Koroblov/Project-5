"""Tests for src/ex05/baseline.py.

The baseline runner loads the live FP16 model via transformers. Here the HF
token, tokenizer, and model are all mocked so the control flow, metric
assembly, and failure handling run fully offline. The real model run is
documented in results/baseline_*.json and logs/baseline_run_*.log.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ex05.baseline import _arm_timeout, _disarm_timeout, run_baseline


def _run_mocked(cfg, model, tokenizer):
    """Run run_baseline with HF token/tokenizer/model patched out."""
    auto_tok = MagicMock()
    auto_tok.from_pretrained.return_value = tokenizer
    auto_model = MagicMock()
    auto_model.from_pretrained.return_value = model
    with (
        patch("ex05.baseline.get_hf_token", return_value="tok"),
        patch("ex05.baseline.AutoTokenizer", auto_tok),
        patch("ex05.baseline.AutoModelForCausalLM", auto_model),
    ):
        return run_baseline(cfg)


def _make_mocks():
    """A tokenizer/model pair that produce one decoded token successfully."""
    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": [[1, 2, 3]]}
    tokenizer.decode.return_value = "Supervised learning uses labels."
    model = MagicMock()
    model.generate.return_value = [[1, 2, 3, 4]]
    return model, tokenizer


class TestRunBaseline:
    def test_happy_path(self, experiment_cfg):
        model, tokenizer = _make_mocks()
        result = _run_mocked(experiment_cfg, model, tokenizer)
        assert result.scenario == "baseline_fp16"
        assert result.error is None
        assert result.output_text == "Supervised learning uses labels."
        assert result.generated_tokens == 1
        assert result.prompt_tokens == len(experiment_cfg.prompt.split())
        assert result.cpu_tdp_watts == experiment_cfg.cpu_tdp_watts
        model.generate.assert_called_once()

    def test_generate_failure_records_error(self, experiment_cfg):
        model, tokenizer = _make_mocks()
        model.generate.side_effect = RuntimeError("out of memory")
        result = _run_mocked(experiment_cfg, model, tokenizer)
        assert result.error == "RuntimeError: out of memory"
        assert result.output_text == ""
        assert result.generated_tokens == 0

    def test_load_failure_records_error(self, experiment_cfg):
        auto_tok = MagicMock()
        auto_tok.from_pretrained.side_effect = OSError("gated repo")
        with (
            patch("ex05.baseline.get_hf_token", return_value="tok"),
            patch("ex05.baseline.AutoTokenizer", auto_tok),
        ):
            result = run_baseline(experiment_cfg)
        assert result.error.startswith("OSError")
        assert result.generated_tokens == 0


class TestTimeoutHelpers:
    def test_win32_helpers_are_noops(self):
        # On Windows both are no-ops and must return without raising; on POSIX
        # arm() schedules SIGALRM which disarm() immediately cancels.
        assert _arm_timeout() is None
        assert _disarm_timeout() is None
