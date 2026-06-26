"""Tests for src/ex05/airllm_runner.py.

AirLLM and transformers are mocked so the runner's control flow, metric
assembly, model-source resolution, and failure handling are validated fully
offline. The live path loads ~16 GB and is documented in results/airllm_*.json
and logs/airllm_*.log.
"""

from __future__ import annotations

import dataclasses
import sys
from unittest.mock import MagicMock, patch

import numpy as np

from ex05.airllm_runner import _make_model, _resolve_model_source, run_airllm


def _make_model_and_tokenizer():
    """Mocks that yield 3 generated tokens and decoded text."""
    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": np.zeros((1, 5), dtype=int)}
    tokenizer.decode.return_value = "Supervised vs unsupervised."
    out2 = MagicMock()
    out2.sequences = np.zeros((1, 8), dtype=int)  # 8 - 5 = 3 new tokens
    model = MagicMock()
    model.generate.side_effect = [MagicMock(), out2]  # pass 1 (TTFT), pass 2
    return model, tokenizer


class TestResolveModelSource:
    def test_returns_local_when_index_present(self, experiment_cfg, tmp_path):
        (tmp_path / "model.safetensors.index.json").write_text("{}")
        cfg = dataclasses.replace(experiment_cfg, layer_shards_path=str(tmp_path))
        assert _resolve_model_source(cfg) == str(tmp_path)

    def test_falls_back_to_repo_id(self, experiment_cfg, tmp_path):
        cfg = dataclasses.replace(experiment_cfg, layer_shards_path=str(tmp_path))
        assert _resolve_model_source(cfg) == cfg.model_id


class TestMakeModel:
    def test_returns_model_and_sets_is_stateful(self):
        class DummyModel:
            pass

        instance = DummyModel()
        fake_airllm = MagicMock()
        fake_airllm.AutoModel.from_pretrained.return_value = instance
        with patch.dict(sys.modules, {"airllm": fake_airllm}):
            result = _make_model("src", "tok", None, "shards")
        assert result is instance
        assert DummyModel._is_stateful is False
        fake_airllm.AutoModel.from_pretrained.assert_called_once()


class TestRunAirllm:
    def _patches(self, model, tokenizer):
        auto_tok = MagicMock()
        auto_tok.from_pretrained.return_value = tokenizer
        return (
            patch("ex05.airllm_runner.get_hf_token", return_value="tok"),
            patch("ex05.airllm_runner._make_model", return_value=model),
            patch("transformers.AutoTokenizer", auto_tok),
        )

    def test_happy_path(self, experiment_cfg, tmp_path):
        cfg = dataclasses.replace(experiment_cfg, layer_shards_path=str(tmp_path))
        model, tokenizer = _make_model_and_tokenizer()
        p1, p2, p3 = self._patches(model, tokenizer)
        with p1, p2, p3:
            result = run_airllm(cfg, None)
        assert result.scenario == "airllm_fp16"
        assert result.error is None
        assert result.output_text == "Supervised vs unsupervised."
        assert result.generated_tokens == 3
        assert result.ttft_seconds > 0
        assert result.prompt_tokens == len(cfg.prompt.split())

    def test_compression_in_scenario_name(self, experiment_cfg, tmp_path):
        cfg = dataclasses.replace(experiment_cfg, layer_shards_path=str(tmp_path))
        model, tokenizer = _make_model_and_tokenizer()
        p1, p2, p3 = self._patches(model, tokenizer)
        with p1, p2, p3:
            result = run_airllm(cfg, "4bit")
        assert result.scenario == "airllm_4bit"

    def test_failure_path_records_error(self, experiment_cfg, tmp_path):
        cfg = dataclasses.replace(experiment_cfg, layer_shards_path=str(tmp_path))
        with (
            patch("ex05.airllm_runner.get_hf_token", return_value="tok"),
            patch("ex05.airllm_runner._make_model",
                  side_effect=RuntimeError("CUDA required")),
        ):
            result = run_airllm(cfg, "8bit")
        assert result.scenario == "airllm_8bit"
        assert result.error == "RuntimeError: CUDA required"
        assert result.generated_tokens == 0
        assert result.output_text == ""
