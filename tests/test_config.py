"""Tests for src/ex05/config.py.

Exercises the JSON loaders against the real config/ files (which ship in the
repo) and the HF_TOKEN environment lookup. Fully offline — no model download
and no network access.
"""

from __future__ import annotations

import dataclasses

import pytest

from ex05.config import (
    EconomicsConfig,
    ExperimentConfig,
    OllamaConfig,
    _load_json,
    get_hf_token,
)

# ---------------------------------------------------------------------------
# _load_json
# ---------------------------------------------------------------------------

class TestLoadJson:
    def test_returns_dict(self):
        data = _load_json("experiment_config.json")
        assert isinstance(data, dict)
        assert "model_id" in data

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            _load_json("does_not_exist.json")


# ---------------------------------------------------------------------------
# ExperimentConfig.load
# ---------------------------------------------------------------------------

class TestExperimentConfigLoad:
    def test_load_from_disk(self):
        cfg = ExperimentConfig.load()
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.model_id
        assert isinstance(cfg.quantization_levels, list)
        assert cfg.max_new_tokens > 0
        assert isinstance(cfg.cpu_tdp_watts, float)

    def test_frozen(self):
        cfg = ExperimentConfig.load()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.model_id = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EconomicsConfig.load
# ---------------------------------------------------------------------------

class TestEconomicsConfigLoad:
    def test_load_from_disk(self):
        cfg = EconomicsConfig.load()
        assert isinstance(cfg, EconomicsConfig)
        assert cfg.hardware_cost_ils > 0
        assert cfg.amortization_years > 0
        assert 0 <= cfg.cache_discount_factor <= 1
        assert cfg.usd_to_ils_rate > 0


# ---------------------------------------------------------------------------
# OllamaConfig.load
# ---------------------------------------------------------------------------

class TestOllamaConfigLoad:
    def test_load_from_disk(self):
        cfg = OllamaConfig.load()
        assert isinstance(cfg, OllamaConfig)
        assert cfg.host.startswith("http")
        assert cfg.num_predict > 0
        assert cfg.force_cpu is True
        assert "q4" in cfg.quant_levels
        assert cfg.prompt  # inherited from the top-level experiment prompt


# ---------------------------------------------------------------------------
# get_hf_token
# ---------------------------------------------------------------------------

class TestGetHfToken:
    def test_returns_token_when_set(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_dummy_token")
        assert get_hf_token() == "hf_dummy_token"

    def test_raises_when_unset(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        with pytest.raises(OSError, match="HF_TOKEN"):
            get_hf_token()
