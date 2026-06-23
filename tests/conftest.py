"""Shared pytest fixtures for EX05 tests."""

from __future__ import annotations

import pytest

from ex05.config import EconomicsConfig, ExperimentConfig


@pytest.fixture
def experiment_cfg() -> ExperimentConfig:
    """Minimal ExperimentConfig for testing (no file I/O required)."""
    return ExperimentConfig(
        model_id="test/model",
        prompt="Test prompt.",
        max_new_tokens=10,
        max_new_tokens_experiment=200,
        quantization_levels=["4bit"],
        layer_shards_path="./model_shards",
        seed=42,
        cpu_tdp_watts=45.0,
    )


@pytest.fixture
def economics_cfg() -> EconomicsConfig:
    """Standard EconomicsConfig with deterministic values for testing."""
    return EconomicsConfig(
        hardware_cost_ils=6000.0,
        amortization_years=3,
        electricity_kwh_ils=0.60,
        avg_power_watts=45.0,
        maintenance_cost_annual_ils=300.0,
        api_input_cost_per_1m_tokens=0.15,
        api_output_cost_per_1m_tokens=0.60,
        avg_input_tokens_per_request=50,
        avg_output_tokens_per_request=200,
        monthly_request_range_min=1,
        monthly_request_range_max=100000,
        cache_discount_factor=0.1,
        cloud_gpu_hourly_usd=0.50,
        cloud_gpu_runtime_per_request_minutes=0.5,
        usd_to_ils_rate=3.7,
    )
