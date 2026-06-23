"""Configuration loader for the EX05 experiment.

Loads experiment_config.json, economics_config.json, and the HF_TOKEN
from the .env file. All runtime-tunable values come from config files —
no magic numbers in source code.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _ROOT / "config"


def _load_json(filename: str) -> dict:
    """Load and return a JSON config file from the config/ directory."""
    path = _CONFIG_DIR / filename
    with path.open() as f:
        return json.load(f)


@dataclass(frozen=True)
class ExperimentConfig:
    """Typed settings for the inference experiment."""

    model_id: str
    prompt: str
    max_new_tokens: int
    max_new_tokens_experiment: int
    quantization_levels: list[str]
    layer_shards_path: str
    seed: int
    cpu_tdp_watts: float

    @classmethod
    def load(cls) -> ExperimentConfig:
        """Load from config/experiment_config.json."""
        d = _load_json("experiment_config.json")
        return cls(
            model_id=d["model_id"],
            prompt=d["prompt"],
            max_new_tokens=d["max_new_tokens"],
            max_new_tokens_experiment=d.get("max_new_tokens_experiment", 200),
            quantization_levels=d["quantization_levels"],
            layer_shards_path=d["layer_shards_path"],
            seed=d["seed"],
            cpu_tdp_watts=float(d["cpu_tdp_watts"]),
        )


@dataclass(frozen=True)
class EconomicsConfig:
    """Typed settings for the economic feasibility analysis."""

    hardware_cost_ils: float
    amortization_years: int
    electricity_kwh_ils: float
    avg_power_watts: float
    maintenance_cost_annual_ils: float
    api_input_cost_per_1m_tokens: float
    api_output_cost_per_1m_tokens: float
    avg_input_tokens_per_request: int
    avg_output_tokens_per_request: int
    monthly_request_range_min: int
    monthly_request_range_max: int
    cache_discount_factor: float
    cloud_gpu_hourly_usd: float
    cloud_gpu_runtime_per_request_minutes: float
    usd_to_ils_rate: float

    @classmethod
    def load(cls) -> EconomicsConfig:
        """Load from config/economics_config.json."""
        d = _load_json("economics_config.json")
        return cls(
            hardware_cost_ils=d["hardware_cost_ils"],
            amortization_years=d["amortization_years"],
            electricity_kwh_ils=d["electricity_kwh_ils"],
            avg_power_watts=d["avg_power_watts"],
            maintenance_cost_annual_ils=d["maintenance_cost_annual_ils"],
            api_input_cost_per_1m_tokens=d["api_input_cost_per_1m_tokens"],
            api_output_cost_per_1m_tokens=d["api_output_cost_per_1m_tokens"],
            avg_input_tokens_per_request=d["avg_input_tokens_per_request"],
            avg_output_tokens_per_request=d["avg_output_tokens_per_request"],
            monthly_request_range_min=d["monthly_request_range_min"],
            monthly_request_range_max=d["monthly_request_range_max"],
            cache_discount_factor=d["cache_discount_factor"],
            cloud_gpu_hourly_usd=d["cloud_gpu_hourly_usd"],
            cloud_gpu_runtime_per_request_minutes=d["cloud_gpu_runtime_per_request_minutes"],
            usd_to_ils_rate=d["usd_to_ils_rate"],
        )


def get_hf_token() -> str:
    """Return the Hugging Face token from the environment.

    Raises:
        EnvironmentError: If HF_TOKEN is not set.
    """
    token = os.getenv("HF_TOKEN")
    if not token:
        raise OSError(
            "HF_TOKEN is not set. "
            "Create a .env file with HF_TOKEN=hf_<your_token>."
        )
    return token
