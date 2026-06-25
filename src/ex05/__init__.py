"""EX05: Running a Massive LLM Locally with AirLLM and Quantization."""

from ex05.config import EconomicsConfig, ExperimentConfig, OllamaConfig, get_hf_token
from ex05.economics import (
    compute_api_cost,
    compute_cloud_gpu_cost,
    compute_onprem_cost,
    find_breakeven,
)
from ex05.metrics import InferenceTimer, MetricsResult, RamMonitor

__version__ = "1.0.0"
__all__ = [
    # config
    "ExperimentConfig",
    "EconomicsConfig",
    "OllamaConfig",
    "get_hf_token",
    # metrics
    "MetricsResult",
    "RamMonitor",
    "InferenceTimer",
    # economics
    "compute_api_cost",
    "compute_onprem_cost",
    "compute_cloud_gpu_cost",
    "find_breakeven",
]
