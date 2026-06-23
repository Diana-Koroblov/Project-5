"""Economic feasibility analysis: On-Prem vs API vs Cloud GPU.

All monetary values are in Israeli Shekels (ILS) unless otherwise noted.
All input parameters are loaded from config/economics_config.json — no
magic numbers appear in this module.
"""

from __future__ import annotations

from ex05.config import EconomicsConfig


def compute_api_cost(cfg: EconomicsConfig, use_cache: bool = False) -> float:
    """Cost per request via third-party API (ILS).

    Args:
        cfg: Economics configuration.
        use_cache: Apply cache_discount_factor to input tokens (simulates
                   prompt-caching offered by providers like OpenAI / Claude).

    Returns:
        Cost per request in ILS.
    """
    input_rate = cfg.api_input_cost_per_1m_tokens
    if use_cache:
        input_rate *= cfg.cache_discount_factor

    cost_usd = (
        cfg.avg_input_tokens_per_request * input_rate
        + cfg.avg_output_tokens_per_request * cfg.api_output_cost_per_1m_tokens
    ) / 1_000_000
    return cost_usd * cfg.usd_to_ils_rate


def _monthly_fixed_cost(cfg: EconomicsConfig) -> float:
    """Total fixed monthly On-Prem cost (ILS): CAPEX + electricity + maintenance."""
    capex_monthly = cfg.hardware_cost_ils / (cfg.amortization_years * 12)
    hours_per_month = 24 * 30
    electricity_monthly = (
        (cfg.avg_power_watts / 1000) * hours_per_month * cfg.electricity_kwh_ils
    )
    maintenance_monthly = cfg.maintenance_cost_annual_ils / 12
    return capex_monthly + electricity_monthly + maintenance_monthly


def compute_onprem_cost(n_requests: int, cfg: EconomicsConfig) -> float:
    """Per-request On-Prem cost (ILS) at a given monthly request volume.

    Args:
        n_requests: Monthly request count (must be > 0).
        cfg: Economics configuration.

    Returns:
        Cost per request in ILS.

    Raises:
        ValueError: If n_requests is not positive.
    """
    if n_requests <= 0:
        raise ValueError("n_requests must be a positive integer.")
    return _monthly_fixed_cost(cfg) / n_requests


def compute_cloud_gpu_cost(cfg: EconomicsConfig) -> float:
    """Per-request cost on a rented Cloud GPU (ILS).

    Computed as: (runtime_per_request_minutes / 60) × hourly_rate × exchange_rate.
    """
    cost_usd = (
        cfg.cloud_gpu_runtime_per_request_minutes / 60
    ) * cfg.cloud_gpu_hourly_usd
    return cost_usd * cfg.usd_to_ils_rate


def find_breakeven(cfg: EconomicsConfig, use_cache: bool = False) -> float:
    """Monthly request volume N* where On-Prem cost equals API cost.

    Closed-form solution: N* = fixed_monthly / api_cost_per_request.

    Args:
        cfg: Economics configuration.
        use_cache: If True, use the cached-prompt API rate.

    Returns:
        N* as a float.

    Raises:
        ValueError: If the API cost per request is zero.
    """
    api_cost = compute_api_cost(cfg, use_cache=use_cache)
    if api_cost <= 0:
        raise ValueError("API cost per request is zero — cannot compute break-even.")
    return _monthly_fixed_cost(cfg) / api_cost
