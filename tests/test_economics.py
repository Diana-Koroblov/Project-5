"""Tests for src/ex05/economics.py.

All tests are fully offline — no API calls, no live data.
"""

from __future__ import annotations

import dataclasses

import pytest

from ex05.economics import (
    compute_api_cost,
    compute_cloud_gpu_cost,
    compute_onprem_cost,
    find_breakeven,
)

# ---------------------------------------------------------------------------
# compute_api_cost
# ---------------------------------------------------------------------------

class TestComputeApiCost:
    def test_basic_calculation(self, economics_cfg):
        # (50 × 0.15 + 200 × 0.60) / 1_000_000 × 3.7
        expected_usd = (50 * 0.15 + 200 * 0.60) / 1_000_000
        expected_ils = expected_usd * 3.7
        assert abs(compute_api_cost(economics_cfg) - expected_ils) < 1e-9

    def test_cached_is_cheaper_than_uncached(self, economics_cfg):
        cached = compute_api_cost(economics_cfg, use_cache=True)
        assert cached < compute_api_cost(economics_cfg)

    def test_cache_applies_only_to_input_tokens(self, economics_cfg):
        # With cache_discount_factor=0.1, cached input cost = 0.1× normal input cost
        normal = compute_api_cost(economics_cfg, use_cache=False)
        cached = compute_api_cost(economics_cfg, use_cache=True)
        # Output cost unchanged; only input is discounted
        assert cached < normal

    def test_zero_tokens_returns_zero(self, economics_cfg):
        cfg = dataclasses.replace(
            economics_cfg,
            avg_input_tokens_per_request=0,
            avg_output_tokens_per_request=0,
        )
        assert compute_api_cost(cfg) == 0.0


# ---------------------------------------------------------------------------
# compute_onprem_cost
# ---------------------------------------------------------------------------

class TestComputeOnpremCost:
    def test_cost_decreases_as_volume_increases(self, economics_cfg):
        assert compute_onprem_cost(100, economics_cfg) > compute_onprem_cost(
            1000, economics_cfg
        )

    def test_zero_requests_raises(self, economics_cfg):
        with pytest.raises(ValueError, match="positive"):
            compute_onprem_cost(0, economics_cfg)

    def test_negative_requests_raises(self, economics_cfg):
        with pytest.raises(ValueError):
            compute_onprem_cost(-1, economics_cfg)

    def test_maintenance_increases_cost(self, economics_cfg):
        cfg_no_maint = dataclasses.replace(
            economics_cfg, maintenance_cost_annual_ils=0.0
        )
        assert compute_onprem_cost(100, economics_cfg) > compute_onprem_cost(
            100, cfg_no_maint
        )

    def test_very_high_volume_approaches_zero(self, economics_cfg):
        assert compute_onprem_cost(1_000_000, economics_cfg) < 0.01


# ---------------------------------------------------------------------------
# compute_cloud_gpu_cost
# ---------------------------------------------------------------------------

class TestComputeCloudGpuCost:
    def test_basic_calculation(self, economics_cfg):
        # (0.5 min / 60) × $0.50/h × 3.7 ILS/$
        expected = (0.5 / 60) * 0.50 * 3.7
        assert abs(compute_cloud_gpu_cost(economics_cfg) - expected) < 1e-9

    def test_longer_runtime_increases_cost(self, economics_cfg):
        cfg_long = dataclasses.replace(
            economics_cfg, cloud_gpu_runtime_per_request_minutes=5.0
        )
        assert compute_cloud_gpu_cost(cfg_long) > compute_cloud_gpu_cost(economics_cfg)


# ---------------------------------------------------------------------------
# find_breakeven
# ---------------------------------------------------------------------------

class TestFindBreakeven:
    def test_breakeven_is_positive(self, economics_cfg):
        assert find_breakeven(economics_cfg) > 0

    def test_breakeven_with_cache_is_higher(self, economics_cfg):
        # Caching makes API cheaper → need more volume before On-Prem wins
        cached = find_breakeven(economics_cfg, use_cache=True)
        assert cached > find_breakeven(economics_cfg)

    def test_zero_api_cost_raises(self, economics_cfg):
        cfg = dataclasses.replace(
            economics_cfg,
            avg_input_tokens_per_request=0,
            avg_output_tokens_per_request=0,
        )
        with pytest.raises(ValueError, match="zero"):
            find_breakeven(cfg)

    def test_breakeven_algebraic_consistency(self, economics_cfg):
        n_star = find_breakeven(economics_cfg)
        api = compute_api_cost(economics_cfg)
        onprem = compute_onprem_cost(max(1, int(n_star)), economics_cfg)
        # At N*, costs should be approximately equal
        assert abs(api - onprem) / api < 0.01
