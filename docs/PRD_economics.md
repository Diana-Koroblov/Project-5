# PRD — Economic Feasibility Model (On-Prem vs API vs Cloud GPU)

**Version:** 1.00 | **Component:** `src/ex05/economics.py` | **Status:** Implemented

---

## 1. Mechanism description

A closed-form cost model that compares the per-request cost of serving the model
three ways and finds the **break-even volume** N\* where self-hosting overtakes a
third-party API. Every parameter (prices, hardware cost, electricity, token
counts, FX) is loaded from `config/economics_config.json` — no magic numbers in
source — so the analysis is transparent and reproducible.

## 2. Theoretical background & formulas

**API (flat per request):**
```
cost_api = (in_tok · in_price + out_tok · out_price) / 1e6 · USD→ILS
```
with cached input tokens discounted by `cache_discount_factor` (models the
Prompt/Context-Caching offered by PagedAttention-based providers).

**On-Prem (fixed monthly cost spread over volume):**
```
fixed_monthly = hardware_cost / (amort_years · 12)        # CAPEX
              + (power_W / 1000) · 720h · elec_rate        # OPEX electricity
              + maintenance_annual / 12                    # OPEX maintenance
cost_onprem(N) = fixed_monthly / N                         # per request
```

**Cloud GPU (flat per request):**
`(runtime_min / 60) · hourly_rate · USD→ILS`.

**Break-even (closed form):** On-Prem per-request `= fixed/N` equals the flat API
cost when
```
N* = fixed_monthly / cost_api
```
Below N\* the API is cheaper; above it, On-Prem wins.

## 3. Inputs / Outputs / Constraints

**Inputs:** `EconomicsConfig` — CAPEX (`hardware_cost_ils`, `amortization_years`),
OPEX (`electricity_kwh_ils`, `avg_power_watts`, `maintenance_cost_annual_ils`),
API pricing (`api_input/output_cost_per_1m_tokens`, `avg_input/output_tokens_per_request`,
`cache_discount_factor`), Cloud GPU (`cloud_gpu_hourly_usd`,
`cloud_gpu_runtime_per_request_minutes`), `usd_to_ils_rate`, request range.

**Outputs:** `compute_api_cost`, `compute_onprem_cost`, `compute_cloud_gpu_cost`,
`find_breakeven` → `results/economics_<ts>.json` + `figures/break_even.png`.

**Constraints:** `n_requests > 0` (raises `ValueError` otherwise); `find_breakeven`
raises if API cost is zero. All monetary values in ILS.

## 4. Results (current config)

| Metric | Value |
|---|---|
| On-Prem fixed cost | ₪247.52 / month |
| API cost/request (no cache / cached) | ₪0.000472 / ₪0.000447 |
| Cloud GPU cost/request | ₪0.015417 |
| **Break-even N\*** (no cache / cached) | **524,694 / 554,024 req/mo** |

**Recommendation:** for this low-token workload the API is far cheaper until
~525k req/mo; On-Prem wins only at very high sustained volume, or when privacy,
offline operation, or no-per-call-billing are hard requirements.

## 5. Alternatives considered

| Alternative | Why not chosen |
|---|---|
| Numerical/iterative break-even search | Unnecessary — the curves cross at a closed-form N\*; the analytic form is exact and testable. |
| Hardcoded prices | Violates the no-magic-numbers rule and kills reproducibility; all inputs live in config. |
| Ignoring prompt caching / Cloud GPU | The assignment calls out caching as a break-even shifter and offers Cloud GPU as a third curve; both are modelled. |

## 6. Success criteria & test scenarios

- **Success:** a break-even graph with all four curves, an explicit N\*, and a
  fully-stated assumptions table. ✅
- **Test scenarios (`tests/test_economics.py`, 100 % coverage):** basic API cost;
  cache < no-cache; zero-token → zero cost; On-Prem decreasing in N; `N=0`/negative
  raise; maintenance increases cost; very-high-N → ~0; cloud cost scales with
  runtime; break-even positive; cached break-even higher; zero-API-cost raises;
  **algebraic consistency** (at N\*, On-Prem ≈ API within 1 %). All offline, no
  live API calls.
