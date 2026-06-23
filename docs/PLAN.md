# PLAN — EX05: Architecture & Technical Design

**Version:** 1.00  
**Status:** Awaiting Approval  
**Author:** Diana Koroblov  
**Date:** 2026-06-23  
**Reference:** PRD v1.00

---

## 1. Overview

This document describes the technical architecture, environment design, experiment pipeline, and measurement strategy for EX05. It translates the requirements in `PRD.md` into a concrete, implementable design.

---

## 2. Repository Structure

```
ex05-airllm/
├── README.md                   # Technical report (the final deliverable)
├── pyproject.toml              # Single source of truth for deps + uv config
├── uv.lock                     # Locked dependency graph
├── .env-example                # Placeholder for secrets (HF_TOKEN, etc.)
├── .gitignore
│
├── docs/
│   ├── PRD.md
│   ├── PLAN.md
│   └── TODO.md
│
├── src/
│   └── ex05/
│       ├── __init__.py
│       ├── config.py           # Load config + .env; expose typed settings
│       ├── metrics.py          # TTFT, TPOT, throughput, RAM measurement
│       ├── baseline.py         # Baseline (direct HF) inference runner
│       ├── airllm_runner.py    # AirLLM inference runner
│       └── economics.py        # Break-even cost calculation logic
│
├── experiments/
│   ├── run_baseline.py         # Entry point: baseline experiment
│   ├── run_airllm.py           # Entry point: AirLLM + quantization sweep
│   └── run_economics.py        # Entry point: economic analysis + graph
│
├── results/
│   └── .gitkeep               # Raw JSON/CSV output lands here
│
├── figures/
│   └── .gitkeep               # Generated graphs (PNG) land here
│
├── tests/
│   ├── conftest.py
│   ├── test_metrics.py
│   └── test_economics.py
│
└── config/
    ├── experiment_config.json  # Prompt, max_tokens, quantization levels
    └── economics_config.json   # Prices, hardware cost, electricity rate
```

All source files are capped at 150 lines per the software guidelines.

---

## 3. Environment Setup (uv)

### 3.1 Rationale for uv

`uv` is the mandatory package manager per the software guidelines. It provides deterministic installs via `uv.lock`, fast resolution, and a clean `pyproject.toml`-based workflow. No `pip`, `venv`, or `virtualenv` calls are permitted anywhere in the project.

### 3.2 Python Version

**Python 3.11** — The highest version confirmed compatible with `airllm`, `bitsandbytes`, and `transformers` as of 2025. Python 3.12+ introduces breaking changes in some dependencies.

### 3.3 Setup Commands

```bash
# Install uv (one-time)
pip install uv

# Create isolated environment and install all dependencies
uv sync

# Add a package (example)
uv add airllm

# Run any script
uv run python experiments/run_baseline.py

# Run tests
uv run pytest tests/

# Lint
uv run ruff check .
```

### 3.4 Secrets Management

The Hugging Face token is loaded exclusively from `.env` via `python-dotenv`. The `.env` file is listed in `.gitignore`. An `.env-example` file with a dummy placeholder is committed.

```
# .env-example
HF_TOKEN=hf_YOUR_TOKEN_HERE
```

---

## 4. Configuration Architecture

All runtime-tunable values live in `config/`. No magic numbers appear in source code.

### `config/experiment_config.json`
```json
{
  "version": "1.00",
  "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "prompt": "Explain the difference between supervised and unsupervised learning in three paragraphs.",
  "max_new_tokens": 200,
  "quantization_levels": ["4bit", "8bit"],
  "layer_shards_path": "./model_shards",
  "seed": 42
}
```

### `config/economics_config.json`
```json
{
  "version": "1.00",
  "hardware_cost_ils": 6000,
  "amortization_years": 3,
  "electricity_kwh_ils": 0.60,
  "avg_power_watts": 45,
  "api_provider": "openai_gpt4o_mini",
  "api_input_cost_per_1m_tokens": 0.15,
  "api_output_cost_per_1m_tokens": 0.60,
  "avg_input_tokens_per_request": 50,
  "avg_output_tokens_per_request": 200,
  "monthly_request_range": [1, 100000]
}
```

---

## 5. Experiment Pipeline Design

### 5.1 Standardized Prompt

A single fixed prompt is used across all scenarios to ensure fair comparison:

> *"Explain the difference between supervised and unsupervised learning in three paragraphs."*

This prompt is short enough (≈12 tokens) to keep Prefill fast, while requesting 200 output tokens to stress the Decode phase.

### 5.2 Experiment A — Baseline (Direct HF Execution)

**Goal:** Demonstrate that loading Meta-Llama-3.1-8B-Instruct in FP16 directly into RAM fails or hangs.

**Design:**
1. Load `AutoTokenizer` and `AutoModelForCausalLM` from Hugging Face in FP16 (`torch_dtype=torch.float16`).
2. Start RAM monitoring (background thread via `psutil`).
3. Attempt `model.generate()` with the standardized prompt.
4. Capture one of:
   - `torch.cuda.OutOfMemoryError` / OS kill signal
   - RAM exceeding 14 GB + system becoming unresponsive
   - Time-to-first-token exceeding a 10-minute timeout
5. Log the failure mode, peak RAM, and elapsed time to `results/baseline.json`.

**Expected outcome:** OOM error or unresponsive system within 2–5 minutes of loading.

**Safety measures:** Set `max_new_tokens=10` to limit runaway generation; wrap the entire call in a try/except with a timeout guard.

### 5.3 Experiment B — AirLLM + Quantization Sweep

**Goal:** Run the same prompt with AirLLM at multiple quantization levels and record all KPIs.

**Design:**

For each quantization level in `["4bit", "8bit"]` (and FP16 as a reference if memory allows):

1. Instantiate `AirLLMLlama` (or `AutoModel` for the Qwen fallback) with the configured `layer_shards_saving_path` and `compression` level.
2. Start background RAM monitor thread.
3. Record `t_start` (wall clock).
4. Call `model.generate()` with a callback hook that records the timestamp of the **first token** (→ TTFT).
5. Record timestamps for each subsequent token (→ TPOT per token).
6. After generation completes, record `t_end` and final peak RAM.
7. Compute and persist: TTFT, mean TPOT, throughput (tokens/s), peak RAM, total elapsed, output text.
8. Conduct qualitative assessment of output at each level (1–5 coherence score).

**AirLLM mechanics (per lecture):** AirLLM streams one transformer layer at a time from disk using `mmap`, computes the forward pass, retains only the hidden state tensor in RAM, then evicts the layer weights. This mirrors the OS virtual memory / paging mechanism — each layer is a "page" fetched on demand.

**Layer shards path:** Explicitly set to `./model_shards` (or a fast external drive if available) to avoid writing tens of gigabytes to the OS partition.

---

## 6. Metrics Architecture

### 6.1 `src/ex05/metrics.py`

This module exposes:

- `RamMonitor` — a `threading.Thread` subclass that samples `psutil.Process().memory_info().rss` every 500 ms and records the peak.
- `InferenceTimer` — a context manager that wraps the generation call, captures `t_start`, intercepts the first-token callback to record TTFT, then records `t_end` and computes TPOT and throughput from the token timestamps list.
- `MetricsResult` — a `dataclass` holding all measured values, with a `.to_dict()` method for JSON serialization.

### 6.2 Token Callback Strategy

AirLLM (like the underlying `transformers` library) supports a `StreamingStdOutCallbackHandler` or a custom `StoppingCriteria` / `LogitsProcessor`. We will implement a lightweight `TokenTimestampLogger` that appends `time.perf_counter()` on each token generation step, enabling precise per-token latency.

### 6.3 Results Persistence

Every experiment run serializes its `MetricsResult` to `results/<scenario>_<timestamp>.json`. The analysis notebook and graph scripts read from these files, not from in-memory state, ensuring reproducibility.

---

## 7. Economic Analysis Architecture

### 7.1 `src/ex05/economics.py`

Computes:

**API cost per request:**
```
cost_api = (input_tokens × price_input + output_tokens × price_output) / 1_000_000
```

**On-Prem cost per request (as a function of monthly volume N):**
```
capex_monthly   = hardware_cost / (amortization_years × 12)
opex_monthly    = (avg_power_W / 1000) × avg_hours_per_month × electricity_rate
fixed_monthly   = capex_monthly + opex_monthly
cost_onprem(N)  = fixed_monthly / N   (per request, decreasing with N)
```

**Break-even point:** Solve `cost_api = cost_onprem(N)` → `N* = fixed_monthly / cost_api`.

**Prompt caching note:** For API providers with context caching (OpenAI, Claude), repeated requests sharing a fixed system prompt pay a reduced rate on cached input tokens (~10% of normal input price). The economics module accepts a `cache_discount_factor` parameter to model this effect.

### 7.2 Break-Even Graph

`experiments/run_economics.py` generates a Matplotlib figure showing:
- X-axis: monthly request volume (log scale)
- Y-axis: cumulative cost (ILS or USD)
- Two lines: API cost (linear) vs On-Prem cost (fixed monthly amortized to per-request)
- Vertical dashed line at N* (break-even)
- Saved to `figures/break_even.png`

---

## 8. Architectural Decisions and Trade-offs

| Decision | Choice | Rationale |
|---|---|---|
| Model format | SafeTensors (HF) | Required by AirLLM's mmap-based layer loading; also safer than pickle-based `.bin` |
| Quantization backend | `bitsandbytes` via AirLLM | Native integration; supports 4-bit and 8-bit on CPU |
| RAM monitor granularity | 500 ms polling | Fine enough to catch peak without significant overhead |
| Token timing mechanism | `perf_counter()` callback | Sub-millisecond precision; avoids CUDA event overhead (no GPU) |
| Configuration format | JSON (not YAML) | No extra dependency; human-readable; strict schema |
| Experiment entry points | Separate scripts per experiment | Allows running phases independently; avoids reloading model between experiments |
| Layer shards location | Configurable path | Prevents OS-drive flooding; allows redirection to fast NVMe partition |

---

## 9. Testing Strategy

Tests are scoped to non-experiment utility modules (metrics computation logic, economics calculation), as the model inference itself cannot be unit-tested without the model present.

| Test File | What it Tests |
|---|---|
| `tests/test_metrics.py` | `MetricsResult` construction, TTFT/TPOT computation from mock timestamp lists, RAM monitor logic with mocked `psutil` |
| `tests/test_economics.py` | Break-even calculation correctness, edge cases (N=0, very high N), cache discount factor application |

Coverage target: ≥ 85% on `src/ex05/metrics.py` and `src/ex05/economics.py`.

---

## 10. Lecture Concept Mapping

| Experiment Finding | Lecture Concept |
|---|---|
| Baseline OOM on FP16 load | VRAM / RAM exhaustion; model size vs available memory |
| AirLLM layer-by-layer execution | Virtual memory paging; mmap; OS page fault mechanism |
| High TPOT in AirLLM | Memory-bound Decode phase; SSD I/O as the new bottleneck replacing VRAM bandwidth |
| TTFT vs TPOT separation | Prefill (compute-bound) vs Decode (memory-bound) |
| Q4 lower RAM, lower quality | Quantization trade-off; NF4 (QLoRA paper) |
| Break-even analysis | On-Prem CAPEX+OPEX vs API per-token pricing |

---

## 11. Extension (Original Initiative)

**Proposed extension:** Generate a **Roofline Model** plot for our hardware, overlaying the measured FLOP/byte ratios of the Prefill and Decode phases against the system's theoretical memory bandwidth and compute ceiling. This directly addresses the assignment's "advanced aspiration" and provides a rigorous, visual proof of which resource is the binding constraint at each phase.

If time permits: compare Llama-3.1-8B against `Qwen2.5-7B-Instruct` at Q4 to observe whether architecture differences (GQA vs MHA) affect TPOT under AirLLM.
