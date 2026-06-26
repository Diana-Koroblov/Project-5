# PLAN — EX05: Architecture & Technical Design

**Version:** 1.01  
**Status:** Approved  
**Authors:** Itay Malich & Diana Koroblov  
**Date:** 2026-06-23 (v1.00) | 2026-06-25 (v1.01 — updated to reflect completed three-stage implementation)  
**Reference:** PRD v1.01

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.00 | 2026-06-23 | Initial architecture design |
| 1.01 | 2026-06-25 | Added Stage 3 (Ollama/GGUF); updated repo structure to match actual implementation; updated config, testing strategy, and architectural decisions |

---

## 1. Overview

This document describes the technical architecture, environment design, experiment pipeline, and measurement strategy for EX05. It translates the requirements in `PRD.md` into a concrete, implementable design.

The experiment runs in three stages:

1. **Stage 1 — Baseline:** Direct FP16 loading via Hugging Face Transformers. Documents near-OOM conditions.
2. **Stage 2 — AirLLM:** Layer-by-layer NVMe streaming. Reduces peak RAM 6.7×; CUDA-only quantization paths are blocked on this AMD box.
3. **Stage 3 — Ollama/GGUF:** CPU-native quantization via llama.cpp. Delivers the real quantization gradient (FP16→Q8→Q4→Q2) that Stage 2 cannot.

---

## 2. Repository Structure

```
Project-5/
├── README.md                       # Technical report (the final deliverable)
├── LICENSE                         # MIT © 2026 Itay Malich & Diana Koroblov
├── pyproject.toml                  # Single source of truth for deps + uv config
├── uv.lock                         # Locked dependency graph
├── .env-example                    # Placeholder for secrets (HF_TOKEN)
├── .gitignore
│
├── docs/
│   ├── PRD.md                      # This project's PRD
│   ├── PLAN.md                     # This file
│   ├── TODO.md                     # Task tracking
│   ├── PROMPT_LOG.md               # AI-assisted development prompt log
│   ├── PRD_airllm.md               # Per-mechanism PRD: AirLLM runner
│   ├── PRD_ollama.md               # Per-mechanism PRD: Ollama/GGUF runner
│   └── PRD_economics.md            # Per-mechanism PRD: economics model
│
├── src/
│   └── ex05/
│       ├── __init__.py             # Package version
│       ├── config.py               # Typed config dataclasses; .env loading
│       ├── metrics.py              # TTFT, TPOT, throughput, RAM measurement
│       ├── baseline.py             # Baseline (direct HF) inference runner
│       ├── airllm_runner.py        # AirLLM layer-streaming inference runner
│       ├── ollama_runner.py        # Ollama CPU-quantization runner (GGUF)
│       └── economics.py            # Break-even cost calculation logic
│
├── experiments/
│   ├── run_baseline.py             # Entry point: Stage 1 baseline
│   ├── run_airllm.py               # Entry point: Stage 2 AirLLM sweep
│   ├── run_ollama.py               # Entry point: Stage 3 Ollama sweep
│   ├── run_economics.py            # Entry point: economic analysis + graph
│   ├── generate_break_even_cumulative.py  # Cumulative total-cost break-even
│   ├── generate_graphs.py          # KPI comparison charts (Stages 1+2)
│   ├── generate_ollama_graphs.py   # Quantization gradient chart (Stage 3)
│   └── generate_roofline.py        # Roofline Model plot (extension)
│
├── results/                        # Raw JSON output (committed — required deliverable)
│   ├── baseline_*.json
│   ├── airllm_*.json
│   ├── ollama_*.json
│   └── economics_*.json
│
├── figures/                        # Generated graphs (PNG, committed)
│   ├── baseline_failure.png
│   ├── ram_comparison.png
│   ├── ttft_comparison.png
│   ├── throughput_comparison.png
│   ├── ollama_quant_comparison.png
│   ├── break_even.png
│   ├── break_even_cumulative.png
│   └── roofline.png
│
├── notebooks/
│   └── results_analysis.ipynb      # Central research artifact
│
├── tests/
│   ├── conftest.py                 # Shared fixtures
│   ├── test_config.py
│   ├── test_metrics.py
│   ├── test_economics.py
│   ├── test_baseline.py
│   ├── test_airllm_runner.py
│   └── test_ollama_runner.py
│
├── config/
│   ├── experiment_config.json      # Prompt, tokens, quant levels, Ollama config
│   └── economics_config.json       # Hardware costs, electricity, API pricing
│
├── logs/                           # Run logs (committed; referenced in README)
└── model_shards/                   # AirLLM layer shards (gitignored; large binary)
```

All source files are capped at 150 non-blank, non-comment lines per the software guidelines.

---

## 3. Environment Setup (uv)

### 3.1 Rationale for uv

`uv` is the mandatory package manager per the software guidelines. It provides deterministic installs via `uv.lock`, fast resolution, and a clean `pyproject.toml`-based workflow. No `pip`, `venv`, or `virtualenv` calls appear anywhere in the project.

### 3.2 Python Version

**Python 3.11** — the highest version confirmed compatible with `airllm`, `bitsandbytes`, and `transformers` as of 2025. Python 3.12+ introduces breaking changes in some dependencies.

### 3.3 Setup Commands

```bash
uv sync                                          # install all dependencies
uv run python experiments/run_baseline.py        # Stage 1
uv run python experiments/run_airllm.py          # Stage 2
uv run python experiments/run_ollama.py          # Stage 3 (requires: ollama serve)
uv run python experiments/run_economics.py       # economics
uv run pytest tests/                             # run tests
uv run ruff check .                              # lint
```

### 3.4 Secrets Management

The Hugging Face token is loaded exclusively from `.env` via `python-dotenv`. The `.env` file is in `.gitignore`. An `.env-example` with a dummy placeholder is committed.

---

## 4. Configuration Architecture

All runtime-tunable values live in `config/`. No magic numbers appear in source code.

### `config/experiment_config.json`
```json
{
  "version": "1.00",
  "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "prompt": "Explain the difference between supervised and unsupervised learning in three paragraphs.",
  "max_new_tokens": 10,
  "max_new_tokens_experiment": 200,
  "quantization_levels": ["fp16", "4bit", "8bit"],
  "layer_shards_path": "./model_shards",
  "seed": 42,
  "cpu_tdp_watts": 65,
  "ollama": {
    "host": "http://localhost:11434",
    "num_predict": 200,
    "temperature": 0,
    "force_cpu": true,
    "quant_levels": {
      "fp16": "llama3.1:8b-instruct-fp16",
      "q8":   "llama3.1:8b-instruct-q8_0",
      "q4":   "llama3.1:8b-instruct-q4_K_M",
      "q2":   "llama3.1:8b-instruct-q2_K"
    }
  }
}
```

### `config/economics_config.json`
```json
{
  "version": "1.00",
  "hardware_cost_ils": 7000,
  "amortization_years": 3,
  "electricity_kwh_ils": 0.60,
  "avg_power_watts": 65,
  "maintenance_cost_annual_ils": 300,
  "api_input_cost_per_1m_tokens": 0.15,
  "api_output_cost_per_1m_tokens": 0.60,
  "avg_input_tokens_per_request": 50,
  "avg_output_tokens_per_request": 200,
  "cache_discount_factor": 0.1,
  "cloud_gpu_hourly_usd": 0.50,
  "usd_to_ils_rate": 3.7
}
```

---

## 5. Experiment Pipeline Design

### 5.1 Standardized Prompt

A single fixed prompt is used across all three stages for fair comparison:

> *"Explain the difference between supervised and unsupervised learning in three paragraphs."*

Short enough (≈12 tokens) to keep Prefill fast; requests 200 output tokens to stress Decode.

### 5.2 Stage 1 — Baseline (Direct HF Execution)

**Goal:** Demonstrate that loading Meta-Llama-3.1-8B-Instruct in FP16 saturates available RAM.

1. Load `AutoTokenizer` and `AutoModelForCausalLM` in FP16.
2. Start RAM monitoring (background thread via `psutil`).
3. Call `model.generate()` with `max_new_tokens=10`.
4. Log peak RAM, TTFT, runtime, and error state to `results/baseline_*.json`.

**Outcome:** 17.01 GB peak RSS — 94% of available RAM. System unresponsive during inference. Documents the near-failure condition.

### 5.3 Stage 2 — AirLLM + Quantization Sweep

**Goal:** Demonstrate layer-streaming RAM reduction and document the CUDA quantization constraint.

For each level in `["fp16", "4bit", "8bit", "2bit"]`:

1. Instantiate `airllm.AutoModel` with the configured `layer_shards_saving_path` and `compression`.
2. Run generation; record TTFT, per-token timestamps (→ TPOT), peak RAM, and output.
3. Persist `MetricsResult` to `results/airllm_<level>_*.json`.

**Outcomes:**
- FP16: ran; 6.7× RAM reduction (17.01 → 2.54 GB); 15 s/token decode (NVMe bound).
- Q4/Q8: `AssertionError: Torch not compiled with CUDA enabled` — bitsandbytes CUDA dependency (RISK-05).
- Q2: ran but silently fell back to FP16 shards (byte-identical on disk); not genuine 2-bit.

### 5.4 Stage 3 — Ollama CPU Quantization Sweep

**Goal:** Deliver the real quantization gradient that Stage 2 cannot, using GGUF/llama.cpp.

For each level in `["fp16", "q8", "q4", "q2"]`:

1. Send streaming HTTP request to `http://localhost:11434/api/generate` with `num_gpu=0`.
2. Collect `(perf_counter, chunk)` events; derive TTFT, TPOT, throughput, token count.
3. Query `/api/ps` for peak RAM (`size − size_vram`).
4. Persist `MetricsResult` to `results/ollama_<level>_*.json`.

**Peak RAM is read from `/api/ps`, not process RSS.** GGUF files are mmap'd by the Ollama server process; the RSS is misleadingly low. `/api/ps` reports the actual model footprint in memory.

**Outcomes:** Monotonic gradient — RAM ↓4.4× (FP16→Q2), throughput ↑4.5×; quality loss only at Q2 (score 4/5 vs 5/5 for FP16–Q4).

---

## 6. Metrics Architecture

### `src/ex05/metrics.py`

- `RamMonitor` — `threading.Thread` that samples `psutil.Process().memory_info().rss` every 500 ms; records peak.
- `record_token(ts)` — appends a `perf_counter()` timestamp to a list for per-token latency.
- `MetricsResult` — frozen `dataclass` holding all KPIs, with `.to_dict()` for JSON serialization; computed via `compute_metrics()`.

### `src/ex05/ollama_runner.py`

- `_stream_generate()` — `urllib` streaming over the Ollama HTTP API; yields `(perf_counter, parsed_json)` events.
- `_model_ram_gb()` — queries `/api/ps`; returns `(size − size_vram) / 1024**3`.
- `_assemble()` — pure function; builds `MetricsResult` from streamed event list.
- CPU forcing: `options.num_gpu=0`; determinism: `temperature=0`.

### Results Persistence

Every experiment run serializes its `MetricsResult` to `results/<scenario>_<timestamp>.json`. The analysis notebook and graph scripts read from these files, ensuring reproducibility on a fresh clone.

---

## 7. Economic Analysis Architecture

### `src/ex05/economics.py`

**API cost per request:**
```
cost_api = (input_tokens × price_input + output_tokens × price_output) / 1_000_000 × USD→ILS
```
With `cache_discount_factor` applied to cached input tokens.

**On-Prem monthly fixed cost:**
```
fixed = hardware_cost / (amort_years × 12)          # CAPEX
      + (avg_power_W / 1000) × 720h × elec_rate     # OPEX electricity
      + maintenance_annual / 12                      # OPEX maintenance
```

**Break-even (closed form):** `N* = fixed_monthly / cost_api`

**Cloud GPU cost per request:** `(runtime_min / 60) × hourly_rate × USD→ILS`

`experiments/run_economics.py` generates `figures/break_even.png` with three curves (API, On-Prem, Cloud GPU) and a vertical dashed line at N*.

---

## 8. Architectural Decisions and Trade-offs

| Decision | Choice | Rationale |
|---|---|---|
| Model format | SafeTensors (HF) + GGUF (Ollama) | SafeTensors required by AirLLM mmap; GGUF required for CPU-native quantization |
| Stage 2 quantization backend | `bitsandbytes` via AirLLM | Native integration; documented as CUDA-only (negative result) |
| Stage 3 quantization backend | Ollama/llama.cpp GGUF | CPU-native; no CUDA required; one-command model management |
| Ollama HTTP transport | Raw `urllib` (no `ollama` Python package) | No extra dependency; full control over streaming timestamps for accurate TTFT |
| Peak RAM measurement (Ollama) | `/api/ps` `size` field | GGUF is mmap'd — RSS is misleadingly low, notably on Windows |
| CPU forcing (Ollama) | `options.num_gpu=0` | Verified by `size_vram=0` in `/api/ps` response |
| RAM monitor granularity | 500 ms polling | Fine enough to catch peaks; negligible overhead |
| Token timing | `perf_counter()` callback | Sub-millisecond precision; no CUDA event dependency |
| Configuration format | JSON | No extra dependency; human-readable; strict schema |
| Experiment entry points | Separate script per stage | Allows running phases independently; avoids reloading model between experiments |

### 8.1 API Gatekeeper — Not Applicable (deliberate)

The software guidelines mandate a central API Gatekeeper (rate limiting, queueing, retries, monitoring) for **all external API calls**. This project has **no external, network-billed, rate-limited API in its runtime path**, so a gatekeeper would add infrastructure with nothing to govern: (1) the **economics analysis is fully offline** — OpenAI/Cloud-GPU pricing is *computed* from `config/economics_config.json`, never called, so there are no tokens to meter or rate-limit; (2) **Ollama inference targets `http://localhost:11434`** — a local server process the user runs, with no quota, no per-call billing, and no provider rate limit to respect (back-pressure against your own CPU is meaningless); and (3) the only genuinely remote calls are **Hugging Face model downloads**, which are one-time setup performed by `transformers`/`huggingface_hub`, not part of the measured experiment, and already retried internally by that library. The concerns a gatekeeper exists to solve — quota exhaustion, throttling, cost runaway, queue back-pressure — do not arise here. Were a paid remote inference API ever added (e.g. calling OpenAI for a real latency/cost comparison rather than computing it), the correct design would be a single `ex05` gatekeeper module wrapping that client, reading `requests_per_minute` / `max_retries` / `retry_delay` from a versioned `config/rate_limits.json`, with a FIFO queue and exponential-backoff retries — and **all** call sites routed through it. That extension point is intentionally left unbuilt because exercising it would require fabricating an API dependency the experiment does not need.

---

## 9. Testing Strategy

Every `src/ex05` module has a matching test file. The model-loading runners (baseline, AirLLM) and the Ollama HTTP path are heavy integration code, but they are still unit-tested by mocking their external dependencies — HF token, `AutoTokenizer`/`AutoModelForCausalLM`, AirLLM's `AutoModel` (via `sys.modules` injection), and `urllib` — so the entire suite runs fully offline with no model download, no CUDA, and no live Ollama server. The real end-to-end runs are additionally evidenced by `results/*.json` and `logs/`.

| Test File | Coverage |
|---|---|
| `tests/test_config.py` | `config.py` — dataclass construction, JSON loading, OllamaConfig, `get_hf_token()` |
| `tests/test_metrics.py` | `metrics.py` — `MetricsResult` construction, TTFT/TPOT derivation, RamMonitor with mocked psutil |
| `tests/test_economics.py` | `economics.py` — break-even correctness, edge cases (N=0, high N), cache discount, algebraic consistency |
| `tests/test_ollama_runner.py` | `ollama_runner.py` — `_build_payload`, `_model_ram_gb` / `_stream_generate` (mocked HTTP), `_assemble`, and `run_ollama` happy + error paths |
| `tests/test_baseline.py` | `baseline.py` — happy path, generation failure, load failure (all HF deps mocked); win32 timeout no-ops |
| `tests/test_airllm_runner.py` | `airllm_runner.py` — `_resolve_model_source` (both branches), `_make_model`, `run_airllm` happy + CUDA-failure paths |

The only lines excluded from coverage are the POSIX-only `SIGALRM` timeout branch in `baseline.py` (marked `# pragma: no cover` — `signal.SIGALRM` does not exist on the Windows dev box and cannot be exercised there).

**Coverage:** the suite auto-fails below 85% global (`--cov-fail-under=85` in `pyproject.toml`). Actual: **100%** across all seven modules (60 tests).

---

## 10. Lecture Concept Mapping

| Experiment Finding | Lecture Concept |
|---|---|
| Baseline saturates RAM at 17.01 GB | RAM exhaustion; model size vs available memory |
| AirLLM: peak RSS drops to 2.54 GB | Virtual memory paging; mmap; OS page-fault mechanism applied to transformer layers |
| AirLLM TPOT = 15 s/token | Memory-bound Decode; NVMe I/O as bottleneck replacing DDR5 bandwidth |
| TTFT vs TPOT separation | Prefill (compute-bound GEMM) vs Decode (memory-bound GEMV) |
| Ollama Q4: 3× less RAM, 3.5× faster | Quantization arithmetic: fewer bits/weight → fewer bytes/token → faster memory-bound decode |
| Roofline: decode at I=1 FLOP/byte | Ridge point theory; both DDR5 and NVMe roofs confirm memory-bound classification |
| Break-even N* ≈ 525k req/mo | On-Prem CAPEX+OPEX vs API per-token pricing; prompt caching shifts the curve |

---

## 11. Extension — Roofline Model

**Implemented in:** `experiments/generate_roofline.py` → `figures/roofline.png`

The Roofline Model plots attainable GFLOP/s against arithmetic intensity (FLOP/byte) on log-log axes:

- **Compute roof:** 2560 GFLOP/s (8 cores × 5 GHz × 64 FLOP/cycle, AVX-512)
- **DDR5 memory roof:** 50 GB/s → ridge point at 51 FLOP/byte
- **NVMe memory roof:** 7 GB/s → ridge point at 366 FLOP/byte
- **Decode operating point:** I = 1 FLOP/byte (2 FLOP per 2-byte weight, GEMV)

Both ridge points are far to the right of I=1, confirming decode is **memory-bound on every roof**. The Baseline (DDR5) and AirLLM (NVMe) points both sit on their respective memory-bandwidth roofs. This is the rigorous, visual proof of the bottleneck claim made throughout the report.
