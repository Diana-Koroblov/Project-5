# EX05 — Running a Large LLM Locally with AirLLM and Quantization

> **Assignment:** EX05 | **Course:** [Course Name]
> **Authors:** [Your Name] & [Partner Name]

---

## Table of Contents

1. [Experiment Description](#1-experiment-description)
2. [Hardware Specifications](#2-hardware-specifications)
3. [Setup & Installation](#3-setup--installation)
4. [Execution Instructions](#4-execution-instructions)
5. [Results Summary](#5-results-summary)
6. [Figures](#6-figures)
7. [Economics Analysis](#7-economics-analysis)
8. [Discussion](#8-discussion)

---

## 1. Experiment Description

### Objective

This experiment explores whether a large language model (LLM) too large for a
laptop's RAM can be run locally using AirLLM's layer-by-layer streaming technique,
and how quantization affects the resulting latency, memory usage, throughput, and
estimated power consumption.

### Methodology

The experiment is divided into two stages:

**Stage 1 — Baseline (Direct FP16 Loading)**
We first attempt to load `meta-llama/Meta-Llama-3.1-8B-Instruct` using the
standard `transformers` pipeline with `torch_dtype=torch.float16` and
`device_map="cpu"`. At FP16 precision the model weights alone occupy ~16.1 GB,
which exceeds the available system RAM. We expect this stage to either crash with
an OOM error or be killed by a 600-second SIGALRM timeout. The resulting
`MetricsResult` (including error message and peak RAM at point of failure) is
saved to `results/baseline_<ts>.json` and included in comparison graphs as a
data point documenting the failure mode.

**Stage 2 — AirLLM Quantization Sweep**
We run the same model with AirLLM (`airllm.AutoModel`) across three quantization
levels in order of decreasing precision: FP16 (AirLLM mmap mode, expected to be
very slow), Q8 (8-bit), and Q4 (4-bit). For each level we:

1. Load the model with AirLLM, which streams transformer layers from disk one
   at a time — analogous to OS virtual memory paging but for model weights.
2. Measure **TTFT** (Time To First Token) via a dedicated single-token pass
   (`max_new_tokens=1`), isolating the Prefill compute phase.
3. Measure full generation throughput via a second pass (`max_new_tokens=200`).
4. Record **peak RAM** (RSS) using a background thread polling `psutil` at 0.5s
   intervals.
5. Compute **estimated power consumption** as
   `(total_runtime_seconds / 3600) × cpu_tdp_watts` where `cpu_tdp_watts` is
   read from `config/experiment_config.json`.
6. Save all KPIs to `results/airllm_<level>_<ts>.json`.

### Prompt

All stages use the identical prompt to ensure comparability:

> *"Explain the difference between supervised and unsupervised learning in three
> paragraphs."*

Token count: ~17 input tokens. Output capped at 200 tokens for throughput
measurement (10 tokens for the smoke test).

### Measurement Tools

| Metric | Implementation |
|---|---|
| TTFT | `InferenceTimer` context manager — records `t_first_token − t_start` |
| TPOT | Mean inter-token gap from `InferenceTimer.token_timestamps` |
| Throughput | `generated_tokens / total_runtime_seconds` |
| Peak RAM | `RamMonitor` (daemon thread, 0.5s polling via `psutil.Process.memory_info().rss`) |
| Power (est.) | `(runtime_sec / 3600) × cpu_tdp_watts` (TDP from config) |

### Quantization Rationale

| Level | Weight size (8B model) | Expectation |
|---|---|---|
| FP16 baseline | ~16.1 GB | OOM / timeout — documents the problem |
| AirLLM FP16 | ~16.1 GB (streamed) | Feasible but extremely slow (disk I/O) |
| Q8 | ~8.5 GB | Fits in RAM; moderate slowdown vs FP16 |
| Q4 | ~4.5 GB | Well within RAM; good throughput |

Q4 is our primary production candidate. Q8 shows the precision–speed trade-off.
FP16 (both direct and AirLLM) anchor the comparison as high-precision baselines.

---

## 2. Hardware Specifications



| Component | Specification |
|---|---|
| CPU | AMD Ryzen 9700X |
| RAM | 32GB DDR5 5200MHz |
| Storage | 512GB NVME PCIe 4.0 |
| OS | Windows 11 Pro |
| CPU TDP | 65W |
| Hardware cost | TBD ILS (update `config/economics_config.json` → `hardware_cost_ils`) |

---

## 3. Setup & Installation

### Prerequisites

- Python 3.11 (exactly — not 3.12+)
- [`uv`](https://docs.astral.sh/uv/) installed globally
- A Hugging Face account with access to
  `meta-llama/Meta-Llama-3.1-8B-Instruct`

### Install

```bash
uv sync
```

### Configure environment

Copy `.env-example` to `.env` and insert your Hugging Face token:

```bash
cp .env-example .env
# edit .env and replace hf_YOUR_TOKEN_HERE with your real token
```

> **Never commit `.env` to Git.** It is already in `.gitignore`.

### Update hardware config (partner)

Edit `config/experiment_config.json` → set `cpu_tdp_watts` to your CPU's TDP.  
Edit `config/economics_config.json` → set `hardware_cost_ils` to the actual
cost of your hardware in ILS.

---

## 4. Execution Instructions

Run each script with `uv run python`. Scripts must be run from the **repository
root**.

### Step 0 — Verify environment (smoke test)

```bash
uv run python -c "from ex05.config import ExperimentConfig; print(ExperimentConfig.load())"
```

Expected: dataclass printed with no errors.

### Step 1 — Disk space check

Ensure ≥ 30 GB free before downloading the model:

```bash
df -h .          # Linux / macOS
# or check via File Explorer on Windows
```

### Step 2 — Stage 1: Baseline experiment

```bash
uv run python experiments/run_baseline.py
```

Expected output: OOM error or 10-minute timeout. A `results/baseline_*.json`
file is saved regardless of outcome. **Monitor RAM** — close all other
applications first.

### Step 3 — Stage 2: AirLLM quantization sweep

```bash
uv run python experiments/run_airllm.py
```

This iterates through all quantization levels defined in
`config/experiment_config.json` (`fp16`, `4bit`, `8bit` by default). Each level
takes 30–90 minutes depending on hardware and storage speed. Results are saved
to `results/airllm_<level>_<ts>.json`.

**Tips:**
- Close RAM-heavy applications before running.
- Keep ≥ 4 GB free RAM for Q4, ≥ 9 GB for Q8.
- Partial runs are fine — each level saves independently.

### Step 4 — Generate comparison graphs

```bash
uv run python experiments/generate_graphs.py
```

Reads all `results/*.json` files (excluding economics), produces three PNG
charts in `figures/`:

- `ttft_comparison.png` — Time to First Token per scenario
- `ram_comparison.png` — Peak RAM per scenario
- `throughput_comparison.png` — Throughput (tokens/sec) per scenario

### Step 5 — Economics analysis

```bash
uv run python experiments/run_economics.py
```

Produces `figures/break_even.png` (four cost curves + break-even annotations)
and `results/economics_<ts>.json`.

### Step 6 — Run tests

```bash
uv run pytest
```

Expected: ≥ 85% coverage on utility modules, all tests pass. Tests are fully
offline — no model download required.

---

## 5. Results Summary

| Scenario | TTFT (s) | TPOT (s/tok) | Throughput (tok/s) | Peak RAM (GB) | Power (Wh) | Error |
|---|---|---|---|---|---|---|
| Baseline FP16 | 6.24 | n/a ¹ | ~1.6 ² | **17.01** | 0.11 | None — but near-OOM |
| AirLLM FP16 | 14.96 | 15.09 | 0.055 | **2.54** | 1.63 | None |
| AirLLM Q8 | — | — | — | — | — | `AssertionError: Torch not compiled with CUDA enabled` |
| AirLLM Q4 | — | — | — | — | — | `AssertionError: Torch not compiled with CUDA enabled` |

> ¹ The baseline timer calls `record_token()` once after `model.generate()` finishes, not per token. TPOT is not resolvable from this single timestamp.
> ² Estimated from 10 output tokens ÷ 6.24 s total runtime.

---

### Stage 1 — Baseline Results

![Baseline Failure Mode](figures/baseline_failure.png)

*Figure: Peak RSS memory for the Baseline (FP16 direct-load) vs AirLLM FP16. The red dashed line marks the
~18 GB of RAM that was free at run time. The baseline consumed 17.01 GB — 94 % of available RAM — leaving
less than 1 GB for the OS and background processes.*

**Run:** `experiments/run_baseline.py` | **Log:** `logs/baseline_run_20260624_174405.log`

**Console output excerpt:**

```
============================================================
BASELINE EXPERIMENT — Direct FP16 Loading
============================================================
Model  : meta-llama/Meta-Llama-3.1-8B-Instruct
WARNING: Expected to OOM or time out. Monitor system RAM.
============================================================
Loading checkpoint shards: 100%|██████████| 4/4 [00:07<00:00,  1.97s/it]
Setting `pad_token_id` to `eos_token_id`:None for open-end generation.

--- Results ---
Peak RAM    : 17.01 GB
Runtime     : 6.2s
Power (est) : 0.1127 Wh
Error       : None (model ran)
```

**Key observations:**

1. **Expected OOM did not occur.** This machine had ~18 GB free at run time; the 16 GB FP16 model fit (barely). On a system with ≤ 16 GB free — the common case for a laptop — this run would crash with a `torch.cuda.OutOfMemoryError` or the kernel OOM-killer terminating the process.

2. **Near-OOM behaviour is still a production failure.** 17.01 GB consumed out of ~18 GB available leaves < 1 GB for the OS scheduler, file-system buffers, and all other processes. The system becomes unresponsive during inference.

3. **Prefill vs Decode bottleneck.** In the standard transformer two-phase model:
   - *Prefill* processes all input tokens in a single parallel pass (compute-bound GEMM). This completed quickly even on CPU.
   - *Decode* generates each output token autoregressively, requiring one full forward pass per token (memory-bandwidth-bound GEMV). Each pass must stream the full 16 GB of weights through the CPU's memory controller. At DDR5 5200 MHz (≈ 50 GB/s effective), the theoretical floor is 16 GB ÷ 50 GB/s ≈ 0.32 s/token; the measured ~0.62 s/token matches this within a 2× overhead factor.

4. **AirLLM's RAM advantage.** By streaming one layer shard at a time from NVMe storage (never materialising the full model in RAM simultaneously), AirLLM reduced peak RSS to **2.54 GB** — a **6.7× reduction** — at the cost of higher latency (NVMe at ~3–7 GB/s is ~10–15× slower than DDR5 for sequential reads, which is reflected in the 15 s/token TPOT).

---

### Stage 2 — AirLLM + Quantization Results

AirLLM (`airllm.AutoModel`) was run across all three quantization levels defined
in `config/experiment_config.json`. Only **FP16** completed; both quantized levels
were blocked by a hard hardware constraint (see note below and RISK-05).

**Run:** `experiments/run_airllm.py` | **Log:** `logs/airllm_sweep_20260624_175040.log`

| Scenario | TTFT (s) | TPOT (s/tok) | Throughput (tok/s) | Peak RAM (GB) | Power (Wh) | Quality (1–5) | Status |
|---|---|---|---|---|---|---|---|
| Baseline FP16 (direct) | 6.24 | n/a ¹ | ~1.6 ² | 17.01 | 0.11 | 4 ³ | ✅ Ran (near-OOM) |
| AirLLM FP16 | 14.96 | 15.09 | 0.055 | **2.54** | 1.63 | 4 ³ | ✅ Ran |
| AirLLM Q8 (8-bit) | — | — | — | — | — | — | ❌ CUDA-blocked ⁴ |
| AirLLM Q4 (4-bit) | — | — | — | — | — | — | ❌ CUDA-blocked ⁴ |

> ¹ See Stage 1 footnote — single timestamp, TPOT not resolvable.
> ² Estimated from 10 output tokens ÷ 6.24 s total runtime.
> ³ Provisional. Both runs were capped at a few tokens to keep the sweep under a
> few minutes; the generated prefixes ("*Supervised learning is …*") are coherent
> and on-topic, but too short for a full 1–5 quality judgement. Formal scoring is
> tracked as A3-07.
> ⁴ `AssertionError: Torch not compiled with CUDA enabled`. AirLLM's 4-bit and
> 8-bit paths use **bitsandbytes**, which requires a CUDA-enabled PyTorch build.
> This machine has an AMD Radeon GPU and no NVIDIA CUDA device, so quantized
> inference cannot run here. This is a documented, reproducible finding, not a
> code defect (RISK-05).

#### KPI Comparisons

![TTFT Comparison](figures/ttft_comparison.png)

![Peak RAM Comparison](figures/ram_comparison.png)

![Throughput Comparison](figures/throughput_comparison.png)

*Each chart includes all four scenarios. The Q4/Q8 bars sit at zero because those
runs aborted at load time before producing any tokens.*

#### Analysis — Decode-phase memory-bound behaviour

The central lecture concept on display here is the **Decode phase being
memory-bound, not compute-bound**. Autoregressive decode generates one token per
forward pass, and each pass must read **every** weight in the model exactly once.
Throughput is therefore governed by *how fast weights can reach the compute
units*, not by FLOPs:

- **Baseline (direct FP16):** weights live in DDR5 RAM (~50 GB/s effective). The
  measured ~0.62 s/token sits within ~2× of the 16 GB ÷ 50 GB/s ≈ 0.32 s/token
  bandwidth floor — classic memory-bound decode.
- **AirLLM FP16:** weights live on the NVMe SSD and are streamed layer-by-layer
  per token. TPOT jumps to **15.09 s/token** — roughly **24× slower** than the
  in-RAM baseline — because the bandwidth bottleneck moved one level down the
  memory hierarchy (NVMe ≈ 3–7 GB/s vs DDR5 ≈ 50 GB/s). The Prefill phase (TTFT
  14.96 s) and the Decode phase (TPOT 15.09 s) are nearly identical, confirming
  that for AirLLM **disk I/O dominates both phases** — the per-layer streaming
  cost swamps the compute difference between parallel Prefill and sequential
  Decode.
- **Quantization's intended effect (unrealised here):** Q4/Q8 would shrink each
  layer shard ~4×/~2×, cutting the bytes streamed per token and therefore the
  memory-bound TPOT roughly proportionally. It would **not** materially improve
  TTFT, since Prefill is comparatively compute-bound. On CUDA hardware we would
  expect Q4 to give the best throughput/latency trade-off; on this box the
  trade-off is moot because bitsandbytes cannot load.

The 6.7× RAM reduction (17.01 → 2.54 GB) is AirLLM's headline win: it converts an
*impossible-to-fit* model into a *runnable-but-slow* one by trading the fast-but-
scarce RAM tier for the slow-but-abundant disk tier — the same space/time trade-off
as OS virtual-memory paging, applied at the granularity of transformer layers.

---

## 6. Figures

All figures regenerate via `uv run python experiments/generate_graphs.py`
(KPI comparisons) and `uv run python experiments/run_economics.py` (break-even).

### TTFT Comparison

![TTFT Comparison](figures/ttft_comparison.png)

### Peak RAM Comparison

![Peak RAM Comparison](figures/ram_comparison.png)

### Throughput Comparison

![Throughput Comparison](figures/throughput_comparison.png)

---

## 7. Economics Analysis

We compare the per-request cost of running the model **On-Premises** (this
machine, amortized) against a third-party **API** (`gpt-4o-mini` pricing) and a
rented **Cloud GPU**. The On-Prem per-request cost falls as monthly volume rises
(fixed costs spread over more requests), whereas the API and Cloud GPU costs are
flat per request. The **break-even point N\*** is where the On-Prem curve crosses
the API curve — below it the API is cheaper, above it On-Prem wins.

![Break-Even Analysis: On-Prem vs Cloud vs API](figures/break_even.png)

*Figure: Cost per request (ILS, log-x) vs monthly request volume. The green
On-Prem curve decreases with volume; the API (no-cache / cached) and Cloud GPU
lines are flat. Vertical lines mark the break-even volumes N\*.*

### Results

| Metric | Value |
|---|---|
| On-Prem monthly fixed cost | **₪247.52 / month** |
| API cost/request (no cache) | ₪0.000472 |
| API cost/request (cached) | ₪0.000447 |
| Cloud GPU cost/request | ₪0.015417 |
| Break-even N\* (no cache) | **524,694 requests/month** |
| Break-even N\* (cached) | **554,024 requests/month** |

### Assumptions

| Category | Parameter | Value |
|---|---|---|
| CAPEX | Hardware cost | ₪7,000 |
| CAPEX | Amortization period | 3 years (→ ₪194.44/mo) |
| OPEX | Electricity rate | ₪0.60 / kWh |
| OPEX | Avg. power draw | 65 W (→ ₪28.08/mo at 24×30 h) |
| OPEX | Annual maintenance | ₪300 (→ ₪25.00/mo) |
| API | Input / output price | $0.15 / $0.60 per 1M tokens (`gpt-4o-mini`) |
| API | Avg. tokens per request | 50 in / 200 out |
| API | Cache discount factor | 0.1 (applied to input tokens) |
| Cloud GPU | Hourly rate / runtime | $0.50/h × 0.5 min per request |
| FX | USD → ILS | 3.7 |

### Recommendation

For this workload, the third-party API is **dramatically cheaper** until volume
becomes extreme. On-Prem fixed costs (~₪247.52/month) only pay off beyond
**~525,000 requests/month** (~17,500/day) — because `gpt-4o-mini` costs only
~₪0.0005 per request. Prompt caching pushes the break-even slightly higher
(~554k). The Cloud GPU option is the most expensive per request here (~₪0.0154)
and never competitive for this low-token workload.

**When On-Prem is preferred:** very high sustained volume (well above the
break-even), or when **data privacy, offline operation, or no-per-call-billing**
are hard requirements — none of which this cost model captures but which are
often the real reason to self-host. **When the API is preferred:** low-to-moderate
volume, spiky traffic, or when avoiding capital expenditure and operational
overhead matters more than marginal per-request cost. (All figures regenerate via
`uv run python experiments/run_economics.py`.)

---

## 8. Discussion

> **[PLACEHOLDER — fill in after reviewing results (TODO R5-02 full analysis)]**

Answer the following questions in this section:

1. **Baseline bottleneck:** What caused the baseline failure? Was it a memory
   allocation error, a swap-induced slowdown, or a timeout? What does this tell
   us about the relationship between model size and available RAM?

2. **AirLLM and virtual memory paging:** How does AirLLM's layer-by-layer
   streaming relate to OS virtual memory paging? What are the analogies and
   differences?

3. **Effect of Q4/Q8 on memory and text quality:** How did quantization affect
   peak RAM? Did the generated text show quality degradation at Q4 vs Q8?
   Provide examples from the output.

4. **TTFT/TPOT and Prefill vs. Decode:** TTFT reflects the Prefill phase
   (compute-bound GEMM). TPOT reflects the Decode phase (memory/disk-bound
   GEMV). What did the measurements show? Did AirLLM's disk I/O dominate
   TPOT regardless of quantization level?

5. **Throughput / Latency trade-off:** Quantization reduces model size and
   thus disk I/O per layer, improving throughput. But does it improve latency
   (TTFT)? Explain the trade-off you observed.
