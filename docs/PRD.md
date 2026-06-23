# PRD — EX05: Running a Massive LLM Locally with AirLLM and Quantization

**Version:** 1.00  
**Status:** Awaiting Approval  
**Author:** Diana Koroblov  
**Course:** On-Premises LLM Deployment (L08)  
**Date:** 2026-06-23

---

## 1. Purpose and Problem Statement

The goal of this experiment is to demonstrate, with measurable data, the full On-Premises workflow for running a large language model on consumer-grade hardware. The central challenge is that modern LLMs with ~8 billion parameters exceed the available physical RAM when loaded naively in full precision.

This experiment quantifies the failure conditions of direct execution (the **Baseline**), then applies AirLLM and quantization to make the same model runnable, and systematically measures the cost of each trade-off across latency, memory, throughput, and economic viability.

---

## 2. Target Audience

- Course evaluator / Dr. Yoram Segal
- Any engineer or researcher evaluating local LLM deployment on memory-constrained hardware (no dedicated GPU)

---

## 3. Hardware Specification and Constraints

| Component | Specification |
|---|---|
| CPU | **TBD — to be documented by partner (see TODO P0-05)** |
| RAM | **TBD — to be documented by partner (see TODO P0-05)** |
| GPU | **TBD — to be documented by partner (see TODO P0-05)** |
| Storage | **TBD — to be documented by partner** (NVMe SSD strongly recommended; critical for AirLLM layer-loading I/O performance) |
| OS overhead | **TBD** — typically ~2–3 GB reserved at idle; actual free RAM depends on partner's OS and background processes |

**Primary bottleneck:** To be confirmed with partner's hardware. If running on CPU only with no discrete GPU, the system will be **memory-bound** and RAM capacity + bandwidth will dominate the Decode phase. If a discrete GPU with dedicated VRAM is available, VRAM capacity becomes the primary constraint for direct execution.

---

## 4. Model Selection and Justification

### Recommended Model: `meta-llama/Meta-Llama-3.1-8B-Instruct`

**Why this model fits the assignment requirements:**

| Property | Value | Rationale |
|---|---|---|
| Parameters | 8.03 billion | Sufficiently large to exhaust 16 GB RAM in baseline execution |
| Weight size (FP16) | ~16.1 GB | Exceeds available free RAM even before OS overhead and KV cache |
| Weight size (Q8) | ~8.1 GB | Marginal; possible but extremely slow due to swap |
| Weight size (Q4) | ~4.5 GB | Comfortable with AirLLM; valid optimization target |
| Weight size (Q2) | ~2.3 GB | Used only to verify pipeline integrity |
| License | Meta Llama 3.1 Community License | Permits research use; requires attribution |
| Format | SafeTensors (Hugging Face) | Secure, mmap-compatible — required by AirLLM |
| Hugging Face ID | `meta-llama/Meta-Llama-3.1-8B-Instruct` | Gated; requires HF token with accepted license |

**Memory arithmetic (why the Baseline fails):**

```
FP16 weights:      8.03B params × 2 bytes = ~16.1 GB
OS + Python idle:                           ~2.5  GB  (typical; varies by system)
KV cache (512 tok, 32 layers):             ~0.5  GB
─────────────────────────────────────────────────────
Total required:                            ~19.1 GB
Available RAM:                             TBD — partner's hardware (see TODO P0-05)
Deficit (if partner RAM < 19 GB free):     TBD — OOM or heavy swap thrashing expected
```

This confirms that FP16 direct execution will fail or become unusably slow, satisfying the assignment's requirement for a demonstrable Baseline failure.

**Why not a larger model (e.g., 13B or 70B):** The assignment explicitly warns against choosing a model that "has no chance of running even with AirLLM." A 70B model in Q4 still requires ~37 GB, which far exceeds our RAM. The 8B model in Q4 (~4.5 GB for weights) remains feasible under AirLLM's layer-by-layer loading strategy.

**Alternative considered:** `Qwen2.5-7B-Instruct` — also a valid choice and slightly smaller. Rejected in favor of Llama-3.1-8B due to broader community documentation and higher reproducibility.

---

## 5. Goals and Objectives

1. **Demonstrate Baseline failure** — show concretely that FP16 direct execution OOMs or becomes pathologically slow on 16 GB RAM.
2. **Enable execution via AirLLM** — show that layer-by-layer memory mapping allows the same model to complete inference.
3. **Quantify the latency/memory trade-off** across quantization levels (FP16 → Q8 → Q4 → Q2).
4. **Identify the system bottleneck** — classify the system as compute-bound (Prefill) vs. memory-bound (Decode) and support the claim with measured TTFT and TPOT.
5. **Perform economic analysis** — compute the On-Prem vs. API break-even point.
6. **Connect findings to lecture concepts** — Prefill/Decode, virtual memory, paging, KV cache, mmap.

---

## 6. Key Performance Indicators (KPIs)

All KPIs are measured per scenario: Baseline, AirLLM-FP16, AirLLM-Q8, AirLLM-Q4, AirLLM-Q2.

| KPI | Definition | Unit | Target (AirLLM-Q4) |
|---|---|---|---|
| **TTFT** | Time To First Token — latency from prompt submission to first output token | seconds | < 120 s |
| **TPOT** | Time Per Output Token — average inter-token latency after first token | seconds/token | Measured and recorded |
| **Throughput** | Total tokens generated ÷ total elapsed time | tokens/sec | Measured and recorded |
| **Peak RAM** | Maximum RSS/system RAM consumed during inference | GB | < 14 GB |
| **Output quality** | Coherence and relevance of generated text per quantization level | Qualitative (1–5 scale) | Recorded at all levels |

---

## 7. Use Cases

### UC-01 — Baseline Execution (Expected Failure)
**Actor:** Researcher  
**Goal:** Load and run Meta-Llama-3.1-8B-Instruct in FP16 directly without AirLLM.  
**Expected outcome:** OOM error, kernel kill, or >10-minute hang — demonstrating RAM exhaustion.

### UC-02 — AirLLM Execution with Quantization
**Actor:** Researcher  
**Goal:** Run the same model and prompt via AirLLM with Q4 quantization and measure all KPIs.  
**Expected outcome:** Inference completes. Latency is high (minutes per response) due to SSD I/O, but the system does not OOM.

### UC-03 — Quantization Level Sweep
**Actor:** Researcher  
**Goal:** Run inference at Q8, Q4, Q2 levels and compare TTFT, TPOT, RAM, and output quality.  
**Expected outcome:** A clear trade-off table showing lower quantization = less RAM and lower quality.

### UC-04 — Economic Analysis
**Actor:** Researcher  
**Goal:** Compute break-even point between On-Prem and OpenAI/Claude API for a given monthly query volume.  
**Expected outcome:** A break-even graph with explicit assumptions.

---

## 8. Functional Requirements

| ID | Requirement |
|---|---|
| FR-01 | The experiment must run on the documented hardware without modifications. |
| FR-02 | A single standardized prompt must be used across all scenarios for fair comparison. |
| FR-03 | TTFT, TPOT, peak RAM, and throughput must be measured automatically by a metrics module. |
| FR-04 | All results must be persisted to `results/` as JSON or CSV for reproducibility. |
| FR-05 | Comparison graphs must be generated programmatically (Matplotlib). |
| FR-06 | Economic analysis must include explicit CAPEX, OPEX, and API pricing assumptions in a config file. |
| FR-07 | The HF token must never appear in source code; it must be loaded from `.env`. |
| FR-08 | AirLLM `layer_shards_saving_path` must point to a non-OS drive partition if available, or an explicitly defined path to avoid flooding `C:\`. |

---

## 9. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-01 | Environment managed exclusively with `uv`; no direct `pip` or `python -m venv` calls. |
| NFR-02 | All source files ≤ 150 lines. |
| NFR-03 | Zero Ruff lint violations. |
| NFR-04 | No secrets committed to the repository. |
| NFR-05 | Test coverage ≥ 85% on non-experiment utility modules (metrics, economics). |
| NFR-06 | Python version ≤ 3.11 (AirLLM/transformers compatibility). |

---

## 10. Assumptions

- The NVMe SSD provides sufficient sequential read bandwidth for AirLLM's layer-streaming I/O.
- A valid Hugging Face account with accepted Meta Llama 3.1 license is available.
- RAM is cleared to ≤ 4 GB occupied before each experiment run (close browser tabs, background apps).
- Electricity cost is assumed at 0.60 ILS/kWh (Israeli residential rate, 2025).
- Hardware amortization period: 3 years.
- API pricing reference: OpenAI `gpt-4o-mini` at $0.15/1M input tokens, $0.60/1M output tokens (as of 2025).

---

## 11. Out of Scope

- Fine-tuning or LoRA training (inference only).
- Multi-GPU or cloud GPU experiments (optional extension only).
- GUI or interactive chat interface.
- Production deployment or serving.

---

## 12. Dependencies

| Dependency | Purpose |
|---|---|
| `airllm` | Core layer-by-layer inference engine |
| `transformers` | Model loading, tokenization |
| `torch` | Tensor operations (CPU) |
| `bitsandbytes` | Quantization backend |
| `psutil` | RAM monitoring |
| `matplotlib` | Graph generation |
| `python-dotenv` | `.env` secret loading |
| `uv` | Package and environment management |

---

## 13. Milestones

| Milestone | Description | Estimated Wall-Clock Time |
|---|---|---|
| M1 | Environment setup + model download complete | 1.5–3 h (mostly I/O wait) |
| M2 | Baseline documented (failure captured) | 0.5–1 h |
| M3 | AirLLM + Q4 run complete with KPIs logged | 2–4 h |
| M4 | Quantization sweep (Q8, Q4, Q2) complete | 3–5 h |
| M5 | Economic analysis + break-even graph generated | 1–1.5 h |
| M6 | README technical report finalized | 1–2 h |

---

## 14. Acceptance Criteria

- [ ] Baseline failure is documented with logs, RAM screenshots, or error output.
- [ ] AirLLM-Q4 produces a coherent response to the standardized prompt.
- [ ] TTFT, TPOT, throughput, and peak RAM are recorded for ≥ 3 quantization levels.
- [ ] At least one comparison graph is embedded in the README.
- [ ] Break-even graph is present with all assumptions stated.
- [ ] Repository structure matches the recommended layout.
- [ ] All code passes `uv run ruff check .` with zero violations.
- [ ] No HF token or secrets appear in any committed file.
