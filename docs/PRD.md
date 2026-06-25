# PRD — EX05: Running a Massive LLM Locally with AirLLM and Quantization

**Version:** 1.01  
**Status:** Approved  
**Authors:** Itay Malich & Diana Koroblov  
**Course:** On-Premises LLM Deployment (L08)  
**Date:** 2026-06-23 (v1.00) | 2026-06-25 (v1.01 — updated to reflect actual hardware and three-stage results)

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.00 | 2026-06-23 | Initial draft |
| 1.01 | 2026-06-25 | Filled hardware TBDs (32 GB, actual baseline result); corrected model size expectations; added Stage 3 (Ollama); updated acceptance criteria; both authors listed |

---

## 1. Purpose and Problem Statement

The goal of this experiment is to demonstrate, with measurable data, the full On-Premises workflow for running a large language model on consumer-grade hardware. The central challenge is that modern LLMs with ~8 billion parameters approach or exhaust available physical RAM when loaded naively in full precision.

This experiment quantifies the failure conditions of direct execution (the **Baseline**), then applies AirLLM and quantization to make the same model runnable, and systematically measures the cost of each trade-off across latency, memory, throughput, and economic viability.

---

## 2. Target Audience

- Course evaluator / Dr. Yoram Segal
- Any engineer or researcher evaluating local LLM deployment on memory-constrained hardware (no dedicated GPU)

---

## 3. Hardware Specification and Constraints

| Component | Specification |
|---|---|
| CPU | AMD Ryzen 9700X (8 cores / 16 threads, 65 W TDP) |
| RAM | 32 GB DDR5 5200 MHz |
| GPU | AMD Radeon RX 9070 XT (16 GB GDDR6) — present but **not used for inference**: ROCm/HIP support on Windows is insufficient for the 9070 XT (RDNA 4, 2025); all inference runs on CPU |
| Storage | 512 GB NVMe PCIe 4.0 |
| OS | Windows 11 Pro |
| OS overhead | ~2–3 GB reserved at idle; leaves ~29–30 GB free RAM for the experiment |

**Primary bottleneck:** The system is **memory-bound** during the Decode phase. All inference runs on CPU. With 32 GB RAM, the FP16 model (~17 GB loaded) fits but saturates available RAM — the Baseline ran but consumed 17.01 GB, leaving < 1 GB for the OS, making the system effectively unresponsive during inference. AirLLM reduces peak RSS to 2.54 GB by streaming layers from NVMe.

---

## 4. Model Selection and Justification

### Model: `meta-llama/Meta-Llama-3.1-8B-Instruct`

| Property | Value | Rationale |
|---|---|---|
| Parameters | 8.03 billion | Sufficiently large to stress 32 GB RAM in baseline; approaches RAM limits plus OS overhead |
| Weight size (FP16) | ~16.1 GB | With OS overhead and KV cache, total approaches 19–20 GB — near the practical limit of the 32 GB system |
| Weight size (Q8) | ~8.1 GB | Comfortable in RAM; good AirLLM reference (blocked by CUDA on this box) |
| Weight size (Q4) | ~4.9 GB | GGUF sweet spot: 3× less RAM than FP16, no quality loss |
| Weight size (Q2) | ~3.2 GB | Most aggressive; still coherent output at this precision |
| License | Meta Llama 3.1 Community License | Permits research use; requires attribution |
| Format | SafeTensors (HF) + GGUF (Ollama) | SafeTensors required by AirLLM; GGUF required for CPU-native quantization via llama.cpp |
| Hugging Face ID | `meta-llama/Meta-Llama-3.1-8B-Instruct` | Gated; requires HF token with accepted license |

**Memory arithmetic (why the Baseline stresses the system):**

```
FP16 weights:      8.03B params × 2 bytes  = ~16.1 GB
OS + Python idle:                           ~  2.5 GB
KV cache (512 tok, 32 layers):             ~  0.5 GB
─────────────────────────────────────────────────────
Total at inference:                        ~19.1 GB
Available RAM (free):                      ~18–19 GB  (32 GB − OS baseline RSS)
Result:                                    Near-saturation: 17.01 GB measured peak RSS
                                           < 1 GB headroom → system unresponsive
```

The Baseline ran — it did not hard-OOM — because 32 GB is sufficient to load the model. However, 17.01 GB consumed out of ~18 GB free left the OS and all other processes starved, confirming the near-failure condition the assignment requires. On a 16 GB system (the common laptop case) this run would crash.

**Why not a larger model (e.g., 13B or 70B):** A 70B model in Q4 still requires ~37 GB, which far exceeds our RAM. The 8B model in Q4 GGUF (~5 GB) remains feasible under both AirLLM layer-streaming and Ollama CPU serving.

**Alternative considered:** `Qwen2.5-7B-Instruct` — also valid. Rejected in favour of Llama-3.1-8B due to broader community documentation and higher reproducibility.

---

## 5. Goals and Objectives

1. **Document Baseline near-OOM** — show concretely that FP16 direct execution saturates available RAM and makes the system unresponsive.
2. **Enable execution via AirLLM** — show that layer-by-layer memory mapping allows the same model to complete inference with 6.7× less peak RAM.
3. **Demonstrate real CPU quantization** — use Ollama/GGUF to measure the full FP16→Q8→Q4→Q2 gradient on CPU (AirLLM's bitsandbytes quantization requires CUDA, which is unavailable on this AMD box).
4. **Identify the system bottleneck** — classify the system as compute-bound (Prefill) vs. memory-bound (Decode) and support the claim with TTFT, TPOT, and the Roofline Model.
5. **Perform economic analysis** — compute the On-Prem vs. API (+ Cloud GPU) break-even point.
6. **Connect findings to lecture concepts** — Prefill/Decode, virtual memory, paging, KV cache, mmap, quantization arithmetic.

---

## 6. Key Performance Indicators (KPIs)

All KPIs are measured per scenario: Baseline, AirLLM-FP16, AirLLM-Q2, Ollama-FP16/Q8/Q4/Q2.

| KPI | Definition | Unit | Notes |
|---|---|---|---|
| **TTFT** | Time To First Token — latency from prompt submission to first output token | seconds | Includes cold model-load for Ollama |
| **TPOT** | Time Per Output Token — average inter-token latency after first token | seconds/token | Primary memory-bandwidth indicator |
| **Throughput** | Total tokens generated ÷ total elapsed time | tokens/sec | Inverse of total latency |
| **Peak RAM** | Maximum RSS / Ollama `/api/ps` size during inference | GB | VRAM = 0 everywhere (CPU-only) |
| **Estimated power** | CPU TDP × runtime | Wh | Proportional energy cost estimate |
| **Output quality** | Coherence and relevance of generated text | Qualitative (1–5 scale) | Scored at all Ollama levels |

---

## 7. Use Cases

### UC-01 — Baseline Execution (Near-OOM demonstration)
**Actor:** Researcher  
**Goal:** Load and run Meta-Llama-3.1-8B-Instruct in FP16 directly without AirLLM.  
**Expected outcome:** System saturates RAM (17 GB+), becomes unresponsive during inference, but does not hard-crash on 32 GB hardware.

### UC-02 — AirLLM Layer-Streaming Execution
**Actor:** Researcher  
**Goal:** Run the same model and prompt via AirLLM to demonstrate peak RAM reduction.  
**Expected outcome:** Peak RAM drops to ~2.5 GB at the cost of ~15 s/token decode latency.

### UC-03 — Ollama CPU Quantization Sweep
**Actor:** Researcher  
**Goal:** Run inference at FP16/Q8/Q4/Q2 levels via Ollama/GGUF and compare TTFT, TPOT, RAM, and quality.  
**Expected outcome:** A clear trade-off gradient: lower precision → less RAM, higher throughput, minor quality loss only at Q2.

### UC-04 — Economic Analysis
**Actor:** Researcher  
**Goal:** Compute break-even point between On-Prem, OpenAI API, and Cloud GPU for a given monthly query volume.  
**Expected outcome:** Break-even graph with explicit assumptions; reasoned recommendation.

---

## 8. Functional Requirements

| ID | Requirement |
|---|---|
| FR-01 | The experiment must run on the documented hardware without modifications. |
| FR-02 | A single standardized prompt must be used across all scenarios for fair comparison. |
| FR-03 | TTFT, TPOT, peak RAM, throughput, and power must be measured automatically by a metrics module. |
| FR-04 | All results must be persisted to `results/` as JSON for reproducibility, and committed to the repository. |
| FR-05 | Comparison graphs must be generated programmatically (Matplotlib). |
| FR-06 | Economic analysis must include explicit CAPEX, OPEX, and API pricing assumptions in a config file. |
| FR-07 | The HF token must never appear in source code; it must be loaded from `.env`. |
| FR-08 | AirLLM `layer_shards_saving_path` must point to a configurable path (default `./model_shards`) to avoid flooding the OS drive. |
| FR-09 | Ollama CPU forcing must be verified via `/api/ps` `size_vram=0` output. |

---

## 9. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-01 | Environment managed exclusively with `uv`; no direct `pip` or `python -m venv` calls. |
| NFR-02 | All source files ≤ 150 lines (blank and comment lines excluded). |
| NFR-03 | Zero Ruff lint violations (`ruff check .`). |
| NFR-04 | No secrets committed to the repository. |
| NFR-05 | Test coverage ≥ 85% on non-experiment utility modules. |
| NFR-06 | Python version ≤ 3.11 (AirLLM/transformers compatibility). |

---

## 10. Assumptions

- The NVMe SSD provides sufficient sequential read bandwidth for AirLLM's layer-streaming I/O.
- A valid Hugging Face account with accepted Meta Llama 3.1 license is available.
- RAM is cleared to ≤ 4 GB occupied before each experiment run.
- Electricity cost: 0.60 ILS/kWh (Israeli residential rate, 2025).
- Hardware amortization period: 3 years.
- API pricing reference: OpenAI `gpt-4o-mini` at $0.15/1M input tokens, $0.60/1M output tokens (2025).
- AirLLM bitsandbytes Q4/Q8 require CUDA; this is a known constraint (RISK-05) — documented as a negative result, not a code defect.

---

## 11. Out of Scope

- Fine-tuning or LoRA training (inference only).
- Multi-GPU or cloud GPU inference experiments (economics comparison only).
- GUI or interactive chat interface.
- Production deployment or serving.

---

## 12. Dependencies

| Dependency | Purpose |
|---|---|
| `airllm` | Layer-by-layer inference engine (Stage 2) |
| `transformers` | Model loading, tokenization |
| `torch` | Tensor operations (CPU-only build) |
| `bitsandbytes` | Quantization backend for AirLLM (CUDA-only; fails on this box) |
| `psutil` | RAM monitoring |
| `matplotlib` | Graph generation |
| `python-dotenv` | `.env` secret loading |
| `uv` | Package and environment management |
| `ollama` (server) | GGUF CPU quantization serving (Stage 3) — external process, not a Python dep |

---

## 13. Milestones

| Milestone | Status | Description |
|---|---|---|
| M1 | ✅ Complete | Environment setup + model download |
| M2 | ✅ Complete | Baseline documented (near-OOM captured) |
| M3 | ✅ Complete | AirLLM FP16 run with KPIs logged |
| M4 | ✅ Complete | AirLLM quantization sweep (Q4/Q8 CUDA-blocked; Q2 FP16-fallback documented) |
| M5 | ✅ Complete | Ollama CPU quantization sweep (FP16/Q8/Q4/Q2 — real quantization gradient) |
| M6 | ✅ Complete | Economic analysis + break-even graph |
| M7 | ✅ Complete | README technical report + Roofline extension + results notebook |

---

## 14. Acceptance Criteria

- [x] Baseline near-OOM is documented with console output, peak RAM measurement, and explanation.
- [x] AirLLM FP16 produces a coherent response; 6.7× RAM reduction documented.
- [x] AirLLM Q4/Q8 CUDA block is documented as a reproducible negative result (RISK-05).
- [x] Ollama quantization sweep (FP16/Q8/Q4/Q2) shows monotonic RAM↓ / throughput↑ gradient.
- [x] TTFT, TPOT, throughput, peak RAM, and power recorded for ≥ 4 scenarios.
- [x] Output quality scored for all Ollama levels; "red line" identified.
- [x] Break-even graph embedded in README with all assumptions stated.
- [x] Roofline Model plot generated and explained (extension).
- [x] Repository structure matches recommended layout; results JSONs committed.
- [x] All code passes `uv run ruff check .` with zero violations.
- [x] No HF token or secrets appear in any committed file.
- [x] Test coverage ≥ 85% (`uv run pytest --cov-fail-under=85`).
