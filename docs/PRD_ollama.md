# PRD — Ollama CPU Quantization Mechanism (GGUF / llama.cpp)

**Version:** 1.00 | **Component:** `src/ex05/ollama_runner.py` | **Status:** Implemented

---

## 1. Mechanism description

This is the experiment's **real quantization path**. AirLLM's quantization is
bitsandbytes-based and needs CUDA, so it cannot run on this AMD/CPU box. Ollama
serves **GGUF** models through the **llama.cpp** backend, which quantizes and runs
**natively on the CPU**. This module sends the standardized prompt to a local
Ollama server's `/api/generate` endpoint with streaming enabled, forcing CPU
execution, and measures the per-token timing and memory footprint at each
quantization level (FP16, Q8, Q4, Q2).

GGUF k-quants (e.g. `Q4_K_M`) store weights at reduced bit-width with per-block
scales/mins, dequantizing on the fly during the GEMV. Fewer bits per weight ⇒
fewer bytes streamed from RAM per token ⇒ faster memory-bound decode.

## 2. Theoretical background

- **Quantization** trades numerical precision for memory and bandwidth. For an
  8B model: FP16 ≈ 16 GB, Q8 ≈ 8.5 GB, Q4 ≈ 4.9 GB, Q2 ≈ 3.2 GB on disk.
- **Memory-bound decode** (same as AirLLM PRD): throughput scales inversely with
  bytes-per-weight, so quantization should raise tok/s roughly in proportion to
  the size reduction — which the measurements confirm.
- **Determinism for fair comparison:** greedy decoding (`temperature=0`) removes
  sampling noise so quality differences are attributable to quantization alone.

## 3. Inputs / Outputs / Constraints

**Inputs:** `OllamaConfig` (loaded from `experiment_config.json → ollama`): `host`,
`num_predict`, `temperature`, `force_cpu`, `quant_levels` (level → model tag),
plus the shared `prompt`, `seed`, `cpu_tdp_watts`.

**Outputs:** `MetricsResult` per level → `results/ollama_<level>_<ts>.json`;
gradient figure `figures/ollama_quant_comparison.png`.

**Constraints:** an Ollama server must be running (`ollama serve`) and the model
tags pulled (`ollama pull …`, ~32 GB total). Peak RAM is read from `/api/ps`
(`size − size_vram`), **not** process RSS — the mmap'd GGUF is invisible to RSS
(notably on Windows). CPU is forced via `options.num_gpu=0` (verified by
`size_vram=0`).

## 4. Performance expectations vs measured (200 tok, CPU, greedy)

| Level | Peak RAM | Throughput | TPOT | Quality (1–5) |
|---|---|---|---|---|
| FP16 | 15.58 GB | 2.81 tok/s | 0.271 s | 5 |
| Q8 | 8.56 GB | 4.70 tok/s | 0.152 s | 5 (identical to FP16) |
| **Q4** | **5.19 GB** | **9.76 tok/s** | 0.086 s | **5 (sweet spot)** |
| Q2 | 3.57 GB | 12.75 tok/s | 0.056 s | 4 (lost structure) |

RAM ↓4.4×, throughput ↑4.5× FP16→Q2 — as theory predicts.

## 5. Alternatives considered

| Alternative | Why not chosen |
|---|---|
| AirLLM bitsandbytes Q4/Q8 | CUDA-only; impossible on this hardware (the gap this stage fills). |
| `llama-cpp-python` directly | Viable, but Ollama gives one-command model management, a stable HTTP API with built-in timing, and easy CPU forcing. |
| `ollama` Python package | Adds a dependency; raw `urllib` over the documented HTTP API needs no new dep and gives full control over streaming + wall-clock TTFT. |
| GPU offload | The RX 9070 XT lacks working ROCm/HIP on Windows; CPU keeps the comparison consistent with the other stages. |

## 6. Success criteria & test scenarios

- **Success:** all four levels run on CPU and show a monotonic RAM↓ / throughput↑
  gradient with coherent output, answering RQ3's memory↔speed↔quality question. ✅
- **Test scenarios (offline unit tests, `tests/test_ollama_runner.py`):**
  (a) `_build_payload` sets `num_gpu=0` only when `force_cpu`; passes deterministic
  options; (b) `_model_ram_gb` parses `/api/ps` (incl. VRAM subtraction, missing
  model, and connection error → 0.0) with a mocked HTTP call; (c) `_assemble`
  derives TTFT, total runtime, TPOT, token count, and text from synthetic streamed
  events. The live `_stream_generate` path is exercised by `run_ollama.py`.
- **Sanity guard:** a reported peak RAM near 0 for a multi-GB model indicates the
  RSS-vs-`/api/ps` pitfall, not a real measurement.
