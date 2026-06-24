# PRD — AirLLM Layer-Streaming Inference Mechanism

**Version:** 1.00 | **Component:** `src/ex05/airllm_runner.py` | **Status:** Implemented

---

## 1. Mechanism description

AirLLM runs a transformer that does **not fit in RAM** by holding only **one
layer resident at a time**. At init it splits the model into per-layer
SafeTensors shards on disk (`layer_shards_saving_path`). During each forward
pass it iterates the layers in order: memory-maps the next layer's weights,
computes that layer's output, then releases it before loading the next. The full
weight set is therefore streamed from disk **once per generated token**.

This module wraps `airllm.AutoModel` for the experiment: it forces CPU execution,
applies the shims AirLLM needs against current `transformers`, runs a two-pass
generation (1 token for TTFT, N tokens for throughput), and returns a
`MetricsResult`.

## 2. Theoretical background

- **Prefill vs Decode.** Prefill is a parallel GEMM over all prompt tokens
  (compute-heavy); Decode is an autoregressive GEMV, one token at a time, whose
  cost is dominated by moving weights through the memory hierarchy.
- **Virtual memory / paging analogy.** AirLLM is application-level paging: the
  "page" is a transformer layer, the "backing store" is the NVMe SSD, and a
  "page fault" is the mmap read when a layer's turn comes. Unlike OS swap (random,
  involuntary, thrash-prone), AirLLM's access is sequential, voluntary, and
  scheduled — one ordered pass per token.
- **Memory-bound decode.** Because each weight is read once per token (arithmetic
  intensity ≈ 1 FLOP/byte), throughput is bounded by the storage tier's bandwidth,
  not by FLOPs.

## 3. Inputs / Outputs / Constraints

**Inputs:** `ExperimentConfig` (model id, prompt, `max_new_tokens_experiment`,
`layer_shards_path`, `cpu_tdp_watts`), `compression ∈ {None, "4bit", "8bit",
"2bit"}`, `HF_TOKEN` from `.env`.

**Outputs:** `MetricsResult` (TTFT, per-token timestamps → TPOT, throughput, peak
RSS, total runtime, estimated power, output text, error) → `results/airllm_<level>_<ts>.json`.

**Constraints:** Python ≤ 3.11; `transformers>=4.44,<4.47`; `optimum<1.24`;
`sentencepiece`. Requires `_is_stateful=False` shim and `use_cache=False`.
`layer_shards_saving_path` must point to a fast, non-OS drive partition with
≥ 2× the model size free.

## 4. Performance expectations vs measured

| Metric | Expectation | Measured (FP16) |
|---|---|---|
| Peak RAM | ≪ full model (one layer) | **2.54 GB** (vs 17 GB baseline) |
| TPOT | High; NVMe-bound | **15.09 s/token** |
| TTFT ≈ TPOT | Yes (I/O dominates both) | 14.96 vs 15.09 s |

## 5. Alternatives considered

| Alternative | Why not chosen |
|---|---|
| Direct FP16 load (baseline) | Needs ~19 GB — near-OOM / fails on ≤16 GB. AirLLM is the fix being studied. |
| `device_map="auto"` disk offload (accelerate) | Similar idea, but AirLLM is purpose-built for layer streaming and is the assignment's named tool. |
| GGUF/Ollama | Different goal: Ollama solves *quantization on CPU*, not *fitting an un-loadable model*. Used as a complementary stage, not a replacement. |

## 6. Success criteria & test scenarios

- **Success:** model that fails/near-fails in direct FP16 completes via AirLLM
  with peak RAM far below the full model size, and KPIs are persisted. ✅
- **Negative finding (documented):** `compression="4bit"/"8bit"` aborts with
  `Torch not compiled with CUDA enabled` (bitsandbytes needs CUDA; this box is
  AMD/CPU). ✅ recorded as a valid result, not a defect.
- **Test scenarios:** (a) FP16 end-to-end produces non-empty coherent text;
  (b) Q4/Q8 error is caught and saved in the `error` field, metrics zeroed;
  (c) Q2 fallback detected by byte-identical shard sizes. Live model paths are
  validated by running `experiments/run_airllm.py` (not the offline unit suite).
