# Prompt Engineering Log

**Project:** EX05 — Running a Massive LLM Locally with AirLLM and Quantization
**Method:** Vibe Coding (AI agent orchestrated by the authors as architects/lab managers)

This log documents the major prompts that drove development, their intent, the
outcome, and the lessons learned. It is a required artifact for AI-assisted
projects: it shows *how* the experiment was steered, not just the final code.

The guiding principle throughout (per the course's software guidelines) was
**"define documentation before code"**: PRD → PLAN → TODO were written and
approved first, so each implementation prompt could reference an agreed spec
rather than inventing requirements on the fly.

---

## Phase 0 — Planning & scaffolding

| Prompt intent | Outcome | Lesson |
|---|---|---|
| "Write `docs/PRD.md` for the experiment: purpose, hardware, model choice with memory arithmetic, KPIs, acceptance criteria." | PRD v1.00 with the 8B-model justification and the ~19 GB FP16 budget. | Forcing the memory arithmetic up front predicted the baseline failure mode before any code ran. |
| "Write `docs/PLAN.md` and `docs/TODO.md` with phased tasks, owners, and Definition of Done." | Phased TODO that became the project's single source of progress truth. | A DoD per task made it unambiguous when the agent could mark something done. |
| "Scaffold the package: `src/ex05`, `experiments/`, `config/`, `tests/`, `uv` project, ruff config, ≤150-line rule." | Clean package layout; `pyproject.toml` as the single dependency source. | Setting the 150-line + ruff + 85%-coverage constraints early kept later code disciplined. |

## Phase 1 — Environment & dependency wrangling

| Prompt intent | Outcome | Lesson |
|---|---|---|
| "Install AirLLM and get it importing on Python 3.11 + CPU." | Required pinning `transformers>=4.44,<4.47`, `optimum<1.24`, adding `sentencepiece`. | "Use the newest version" is a trap — AirLLM 2.11 needed *older* transformers. Feed the exact error back to the agent and let it resolve pins. |
| "AirLLM `generate()` crashes with missing `_is_stateful` / cache errors." | Added `_is_stateful=False` shim and `use_cache=False`. | Library-vs-transformers drift is the main friction in local LLM work; small shims beat downgrading everything. |

## Phase 2 — Baseline (direct FP16)

| Prompt intent | Outcome | Lesson |
|---|---|---|
| "Run the model directly in FP16 on CPU and capture what happens — expect OOM/slowness." | It did **not** OOM (18 GB free fit the 16 GB model) but hit 17 GB = 94 % RAM (near-OOM). | The honest result wasn't the predicted one. We documented the *actual* near-OOM failure mode instead of forcing the expected OOM. |
| "Save a proper UTF-8 log of the run." | First attempt via PowerShell `Tee-Object` produced UTF-16; redone via `tee`. | On Windows, watch console encoding — default `Tee-Object` is UTF-16LE. |

## Phase 3 — AirLLM + quantization (the negative result)

| Prompt intent | Outcome | Lesson |
|---|---|---|
| "Run the AirLLM sweep across FP16/Q4/Q8." | FP16 ran (peak RAM 2.54 GB, ~15 s/token). **Q4/Q8 crashed**: `Torch not compiled with CUDA enabled`. | AirLLM's quantization is bitsandbytes → CUDA-only. On an AMD/CPU box this is unavoidable — a legitimate **negative result** worth documenting, not hiding. |
| "Also try Q2." | Q2 "ran" but produced **byte-identical FP16 shards** — a silent fallback, not real 2-bit. | Verify claims physically: comparing shard byte-sizes on disk exposed the fallback. |

## Phase 4 — Economics

| Prompt intent | Outcome | Lesson |
|---|---|---|
| "Build the On-Prem vs API break-even model, config-driven, with caching and an optional Cloud GPU curve." | `economics.py` + break-even graph; N* ≈ 525k req/mo. | Keeping every price/assumption in `economics_config.json` made the analysis transparent and reproducible (a grading requirement). |

## Phase 5 — Report, lint, coverage, extension

| Prompt intent | Outcome | Lesson |
|---|---|---|
| "Write the README as the full technical report (hardware, stages, results, lecture concepts)." | Sectioned README with embedded figures and answers to all research questions. | The README *is* the deliverable — analysis and "why", not just numbers. |
| "Add an original extension." | Implemented a **Roofline Model** (`generate_roofline.py`) proving memory-bound behaviour with hardware constants. | A single unifying figure (baseline on DDR5 roof, AirLLM on NVMe roof) communicated the thesis better than any table. |
| "Final lint + coverage pass." | Fixed 11 pre-existing ruff issues; reached 100 %/92 % coverage on testable modules. | `known-first-party` isort config + omitting live-model modules (documented) made the gate honest. |

## Phase 6 — Real quantization via Ollama (the positive result)

| Prompt intent | Outcome | Lesson |
|---|---|---|
| "Quantization never actually ran (CUDA). Do it for real on CPU via Ollama." | GGUF/llama.cpp sweep FP16→Q8→Q4→Q2: RAM 15.58→3.57 GB, throughput 2.81→12.75 tok/s, Q2 first visible quality dip. | The right *engine* matters more than the right flag: GGUF quantizes on CPU where bitsandbytes can't. This turned the negative result into a complete story. |
| "Peak RAM reads as ~0.1 GB — wrong." | Ollama mmaps the GGUF, so RSS misses it; switched to `/api/ps` `size`. | Measure the thing, then sanity-check the magnitude — a 0.1 GB "8B model" is an obvious red flag. |
| "Force CPU and make it deterministic for fair quality comparison." | `num_gpu=0` (verified `size_vram=0`) + `temperature=0` greedy decode. | Determinism isolates the variable under study (quantization) from sampling noise. |

---

## Cross-cutting lessons

1. **Documentation-first paid off.** Every implementation prompt could cite the
   PRD/TODO, so the agent built to spec instead of guessing.
2. **Feed errors back verbatim.** The fastest path through dependency and
   library-drift failures was pasting the exact traceback and letting the agent
   diagnose.
3. **Negative results are results.** The CUDA wall and the Q2-fallback were not
   failures to hide — documenting them (and then routing around them with Ollama)
   was the most instructive part of the project.
4. **Trust, but verify physically.** Shard byte-sizes, `/api/ps` footprints, and
   roofline sanity-checks caught two measurement/claim errors that the numbers
   alone would have hidden.
5. **The architect stays in the loop.** The agent accelerated implementation
   enormously, but the authors' judgment — choosing the model size, interpreting
   the near-OOM result, deciding to pivot to Ollama — is what made it an
   *experiment* rather than a script.
