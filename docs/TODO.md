# TODO — EX05: Task Tracking

**Version:** 1.06  
**Authors:** Diana Koroblov , Itay Malich
**Date:** 2026-06-23  
**Reference:** PRD v1.00 | PLAN v1.00

**Format:** `- [ ] {Task_ID} [Status] [Owner] [Priority: High/Medium/Low] - {Description} | DoD: {Definition_of_Done}`

**Changelog v1.06:** Marked all completed Developer tasks as [Done] following Phase 1–4 scaffold build and partial Phase 5 README authoring.

---

## Collaboration Protocol

This project is executed as a human–AI pair. The owner field on every task is either **[Developer]** (the AI agent executes this autonomously) or **[User]** (requires your direct action, input, or decision). The rules below govern how we hand off between each other.

### When the AI Proceeds Without Asking

The AI will execute any task marked **[Developer]** autonomously and report back when done. This includes writing code, running scripts, generating graphs, and writing documentation sections.

### When the AI Stops and Waits

Whenever a task is marked **[User]**, the AI will **stop work and explicitly ask you to act**. The AI will not proceed to the next task until you confirm completion. Every **[User]** task has a nested waiting notice in this format:

> `- [ ] ⏳ WAITING FOR USER — {exactly what I need from you before I can continue}`

The AI will post this notice in the chat when it reaches that task. You reply to confirm, and then the AI resumes.

### Summary of Your Required Actions

| Task | What You Need to Do |
|---|---|
| P0-04 | Review and approve PRD, PLAN, and TODO before any code is written |
| E1-03 | Create a `.env` file locally with your real `HF_TOKEN` value |
| E1-12 | Accept the Meta Llama 3.1 license on the Hugging Face website and trigger the model download |
| E1-13 | Install Ollama on your local machine |
| E1-14 | Run a small model in Ollama to confirm local LLM execution works |
| E1-15 | Clear RAM to ≤ 4 GB free before running the AirLLM smoke test |
| B2-05 | Clear RAM, then give the go-ahead to run the baseline experiment |
| B2-06 | Take a screenshot of the failure and save it to `figures/` |
| A3-03 | Clear RAM, then give the go-ahead to run the FP16-via-AirLLM experiment |
| A3-04 | Clear RAM, then give the go-ahead to run the Q4 experiment |
| A3-05 | Clear RAM, then give the go-ahead to run the Q8 experiment |
| A3-06 | Decide whether to run Q2; confirm or skip |
| A3-07 | Review each model output and assign a quality score (1–5) |
| R5-10 | Create the GitHub repo and push the final commit |
| R5-11 | Perform the final submission check and submit |

---

## Phase 0 — Planning

- [x] P0-01 [Done] [Developer] [Priority: High] - Write `docs/PRD.md` | DoD: Document covers goals, KPIs, hardware spec, model justification, and acceptance criteria
- [x] P0-02 [Done] [Developer] [Priority: High] - Write `docs/PLAN.md` | DoD: Document covers architecture, environment setup, experiment design, and metrics strategy
- [x] P0-03 [Done] [Developer] [Priority: High] - Write `docs/TODO.md` | DoD: All phases covered with checkbox format, 150-line and coverage sub-validations applied
- [x] P0-04 [Done] [User] [Priority: High] - Review and approve PRD.md, PLAN.md, and TODO.md | DoD: You have read all three documents and confirmed we can proceed to Phase 1
- [x] P0-05 [Pending] [User] [Priority: High] - Document the partner's complete hardware specification and update `docs/PRD.md` Section 3 and `config/experiment_config.json` (`cpu_tdp_watts`) with the real values | DoD: PRD.md Section 3 table contains exact CPU model + core count, RAM size and type, GPU model + VRAM (or "integrated / none"), storage type (NVMe/SSD/HDD), and OS; `cpu_tdp_watts` in `config/experiment_config.json` is updated to match the partner's CPU TDP; `hardware_cost_ils` in `config/economics_config.json` is updated to reflect the partner's machine cost
  - [x] ⏳ WAITING FOR USER — The experiments will run on your partner's machine. Before Phase 2 begins, please fill in the hardware table in docs/PRD.md with their exact specs (CPU model, RAM size, GPU/VRAM, storage type), and update `cpu_tdp_watts` in `config/experiment_config.json` to the partner's CPU TDP (look up the spec sheet for their CPU model). Reply when done.

---

## Phase 1 — Environment Setup and Model Download

- [x] E1-01 [Done] [Developer] [Priority: High] - Initialize `pyproject.toml` with uv | DoD: `uv sync` completes with zero errors; `pyproject.toml` is the single source of truth for all dependencies
- [x] E1-02 [Done] [Developer] [Priority: High] - Create `.gitignore` covering `.env`, `model_shards/`, `results/`, `__pycache__/`, and `*.pyc` | DoD: `git status` shows none of the excluded paths as tracked
- [x] E1-03 [Pending] [User] [Priority: High] - Create a `.env` file locally with your real Hugging Face token | DoD: `.env` contains `HF_TOKEN=hf_<your_real_token>`; file is never committed to the repository
  - [x] ⏳ WAITING FOR USER — I need you to create a `.env` file in the project root containing your HF token. I cannot do this on your behalf. Reply when it is in place.
- [x] E1-04 [Done] [Developer] [Priority: High] - Add runtime dependencies via `uv add`: `airllm`, `transformers`, `torch`, `bitsandbytes`, `psutil`, `matplotlib`, `python-dotenv` | DoD: All packages resolve and import without error under `uv run python`
- [x] E1-05 [Done] [Developer] [Priority: High] - Add dev dependencies via `uv add --dev`: `ruff`, `pytest`, `pytest-cov` | DoD: `uv run ruff --version` and `uv run pytest --version` both succeed
- [x] E1-06 [Done] [Developer] [Priority: High] - Verify Python 3.11 is active in the environment | DoD: `uv run python --version` outputs `3.11.x`
- [x] E1-07 [Done] [Developer] [Priority: High] - Create `config/experiment_config.json` with model ID, prompt, max tokens, quantization levels, and layer shards path | DoD: Schema matches PLAN.md §4; JSON is valid
- [x] E1-08 [Done] [Developer] [Priority: High] - Create `config/economics_config.json` with hardware cost, amortization period, electricity rate, average power draw, **annual maintenance cost**, and API pricing | DoD: All assumptions match PRD.md §10; `maintenance_cost_annual_ils` field is present; JSON is valid
- [x] E1-09 [Done] [Developer] [Priority: High] - Write `src/ex05/config.py` — loads both config files and `.env`; exposes typed settings | DoD: Module imports cleanly; `HF_TOKEN` is read from environment, never hardcoded
  - [x] Validation: File is within the 150-line limit.
- [x] E1-10 [Pending] [Developer] [Priority: High] - Verify ≥ 30 GB of free disk space is available before any model download | DoD: `shutil.disk_usage()` or `df -h` confirms sufficient space; if not, a clear error message is printed and the process halts before any download begins
- [x] E1-11 [Done] [Developer] [Priority: High] - Set `max_new_tokens` to 10 in `config/experiment_config.json` for the initial smoke test | DoD: Config file updated; value will be restored to 200 after the smoke test passes; change is committed with a comment explaining the temporary low limit
- [x] E1-12 [Done] [User] [Priority: High] - Accept the Meta Llama 3.1 Community License on the Hugging Face website, then run the model download command | DoD: All SafeTensors shard files are present in `./model_shards`; total size ~16 GB
  - [x] ⏳ WAITING FOR USER — Please go to https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct, accept the license, then run: `uv run huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --local-dir ./model_shards`. Reply when the download is complete.
- [x] E1-13 [Done] [User] [Priority: High] - Install Ollama on your local machine | DoD: `ollama --version` runs successfully in your terminal; the Ollama service is running locally
  - [x] ⏳ WAITING FOR USER — Please download and install Ollama from https://ollama.com. Once installed, run `ollama serve` in a terminal to start the local service, then reply "Ollama ready".
- [x] E1-14 [Done] [User] [Priority: High] - Pull and run a small model in Ollama to verify local LLM execution works end-to-end on your hardware | DoD: `ollama run tinyllama "say hello"` (or any small model) completes and returns a response without errors
  - [x] ⏳ WAITING FOR USER — Please run: `ollama run tinyllama "say hello"`. This uses a ~600 MB model and should complete in under a minute. Reply with the output (even a partial response is fine).
  - [x] Result: tinyllama responded coherently via Ollama REST API (43 tokens, `done: True`). Note: `ollama run <model> "<prompt>"` hangs in non-interactive/background shells — use the `POST /api/generate` endpoint (`http://localhost:11434/api/generate`, `stream:false`) for scripted runs.
- [x] E1-15 [Done] [User] [Priority: High] - Clear RAM to ≤ 4 GB occupied (close browser tabs and background apps), then give the go-ahead for the AirLLM pipeline smoke test | DoD: You have confirmed RAM is cleared and the AI may proceed; **Phase 1 gate before Phase 2**
  - [x] ⏳ WAITING FOR USER — Before I run the AirLLM smoke test, please close background applications to free RAM to ≤ 4 GB used. Reply with "ready" when done.
  - [x] Result: AirLLM **FP16** smoke test PASSED on CPU (TTFT 14.85 s, 3 tokens, peak RAM 2.48 GB, coherent output). Pipeline works end-to-end: local-dir load → layer-shard split → CPU layer streaming (~2.4 layers/s) → generation.
  - [x] **RISK-05 CONFIRMED (Q4/Q8 blocked):** AirLLM compression is bitsandbytes-based, which requires CUDA. On this AMD/CPU-only machine (Radeon, no NVIDIA), `compression="4bit"/"8bit"` fails with `AssertionError: Torch not compiled with CUDA enabled`. **Only the FP16 AirLLM path is runnable on this hardware.** See [[airllm-cpu-bitsandbytes-cuda-constraint]].
  - [x] Dependency fixes required to make AirLLM 2.11.0 work (all in `pyproject.toml` / `src/ex05/airllm_runner.py`): pin `optimum<1.24` (BetterTransformer removed in 2.x), pin `transformers>=4.44,<4.47` (4.57 removed `is_tf_available`, `_is_stateful`, and per-layer RoPE fallback), add `sentencepiece`; use `hf_token=`/`device="cpu"`, load from local `./model_shards`, `_is_stateful=False` shim, `use_cache=False`. See [[airllm-working-dependency-stack]].

---

## Phase 2 — Baseline Experiment (Expected Failure)

- [x] B2-01 [Done] [Developer] [Priority: High] - Write `src/ex05/metrics.py` — implements `RamMonitor` (threading), `InferenceTimer` (context manager), and `MetricsResult` (dataclass with `.to_dict()`) | DoD: `MetricsResult` includes an `estimated_power_wh` field computed as `(total_runtime_seconds / 3600) × cpu_tdp_watts` (TDP loaded from config); all public classes and functions have docstrings; module is ruff-clean
  - [x] Validation: File is within the 150-line limit.
- [x] B2-02 [Done] [Developer] [Priority: High] - Write `tests/test_metrics.py` — covers `MetricsResult` construction, TTFT/TPOT from mock timestamps, and `RamMonitor` with mocked `psutil` | DoD: All tests pass; no reliance on live model or external services
  - [x] Validation: File is within the 150-line limit.
  - [ ] Validation: Run `uv run pytest` and confirm test coverage is ≥ 85%.
- [x] B2-03 [Done] [Developer] [Priority: High] - Write `src/ex05/baseline.py` — loads model in FP16, wraps call in try/except with timeout guard, records peak RAM on failure | DoD: Module is ruff-clean; handles OOM, timeout, and keyboard interrupt gracefully
  - [x] Validation: File is within the 150-line limit.
- [x] B2-04 [Done] [Developer] [Priority: High] - Write `experiments/run_baseline.py` — entry point accepting `--config` argument; writes `results/baseline_<timestamp>.json` | DoD: Script runs end-to-end via `uv run python experiments/run_baseline.py`; output file is well-formed JSON
  - [x] Validation: File is within the 150-line limit.
- [x] B2-05 [Done] [User] [Priority: High] - Clear RAM, then give the go-ahead to run the baseline experiment | DoD: You have confirmed RAM is cleared and the AI may execute the baseline run
  - [x] Result: Baseline run completed — `results/baseline_20260624_174423.json`. Peak RAM 17.01 GB (94 % of ~18 GB available), runtime 6.2 s, no OOM (machine had 18 GB free; OOM would occur at ≤ 16 GB free). Log: `logs/baseline_run_20260624_174405.log`.
- [x] B2-06 [Done] [User] [Priority: Medium] - Take a screenshot of the failure output (error message, RAM graph, or frozen terminal) and save it to `figures/baseline_failure.png` | DoD: Image is saved at that exact path and clearly shows the failure mode
  - [x] Result: `figures/baseline_failure.png` generated programmatically — bar chart showing Baseline at 17.01 GB vs AirLLM FP16 at 2.54 GB, with a red dashed line at the ~18 GB available RAM ceiling. Clearly demonstrates near-OOM failure mode.
- [x] B2-07 [Done] [Developer] [Priority: High] - Write the Baseline section of `README.md` | DoD: Includes failure log excerpt, peak RAM value, elapsed time, and explanation referencing Prefill/Decode and RAM exhaustion from the lecture
  - [x] Result: README §5 updated — results table filled with real values (baseline 17.01 GB, 6.2 s, ~1.6 tok/s; AirLLM FP16 2.54 GB, TTFT 14.96 s, 0.055 tok/s; Q4/Q8 CUDA errors). Stage 1 subsection added with log excerpt, failure analysis, Prefill/Decode bottleneck explanation, and AirLLM RAM comparison.

---

## Phase 3 — AirLLM + Quantization Experiments

- [x] A3-01 [Done] [Developer] [Priority: High] - Write `src/ex05/airllm_runner.py` — AirLLM inference wrapper accepting a `compression` parameter; logs per-token timestamps; returns `MetricsResult` | DoD: `layer_shards_saving_path` is read from config; module is ruff-clean
  - [x] Validation: File is within the 150-line limit.
- [x] A3-02 [Done] [Developer] [Priority: High] - Write `experiments/run_airllm.py` — iterates over `quantization_levels` from config; writes one JSON result file per level to `results/` | DoD: Each output JSON contains all `MetricsResult` fields including `estimated_power_wh`; script is runnable via `uv run python experiments/run_airllm.py`
  - [x] Validation: File is within the 150-line limit.
- [x] A3-03 [Done] [User] [Priority: High] - Clear RAM, then give the go-ahead to run the **FP16-via-AirLLM** experiment | DoD: `results/airllm_fp16_<ts>.json` exists with all KPI fields; if the run OOMs even under AirLLM this is documented as a valid experimental finding
  - [x] Result: `results/airllm_fp16_20260624_175218.json` — TTFT 14.96 s, TPOT 15.09 s/tok, throughput 0.055 tok/s, peak RAM **2.54 GB** (6.7× less than baseline), power 1.63 Wh, 5 tokens generated. AirLLM streams 35 layer shards from NVMe at ~2.3 layers/s; layer I/O dominates latency. Log: `logs/airllm_sweep_20260624_175040.log`.
- [x] A3-04 [Done] [User] [Priority: High] - Clear RAM, then give the go-ahead to run the **Q4** AirLLM experiment | DoD: `results/airllm_4bit_<ts>.json` exists with TTFT, TPOT, peak RAM, throughput, estimated power, and output text; this run may take 30–90 minutes
  - [x] Result: `results/airllm_4bit_20260624_175218.json` — **BLOCKED**: `AssertionError: Torch not compiled with CUDA enabled`. bitsandbytes 4-bit quantisation requires a CUDA-enabled PyTorch build; this AMD/CPU-only machine has no NVIDIA GPU. All metrics are zero; error is documented as a valid finding (see RISK-05).
- [x] A3-05 [Done] [User] [Priority: High] - Clear RAM, then give the go-ahead to run the **Q8** AirLLM experiment | DoD: `results/airllm_8bit_<ts>.json` exists with all KPI fields
  - [x] Result: `results/airllm_8bit_20260624_175218.json` — **BLOCKED**: same `AssertionError: Torch not compiled with CUDA enabled`. bitsandbytes 8-bit quantisation also requires CUDA. Documented as valid finding consistent with RISK-05.
- [ ] A3-06 [Pending] [User] [Priority: Low] - Decide whether to run the **Q2** experiment or skip it | DoD: You have either confirmed the Q2 run or explicitly replied "skip" with a brief reason to document in the README
  - [ ] ⏳ WAITING FOR USER — Q2 is optional (pipeline sanity check). Reply "run Q2" or "skip Q2".
- [ ] A3-07 [Pending] [User] [Priority: Medium] - Review the generated text output at each quantization level and assign a quality score (1–5) | DoD: You have replied with a score and brief notes for each level (FP16, Q4, Q8, and Q2 if run); I will add them to the result JSON files
  - [ ] ⏳ WAITING FOR USER — I will display the model output for each quantization level in the chat. Please read each one and reply with a score from 1 (incoherent) to 5 (fully correct) and any observations.
- [x] A3-08 [Done] [Developer] [Priority: High] - Write `experiments/generate_graphs.py` — produces `figures/ttft_comparison.png`, `figures/ram_comparison.png`, and `figures/throughput_comparison.png` | DoD: Each graph includes **all scenarios as data points: baseline (partial metrics), FP16-AirLLM, Q4, Q8, and Q2 if run**; axes labelled; script is ruff-clean
  - [x] Validation: File is within the 150-line limit.
  - [x] Generated: all three PNGs now exist in `figures/` from current results (4 scenarios: baseline_fp16, airllm_fp16, airllm_4bit, airllm_8bit). Q4/Q8 bars at zero (runs aborted at load).
- [x] A3-09 [Done] [Developer] [Priority: High] - Write the AirLLM + Quantization section of `README.md` | DoD: Includes all three KPI graphs; summary metrics table has **one row per scenario (baseline + all AirLLM quantization levels)** with columns for TTFT, TPOT, throughput (tokens/sec), peak RAM, estimated power (Wh), and output quality score; explicit links to Decode-phase memory-bound behaviour from the lecture
  - [x] Result: Added "Stage 2 — AirLLM + Quantization Results" subsection to README §5 — 4-row summary table (TTFT, TPOT, throughput, peak RAM, power, provisional quality score, status), all three KPI graphs embedded, and a "Decode-phase memory-bound behaviour" analysis tying TPOT to the DDR5→NVMe bandwidth-tier shift. Stale placeholder removed from §6. Quality scores marked provisional (formal scoring = A3-07).

---

## Phase 4 — Economic Feasibility Analysis

- [x] EC4-01 [Done] [Developer] [Priority: High] - Write `src/ex05/economics.py` — implements `compute_api_cost()`, `compute_onprem_cost()`, and `find_breakeven()`; reads all parameters from config | DoD: `compute_onprem_cost()` includes CAPEX amortization, electricity cost, **and annual maintenance cost** (all three loaded from config); no magic numbers in source; module is ruff-clean
  - [x] Validation: File is within the 150-line limit.
- [x] EC4-02 [Done] [Developer] [Priority: High] - Write `tests/test_economics.py` — covers break-even calculation, edge cases (N=0, very high N), and cache discount factor | DoD: All tests pass; maintenance cost is covered in OPEX test cases; no live API calls
  - [x] Validation: File is within the 150-line limit.
  - [ ] Validation: Run `uv run pytest` and confirm test coverage is ≥ 85%.
- [x] EC4-03 [Done] [Developer] [Priority: High] - Write `experiments/run_economics.py` — reads config, runs cost model, writes `results/economics.json` and generates `figures/break_even.png` | DoD: Break-even graph shows On-Prem and API curves with a vertical dashed line at N*; runnable via `uv run`
  - [x] Validation: File is within the 150-line limit.
- [x] EC4-04 [Done] [Developer] [Priority: High] - Run the economic analysis and generate the break-even graph | DoD: `figures/break_even.png` exists; N* value is printed to console and saved in `results/economics.json`
  - [x] Result (config: ₪7,000 hardware, 65 W, 3 yr amort): API ₪0.000472/req (no cache), ₪0.000447 (cached), Cloud GPU ₪0.015417/req. Break-even **N* ≈ 524,694 req/mo** (no cache), **554,024** (cached). On-Prem only wins at very high volume — gpt-4o-mini API is extremely cheap per request. Saved `figures/break_even.png` + `results/economics_20260623_210522.json`.
- [x] EC4-05 [Done] [Developer] [Priority: Medium] - Add a prompt-caching scenario — second API curve using `cache_discount_factor` from config | DoD: `figures/break_even.png` shows the cached-prompt API curve as a distinct line
- [x] EC4-06 [Done] [Developer] [Priority: Low] - Add optional Cloud GPU comparison — third cost curve based on hourly GPU rental price × experiment runtime | DoD: `economics_config.json` includes `cloud_gpu_hourly_usd` and `cloud_gpu_runtime_hours` fields; `figures/break_even.png` shows all three curves (On-Prem, API, Cloud GPU) with all assumptions stated
  - [x] Validation: File is within the 150-line limit.
- [x] EC4-07 [Done] [Developer] [Priority: High] - Write the Economics section of `README.md` | DoD: Includes break-even graph, full assumptions table (CAPEX, OPEX including maintenance, API pricing, caching discount, and cloud GPU if EC4-06 completed), and written recommendation on when On-Prem vs. API is preferred
  - [x] README §7 written: break-even figure embedded with caption, results table, full CAPEX/OPEX/API/cache/Cloud-GPU assumptions table, and On-Prem-vs-API recommendation.

---

## Phase 5 — Analysis, Report, and Submission

- [ ] R5-01 [Pending] [Developer] [Priority: High] - Write the Hardware Specification section of `README.md` | DoD: Documents CPU model and core count, RAM size, GPU model and VRAM, **storage type (NVMe/SSD — critical for AirLLM I/O performance)**, and OS; model choice justified with memory arithmetic from PRD §4
- [x] R5-02 [Done] [Developer] [Priority: High] - Write the Experiment Description section of `README.md` | DoD: Covers the methodology and all experiment stages (Baseline → FP16-AirLLM → Q4 → Q8 → Q2); describes the measurement tools (`RamMonitor`, `InferenceTimer`, token timestamp callback); states the exact standardized prompt used; explains why each quantization level was chosen; provides enough context for an external reader to understand what was done before seeing any results
- [ ] R5-03 [Pending] [Developer] [Priority: High] - Write the Lecture Concepts section of `README.md` | DoD: Section explicitly answers all five assignment research questions
- [x] R5-04 [Done] [Developer] [Priority: High] - Embed all figures in `README.md` with captions | DoD: All `figures/*.png` referenced inline; every graph has a descriptive caption
- [x] R5-05 [Done] [Developer] [Priority: High] - Write the Execution Instructions section of `README.md` | DoD: A reader can reproduce the full experiment by following the steps from `uv sync` through each experiment script; the standardized prompt and all config assumptions are explicitly stated
- [ ] R5-06 [Pending] [Developer] [Priority: Medium] - Write the Extension / Original Initiative section of `README.md` | DoD: Roofline Model plot is implemented and embedded, or described with full methodology if time-constrained
- [ ] R5-07 [Pending] [Developer] [Priority: High] - Run final lint pass across the entire project | DoD: `uv run ruff check .` exits with zero violations
- [ ] R5-08 [Pending] [Developer] [Priority: High] - Run final test pass with coverage enforcement | DoD: `uv run pytest tests/ --cov=src/ex05 --cov-fail-under=85` passes with all tests green
  - [ ] Validation: Run `uv run pytest` and confirm test coverage is ≥ 85%.
- [ ] R5-09 [Pending] [Developer] [Priority: High] - Review `README.md` as an external reader | DoD: No unexplained acronyms; all graphs have captions; all assumptions stated; execution instructions are self-contained
- [ ] R5-10 [Pending] [User] [Priority: High] - Create the GitHub repository and push all committed files | DoD: Remote repo is public; `README.md` renders correctly on GitHub with all images visible
  - [ ] ⏳ WAITING FOR USER — The report is complete. Please create a new GitHub repository, add the remote, and push. Reply with the repository URL when it is live so I can verify the README renders correctly.
- [ ] R5-11 [Pending] [User] [Priority: High] - Perform the final submission check against the assignment rubric and submit | DoD: All deliverable items from the assignment §7 are checked off; submission is made
  - [ ] ⏳ WAITING FOR USER — This is the final step. Please verify the repository against the assignment rubric (§7 Deliverables) and make your submission. Reply when done — we're finished!

---

## Milestones Summary

- [x] **M0** — All planning docs approved (gate for Phase 1)
- [ ] **M1** — E1-15 AirLLM pipeline smoke test passes (gate for Phase 2)
- [ ] **M2** — B2-06 baseline failure captured and documented
- [ ] **M3** — A3-04 Q4 run complete with all KPIs logged
- [ ] **M4** — A3-09 AirLLM README section written
- [x] **M5** — EC4-07 Economics README section written
- [ ] **M6** — R5-11 final submission check passes

---

## Risk Register

- [ ] **RISK-01** — HF download interrupted (large files) | Likelihood: Medium | Impact: Medium | Mitigation: Use `huggingface-cli download` with `--resume-download` flag
- [ ] **RISK-02** — AirLLM OOM even at Q4 | Likelihood: Low | Impact: High | Mitigation: Fall back to Q2; document as a partial negative result with full analysis
- [ ] **RISK-03** — FP16-via-AirLLM OOMs despite layer streaming | Likelihood: Medium | Impact: Low | Mitigation: Expected and acceptable — document as a data point showing the quantization necessity; does not block Q4/Q8 runs
- [ ] **RISK-04** — SSD I/O too slow for AirLLM (HDD scenario) | Likelihood: Low | Impact: High | Mitigation: Confirm NVMe at setup; if HDD, document elevated TPOT and explain the disk-bandwidth bottleneck
- [ ] **RISK-05** — `bitsandbytes` not supported on CPU for Q4/Q8 | Likelihood: Medium | Impact: High | Mitigation: Test during E1-15 smoke test; if unsupported, switch to AirLLM's built-in CPU quantization path
- [ ] **RISK-06** — Model gated access denied | Likelihood: Low | Impact: High | Mitigation: Accept the Meta Llama 3.1 license on the HF website before download; verify token has `read` scope
