Assignment 05

EX05 - Running a Massive LLM Locally: AirLLM, Quantization and Performance Benchmarking

Keywords: AirLLM, quantization, LoRA, Ollama, Hugging Face, local inference, On-Premises LLM, GPU vs CPU, Prefill, Decode, VRAM, virtual memory, Paging, Throughput, Latency, performance metrics, benchmarking, SafeTensors, GGUF, technical report

Table of Contents
Assignment Overview  
The Core Mission - In-Depth Technical Report  
Objectives  
Research Questions and Understanding  
Core Tasks  
   5.1 Hardware documentation and justification for model selection  
   5.2 Baseline - direct execution  
   5.3 Integrating AirLLM and quantization  
   5.4 Performance measurement and comparison  
   5.5 Economic feasibility analysis: On-Prem vs API  
   5.6 Analysis of lecture concepts  
   5.7 Extensions and original initiatives  
Planning and Efficient Work  
   6.1 Do  
   6.2 Don't  
Submission Deliverables  
Detailed README Requirements  
Recommended Repository Structure  
Expectations from the Assignment  
Appendix - Realistic Time Estimation Using the Vibe Coding Method  
   11.1 Stage 1: Installation, environment setup, and model download  
   11.2 Stage 2: Runs, experiments, and measurements  
   11.3 Stage 3: Data processing, performance comparison, and economic analysis  
   11.4 Stage 4: Integrating everything and writing the technical report (README)  
   11.5 Summary of time estimation  

Assignment Overview

In this assignment, you are required to demonstrate, in a practical and well-reasoned way, that you understand the full On-Premises workflow of running a large language model locally on your own hardware. The central idea is to take one model that is too large, fails, or runs unbearably slowly in direct execution, and then apply optimization techniques - AirLLM and quantization - to make that same model run anyway, and to analyze in depth the cost and benefit of each approach.

General guideline: This is an open-ended, investigation-based assignment. You are expected to design the experiment yourselves, choose the model and the hardware, and justify every decision. The goal of the assignment is not the quality of the model's output, but rather understanding the inference mechanisms, drawing conclusions, and presenting a deep engineering and economic analysis.

The work must be submitted as a complete GitHub repository including code, experiments, and a comprehensive technical report.

Important Consideration - Scope of the Experiment and Model Size Selection

The purpose of the assignment is not to run models for hours or days. You must analyze the computational power of your personal computer and, accordingly, choose a model that is large - but not too large. Such a model should make it possible to demonstrate high latency, or an inability to run directly, versus the improvement and load relief enabled by AirLLM.

Remember that this is an experiment: if you were unable to improve performance, it is acceptable and even desirable to present a negative result, provided that you explain and justify why things behaved that way in your case. A well-analyzed negative result is worth no less than a positive one.

The Core Mission - An In-Depth Technical Report

At the heart of the assignment is an in-depth technical report (deep-dive technical report) documenting your experiment. Below is the definition of the central task as you are expected to implement it:

Mission Statement

Write a comprehensive and in-depth technical report (deep-dive) documenting your practical experiment in running a large language model (LLM) locally. Begin by documenting the exact specification of your computer hardware (CPU/GPU) in order to justify your model choice, and document live what happens - especially the unavoidable performance bottlenecks - when attempting to run it directly on your processor.

Then integrate AirLLM and a quantization mechanism into the pipeline, run the experiment again, and show how these optimization techniques fundamentally change resource allocation. The final narrative must go beyond raw data and deeply analyze the lecture concepts using comparative metrics, illustrative graphs, and original experimental design that demonstrates a thorough understanding of On-Premises LLM Deployment.

Objectives

The assignment should demonstrate the ability to plan and execute a complete engineering experiment, measure it, adapt it to the hardware, and analyze it from both a technical and an economic perspective. You are expected to identify the real bottleneck (compute vs memory), apply optimization techniques in a controlled way, and quantify their impact.

You are required to present not only what happened, but also why it happened - linking every result to lecture concepts such as Prefill/Decode, VRAM, virtual memory, and quantization - and drawing conclusions about when each approach is worthwhile.

A key objective of the assignment is to demonstrate the use of standard industry performance metrics. The broader goal is to prove with data - not merely by hypothesis - whether the chosen model and hardware are compute-bound during input processing, or memory-bound during output generation. An advanced aspiration is to draw a Roofline Model that visually illustrates when the system transitions from one resource limitation to another.

Research Questions

Throughout the assignment, you must explicitly address the following questions as part of the report:

• What was the bottleneck that prevented direct execution - memory (RAM/VRAM) or compute power? How did you identify it?
• How does AirLLM change resource allocation, and what is the connection to virtual memory and the paging mechanism used by operating systems?
• What was the effect of quantization on memory consumption, speed, and output quality? Where was the "red line" of accuracy crossed?
• How are the Prefill and Decode phases reflected in your measurements, and how do they appear in the separation between TTFT (Time To First Token), which represents compute load, and TPOT (Time Per Output Token), which represents memory load?
• What is the Throughput/Latency price you pay for the ability to run a large model on modest hardware?
• When is it economically worthwhile to work locally, and when is it preferable to use an external API?

Core Tasks
5.1 Hardware Documentation and Justification for Model Selection

Document the exact specification of your computer: the CPU model, number of cores, RAM size, GPU model and VRAM size (if available), as well as storage type (NVMe/SSD). Based on this specification, choose a model from Hugging Face that is too large to fit comfortably on your hardware, and explain the logic behind the choice (number of parameters, format, size).

5.2 Baseline - Direct Execution

Attempt to run the chosen model directly on your hardware (for example via Ollama or Hugging Face). Document live what happens: does the model fail to load, get stuck, or run with unbearable slowness? Identify the bottleneck and explain it. This is the baseline against which you will compare the rest of the experiments.

5.3 Integrating AirLLM and Quantization

Integrate AirLLM and a quantization mechanism into your pipeline, and run exactly the same task again. Show how these techniques fundamentally change resource allocation and allow the model to run. Document which quantization levels you tested (for example FP16, Q8, Q4) and what effect each had.

5.4 Performance Measurement and Comparison

For each scenario, systematically measure at least the following metrics, and present them in tables and graphs:

• TTFT (Time To First Token) - the time from request submission until the first token is produced; a metric for the load of the Prefill phase (including KV cache construction and compute intensity).
• TPOT (Time Per Output Token) / ITL (Inter-Token Latency) - the rate of token flow after the first token (milliseconds per token); a metric for the Decode stage load and data movement from memory.
• Total Throughput (in tokens/sec).
• Peak memory consumption (RAM and VRAM).
• Total runtime and estimated power consumption.
• A qualitative assessment of output quality at each quantization level.

5.5 Economic Feasibility Analysis: On-Prem vs API

This is a general and mandatory requirement. You must perform a cost calculation and compare execution of the exact same task in two ways, to show when it is economically preferable to work locally (On-Premises) and when it is better to use a third-party API:

Third-party API cost  
   Calculate the number of tokens (input + output) for each request, and multiply by the token pricing of a provider of your choice (for example OpenAI, Claude, and so on). Present the cost per single request and the cost for a given volume of requests.

Local On-Prem cost  
   Calculate hardware cost (CAPEX) amortized over time, plus electricity and maintenance cost (OPEX). Derive from this an effective cost per request as a function of usage volume.

Based on the two calculations, find the break-even point: from what usage volume (number of requests or tokens per month) does local work become cheaper than the API? Present this in a graph of cumulative cost vs usage volume, and formulate a reasoned recommendation: in which scenarios the API is preferable, and in which On-Prem is preferable, including privacy and data security considerations, not just cost.

It is worth noting a current pricing phenomenon that changes the picture in API cost calculations: Prompt/Context Caching. Providers based on the PagedAttention architecture keep the fixed parts of the system prompt in memory for repeated use, and therefore offer a significantly reduced rate for those recurring input tokens without recomputing hidden states during Prefill. In scenarios involving repeated questions over a long document, this mechanism may shift the break-even point - consider this in your analysis.

Optional - Third Option: Cloud GPU

Students who wish may add a third comparison option: the cost of performing the same task on a rented Cloud GPU from a cloud provider (for example, GPU hourly price multiplied by runtime). Integrate it into the same break-even graph alongside API and On-Prem.

You must explicitly state all assumptions underlying the calculation (prices, usage volume, hardware lifetime, electricity rate) so that the analysis is transparent and reproducible.

5.6 Analysis of Lecture Concepts

Relate every finding to the lecture concepts. Explain the results through Prefill vs Decode, compute-bound vs memory-bound, the role of VRAM, and the analogy to virtual memory and the paging mechanism on which AirLLM is based.

5.7 Extensions and Original Initiatives

You are required to propose at least one original extension: the assignment is only a guideline. This may be an additional experiment design, a new metric, an interesting comparative graph, integration of an additional training or adaptation technique, or comparison between multiple models of different sizes (for example LoRA/QLoRA).

Planning and Efficient Work

The purpose of this section is to help you complete the assignment within a reasonable amount of time and without wasting resources. The most important principle is to start small, verify that the pipeline works, and only then scale up.

6.1 Do
• Create an isolated virtual environment (uv is recommended).
• Check your Python version - do not use the newest version, since many packages are still not adapted to it.
• Start with a small model and an aggressive quantization level (for example Q2) just to make sure the "pipeline" works before moving to the large model.
• Set a low maximum token count in the initial tests.
• Make sure there is enough free disk space before downloading large models.
• Properly manage where AirLLM saves layer shards: decomposing a heavy model creates many large SafeTensors files and heavy Disk I/O. Explicitly define the layershardssaving_path to direct the cache to a fast drive or dedicated partition, thereby avoiding flooding the operating system drive (C: during the experiment).
• When working with models from the Qwen family and similar ones, use the general AutoModel function during initialization so that the system adapts to the correct class and avoids Class mismatch errors.
• Measure consistently and save all raw numerical data for later graphing.

6.2 Don't
• Do not choose a gigantic model that has no chance of running even with AirLLM.
• Do not save the Hugging Face token explicitly in the code.
• Do not settle for raw data without analysis, graphs, and linking back to lecture concepts.
• Do not ignore the economic aspect - cost analysis is an essential part of the assignment.
• Do not turn the assignment into a final project; a focused, clean, and well-analyzed experiment is preferable.

Deliverables

The submission must be a complete GitHub repository including at least:

• Full experiment code and scripts for execution and measurement.
• The in-depth technical report, including hardware documentation, the baseline, AirLLM, quantization experiments, and all graphs, tables, and screenshots.  
  The report must serve as the README file of the GitHub project, and everything must be embedded directly inside the README itself.
• Comparative tables and graphs of performance metrics.
• Economic feasibility analysis (On-Prem vs API, and optionally cloud GPU), including the break-even graph and all assumptions.
• Analysis connecting the results to lecture concepts.
• Documentation of extensions and original ideas.

Detailed README Requirements

The README.md file is an essential part of the submission and must be clear and readable to an external reader. The README must include:

• Hardware specification and justification for model choice.
• Description of the experiment, its stages, and the measurement tools.
• Summary of findings: baseline vs AirLLM vs quantization.
• Summary of the economic feasibility analysis and a reasoned recommendation.
• Explanation connecting the results to lecture concepts.
• Clear execution instructions for reproducing the experiment.

In addition, the README must include visual elements: tables, comparison graphs, and screenshots that support the analysis and presentation of the process.

Recommended Repository Structure

A possible submission repository structure:

• README.md
• pyproject.toml or requirements.txt
• src/
• experiments/
• results/
• reports/
• figures/

You may adapt the structure to the project's needs, but it should remain consistent, clear, and easy to navigate.

Expectations from the Assignment

The expected work is a complete and analyzed engineering experiment that demonstrates genuine understanding of running large models locally. You are expected to show the ability to adapt a model to hardware, identify bottlenecks, apply optimization techniques, measure systematically, and analyze the results both technically and economically.

This assignment is not intended to be a mechanical execution of technical steps alone. It is designed to examine how you plan an experiment, how you measure and present data, and how you conclude - on the basis of metrics and cost calculations - when local model deployment is preferable, and when working with an external API is the correct choice.

Appendix - Realistic Time Estimation Using the Vibe Coding Method

Since you are carrying out this assignment using the Vibe Coding paradigm - meaning the use of AI agents based on large language models to write code and orchestrate the experiments - the development experience and time management differ completely from traditional coding.

In this assignment, you function as the architect and lab manager, while the AI agents are the ones who "get their hands dirty" in actual implementation. It is important to understand that although coding time, script writing, and debugging are dramatically shortened thanks to the leverage of LLMs, the physical hardware limitations, network download processes, memory mapping, and disk I/O operations remain unchanged. These are what will dictate the pace.

Below is an honest and realistic time analysis that will help you set expectations and know whether you are operating within a reasonable schedule.

11.1 Stage 1: Installation, Environment Setup, and Model Download

Active work time: about 15 minutes  
Estimated total time: 1.5 to 3 hours (mostly passive waiting time)

What happens here:  
Requesting the AI agent to set up a virtual environment (venv), install AirLLM, and write a download script from Hugging Face takes only seconds. The real bottleneck is downloading huge model files (tens of gigabytes), depending on your bandwidth. In addition, AirLLM performs sharding of the model into separate layers on disk - a physically heavy task that software cannot bypass.

11.2 Stage 2: Runs, Experiments, and Measurements

Active work time: 30 to 45 minutes  
Estimated total time: 3 to 5 hours (physical compute time)

What happens here:  
The agent will immediately write the precise measurement script to calculate TTFT and TPOT. However, when producing the baseline, you may need to wait until the computer fills physical memory and swap memory, which can cause the system to hang for several minutes. Running the model through AirLLM is inherently slow: the library uses mmap to load and unload each transformer layer separately from the SSD, and generating each token requires massive disk reads. You will need to run the experiment several times for different quantization levels, so simply wait patiently.

Tip:  
If debugging drags on for hours - stop, feed the error message back to the agent, and let it work.

11.3 Stage 3: Data Processing, Performance Comparison, and Economic Analysis

Active work time: about 30 minutes  
Estimated total time: 1 to 1.5 hours

What happens here:  
This is where the greatest advantage of Vibe Coding appears. Feed the measurement results and hardware costs to the agent, and ask it to generate Python code (for example with Matplotlib) that plots the break-even graph. Let the agent perform the electricity and depreciation calculations, and instruct it to incorporate modern assumptions such as cached prompts in APIs if relevant. Here, the physical waiting time is short, and most of the value comes from fast analytical iteration.

11.4 Stage 4: Integrating Everything and Writing the Technical Report (README)

Active work time: 45 to 60 minutes  
Estimated total time: 1 to 2 hours

What happens here:  
At this stage, the agent can help assemble the entire story into a polished technical report: generating tables, drafting explanations, writing the interpretation of the graphs, and organizing the README structure. Your role is not to type everything manually, but to review critically, verify technical correctness, make sure the explanations truly match the data, and ensure that the final report reflects your understanding.

11.5 Summary of Time Estimation

A realistic total estimate for the assignment is:

• Active student work: about 2 to 3 hours
• Total elapsed wall-clock time: about 7 to 11.5 hours

The gap between active work time and total elapsed time stems from the fact that the student no longer spends most of the time writing code manually, but still depends on download speed, disk performance, memory limits, and repeated model inference runs.

In other words: AI dramatically reduces the human effort, but it does not eliminate the physical runtime of the experiment.

If you want this in a cleaner submission-ready form, the next logical step is to turn it into polished academic English and fix a few awkward OCR fragments from the source.