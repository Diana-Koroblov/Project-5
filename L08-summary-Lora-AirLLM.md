Below is the full English translation of the file.

Local Inference and Training of Large Language Models

On-Premises LLM Deployment: Utilizing Ollama, LoRA, and AirLLM (L08)

Dr. Yoram Segal

All rights reserved - Dr. Yoram Segal

June 2026

L08 Lesson Summary

Local Inference and Training of Large Language Models  
Lecturer: Dr. Yoram Segal

Keywords: inference, training, GPU vs CPU, CUDA, quantization, VRAM, memory, Prefill phase, Decode phase, Hugging Face, Ollama, SafeTensors format, GGUF format, LoRA, PTX, SASS, Virtual Memory, Paging, AirLLM, PEFT, OLoRA, QLoRA, model licensing.

This summary describes the complete workflow for working locally with large language models: from hardware (GPU/CPU), through formats, licensing, and quantization, to efficient training with LoRA and running huge models on CPU using AirLLM and the virtual-memory principles of operating systems.

© All rights reserved to Dr. Yoram Segal

Table of Contents
Introduction: Three Ways to Run Models  
   1.1 The three approaches to running models  
   1.2 Cost and information security in local work  

Hardware: CPU vs GPU  
   2.1 Parallelism vs serial processing  
   2.2 CUDA, PTX, and SASS  
   2.3 Extension - Fat Binary and Just-In-Time (JIT) compilation  
   2.4 Advanced topic - the evolution of Warp Divergence: from Pascal to Volta  

Inference: Prefill and Decode  
   3.1 The two stages of inference  
   3.2 Intermediate representations (Latent)  
   3.3 VRAM - graphics-card memory  
   3.4 Extension - the mathematical basis and the KV Cache  
   3.5 Advanced topic - distributed inference (Disaggregated Serving)  

Working with Hugging Face: Choosing a Model  
   4.1 Steps in model selection  
   4.2 Model licensing  
   4.3 Formats: SafeTensors and GGUF  
   4.4 Extension - why SafeTensors is safe: pickle vs Flat Buffer  

Quantization  
   5.1 Advanced topic - QLoRA and NormalFloat (NF4)  

Ollama: Local Execution  

Training and LoRA  
   7.1 Training vs inference  
   7.2 PEFT and Transfer Learning  
   7.3 The idea behind LoRA  
   7.4 QLoRA and OLoRA  
   7.5 Extension - the mathematical basis of LoRA and OLoRA  

AirLLM: Running Massive Models on CPU  
   8.1 The idea  
   8.2 Reminder: virtual memory from operating systems  
   8.3 The price: latency vs memory  
   8.4 Extension - how AirLLM uses mmap  
   8.5 Advanced topic - the research frontier: FlexGen, LLM in a Flash, and PagedAttention  

The Practical Assignment  
   9.1 What needs to be done  
   9.2 Summary of the journey  

Introduction: Three Ways to Run Models

In this lesson we go through the full path of working locally with AI agents on our own computers. First, we will discuss hardware and the difference between CPU and GPU, and how the transformer architecture is reflected in hardware. Then we will see how models are downloaded using Hugging Face and Ollama, and we will address the problem that large models are very difficult to run. Finally, we will present two central methods: LoRA for efficient training of very large models, and AirLLM for running large models on a regular CPU.

1.1 The Three Approaches to Running Models

There are three main options for running models:

API - the most basic and simplest approach. You use an external service such as OpenAI or Claude, pay per token, and work. This is always a good starting point for a new project: run an experiment, and if it fits, continue.

Cloud GPU - purchase GPU servers in the cloud and run the computation remotely. The advantage: no need to buy hardware; you pay by usage. The disadvantage: more expensive over time.

Local (On-Premise) - run all models on your own computer. The stronger the computer, the easier life becomes, but there is a cost in computing power.

Table 1: Comparison of the three execution approaches

| Approach | Advantages | Disadvantages |
|---|---|---|
| API | Maximum simplicity, no hardware, quick start | Pay per token, data leaves the organization |
| Cloud GPU | No hardware purchase, pay by usage | More expensive, data still in the cloud |
| Local On-Premise | Full privacy, no ongoing token payments | Upfront capital investment, electricity cost |

1.2 Cost and Information Security in Local Work

In local On-Premise work, the cost shifts to capital expenditure: instead of paying per use, you buy a strong machine in advance, sometimes costing tens of thousands of dollars, and hope the investment pays for itself.

The enormous advantage, which in some security systems has no price, is information security: nothing leaves the organization. In the IDF and in sensitive environments, this is a decisive consideration, because once information is sent to the cloud, you can never know for certain who is listening in the middle or what is being done with the data.

Hardware: CPU vs GPU
2.1 Parallelism vs Serial Processing

A GPU has thousands of cores operating in parallel - massive multithreading. A CPU has relatively few cores, but it is more efficient when the task is serial and complex, meaning when one operation depends on the result of the previous one. There is no advantage to a GPU in serial tasks.

Transformer models perform many matrix multiplications in a fixed pattern, where only the values change but the operation is identical and repeated many times. Because of that, the GPU is faster by orders of magnitude than the CPU, and is therefore the ultimate solution for large language models.

2.2 CUDA, PTX, and SASS

CUDA is NVIDIA's programming platform. One writes in it using an extended version of C++ with special libraries. The compilation chain goes through three stages:

CUDA - source code in an extended language.
PTX - a universal GPU assembly language that is not tied to a specific graphics card and is suitable for all GPUs.
SASS - the stage in which the code is compiled specifically for the exact hardware of the graphics card, one-to-one.

Two Important Phenomena

Warp Divergence: when a branch appears during parallel execution - some operations go one way and others another way - processing becomes serial and the parallelism advantage is lost. Therefore, processing is designed in advance so that identical operations are grouped together.

Cache: compilation is done again on each run, like an interpreter. What saves us from repetition is the card's cache: if nothing changed, there is no need to compile again.

2.3 Extension - Fat Binary and Just-In-Time (JIT) Compilation

The NVCC compiler packages the virtual PTX code and the dedicated SASS code together into a combined file called a Fat Binary. This approach simultaneously achieves maximum optimization through SASS and forward compatibility through PTX, which serves as a virtual instruction set.

SASS is the binary code matched one-to-one to the specific card. PTX is the virtual code. When running the program on a new card for which no matching SASS exists in the file, the driver performs JIT Compilation, compiling from PTX to the required SASS in real time. This process creates an initial delay, so the driver stores the result in a dedicated JIT Cache.

This explains why the first run of a model or a new library version, such as after updating PyTorch or Triton, may be significantly slower than subsequent runs.

2.4 Advanced Topic - The Evolution of Warp Divergence: From Pascal to Volta

The Warp Divergence mechanism underwent an important architectural evolution. In architectures before Volta (such as Pascal), all threads in a warp shared a single Program Counter. Therefore, the hardware had to use an Active Mask and disable threads that were not on the current branch, which led to serial processing and loss of parallelism in that cycle.

Starting with Volta architecture and onward (V100, A100, H100), Independent Thread Scheduling was introduced. This gives each thread its own program counter and call stack, enabling divergence and reconvergence at a finer granularity, even though the SIMT principle still remains at the core level.

Even so, the programming goal remains to minimize divergence in order to exploit the compute cores as fully as possible.

Inference: Prefill and Decode

When running a trained model to obtain an answer, inference is performed in two stages with completely different characteristics.

3.1 The Two Stages of Inference

Prefill stage: the model receives the entire context window at once, performs all matrix multiplications in parallel on the GPU, and computes all intermediate representations. At the end of this stage, only one token is obtained. The bottleneck here is the GPU's compute power - it is compute-bound.

Decode stage: the produced token is inserted back into the context window, the next token is generated, and so on - token by token. This is a serial operation, so the bottleneck is different. Since in Prefill all weights were already processed, in each token only a small part changes, but the weights must be brought from memory each time. Therefore, the bottleneck here is memory - specifically GPU memory - and the stage is memory-bound.

Table 2: Comparison between Prefill and Decode

| Feature | Prefill | Decode |
|---|---|---|
| Computational nature | Parallel | Serial |
| Algebraic form | Matrix-matrix multiplication (GEMM) | Matrix-vector multiplication (GEMV) |
| Output | First token + KV Cache construction | Single token per iteration |
| Bottleneck | Compute power | Memory |
| Main limitation | Number of cores and FLOPs | Bandwidth and VRAM size |

3.2 Intermediate Representations (Latent)

The intermediate representations are the results passed from layer to layer in the model - the latent states. Each layer performs multiplications and produces its own representation.

The subtle idea is that if the result of certain layers is already known and does not change because of an added token, there is no need to recompute the entire path - similar to a state machine.

3.3 VRAM - Graphics Card Memory

The term VRAM (Video RAM) is a historical name originating from graphics cards. This is the memory of the GPU. It is one of the most important parameters when buying a graphics card.

For the Prefill stage, the number of cores is important; for the Decode stage, what matters most is VRAM size. It is recommended to check the technical specifications of the graphics card in your computer.

3.4 Extension - The Mathematical Basis and the KV Cache

The difference between the stages stems from the mathematics behind them. The Prefill stage is based on GEMM (Dense Matrix-Matrix Multiplication), which efficiently exploits the GPU's Tensor Cores, and is therefore limited only by compute power and FLOPS. The output of this stage is the KV Cache - the intermediate state stored in memory and used for continued text generation.

The Decode stage, by contrast, requires multiplying the weight matrix by the vector of a single token (Matrix-Vector multiplication). The amount of computation per weight drops, and the cores wait while the weights and the growing KV Cache are fetched from physical memory (VRAM/HBM). Therefore, this stage is limited by memory bandwidth.

3.5 Advanced Topic - Distributed Inference (Disaggregated Serving)

The distinction between the conflicting bottlenecks led to a modern research-frontier paradigm: Disaggregated Serving. Leading studies from 2024, such as DistServe (OSDI 2024) and Splitwise (ISCA 2024), showed that combining Prefill and Decode tasks on the same card creates interference that harms performance.

The solution is physical separation: a group of compute-heavy cards performs only Prefill and transfers the KV Cache over a fast network to a second group of cards optimized for memory and bandwidth, which performs only Decode. This avoids the conflict and achieves maximum cost-effectiveness in data centers.

Working with Hugging Face: Choosing a Model

Hugging Face is the main source for models - the "YouTube of language models." Anyone who wants to be considered a professional in the field should have an account there. When choosing a model, one should go through a sequence of steps.

4.1 Steps in Model Selection
Task - decide what you want: a model that answers people, a model that generates images, sound, and so on.
License - check the licensing terms.
Format - verify that the format fits your working environment.
Model size - the million-dollar question. A large model is not necessarily appropriate for the task; you must check whether it fits in VRAM.
Latency - how quickly a response is obtained and whether that meets expectations.

The Truck and Motorcycle Analogy

A larger model is not necessarily better for your task. A truck is larger than a motorcycle, but it is not necessarily better at maneuvering through traffic. Choosing the right model size is critical.

4.2 Model Licensing

Models on Hugging Face have license terms. Licensing can bring down an entire organization. Some licenses are completely open; others allow use under various conditions, such as requiring that you state you used the model; and some allow use only for learning, while commercial use requires permission from the owner.

It is always recommended to consult legal counsel in a serious organization.

Rule of Thumb for License Dependency

If you remove the licensed component and your product stops functioning - meaning it has no substitute without it - then you depend on it and are legally exposed. If the product keeps working without it, meaning the component is not essential, the legal exposure is lower.

This is not legal advice.

4.3 Formats: SafeTensors and GGUF

SafeTensors: a format that contains no code at all - no read/write commands. It contains only a description of the layers and architecture (metadata in JSON) and the list of weights. Therefore, it is safe to load: nothing in it is executable code. Risks still exist, for example a model intentionally trained maliciously, but the risk level is reduced. In an organization, if you encounter models that were not downloaded in SafeTensors format, that should raise a red flag.

GGUF: a format that packages weights and metadata into a compressed file convenient for local execution. The file contains the technical specification of the model and allows convenient local control. Ollama knows how to read it.

4.4 Extension - Why SafeTensors Is Safe: pickle vs Flat Buffer

To understand the advantage of SafeTensors, it is useful to know the risk in older traditional formats. Files such as pytorchmodel.bin rely on Python's pickle module, which is intended for object and code reconstruction. Therefore, loading an infected model may execute arbitrary malicious code on the target machine when the file is loaded into memory.

SafeTensors was designed from the ground up as a flat data structure (Flat Byte Buffer) containing only JSON metadata and raw numerical arrays, with no ability to contain executable code. Beyond security, this structure also allows direct memory mapping (mmap) of the file from disk into virtual memory, dramatically accelerating layer-loading times.

Quantization

The model weights are numerical values represented in Floating Point. The problem is how many digits to store after the decimal point. Quantization is a method of compressing information at the expense of precision: reducing the number of bits per weight saves memory.

For example, moving from FP32 to FP16 cuts memory usage in half - 16 bits per weight instead of 32. The price is a possible drop in accuracy, but sometimes this is actually unnecessary overhead, and some studies have even found that reducing precision gave the model greater flexibility and better results in certain cases.

Table 3: Typical quantization levels (bits per weight)

| Representation | Bits | Note |
|---|---:|---|
| FP32 | 32 | Full precision, maximum memory usage |
| FP16 | 16 | Half the memory, good balance |
| FP8 | 8 | Significant compression |
| Q4 | 4 | Aggressive quantization |
| Q2 | 2 | Extreme compression for testing "pipeline release" |

Starting with Q2

At the beginning of working with a model, it is advisable to start with Q2 just to make sure the pipeline works - that everything flows and the model runs. At this stage, output quality is not the focus; only whether the system is functioning.

5.1 Advanced Topic - QLoRA and NormalFloat (NF4)

A major research leap in quantization was introduced by QLoRA (Dettmers et al., NeurIPS 2023). The study showed that generic bit reduction, such as from FP16 to full 4-bit representation, can hurt accuracy because of the asymmetric distribution of weights.

The solution is a specialized data type, NF4 (4-bit NormalFloat), statistically adapted to the natural normal distribution of trained neural-network weights, enabling 4-bit compression with almost no loss in quality.

The study added two complementary mechanisms:

• Double Quantization, which compresses even the normalization coefficients.
• Paged Optimizers, which prevent memory crashes during momentary spikes.

This combination became an industry standard for running and training huge models on advanced home hardware.

Ollama: Local Execution

Ollama is like the "YouTube player of models": it knows how to read a GGUF file and the Modelfile, load the model, and run it.

Ollama exposes an OpenAI-compatible API, so it is possible to connect external tools to it - for example by pointing the API of a CLI client such as Claude CLI to the local Ollama server - and work locally on the computer with all the capabilities such as skills and sessions.

Basic Ollama Commands

``bash
Pull and run a model locally
ollama pull llama3
ollama run llama3 "What is the capital of Israel?"

List installed models and serve the local API
ollama list
ollama serve   # exposes an OpenAI-compatible API on localhost:11434

Point an OpenAI-compatible client to the local server
export OPENAIBASEURL="http://localhost:11434/v1"
export OPENAIAPIKEY="ollama"
`

Once you understand that the model runs locally, you can deploy it to a server, provide a URL, and allow other people to use it. The issue of model licensing must always be kept in mind.

Training and LoRA
7.1 Training vs Inference

In training, the weights are changed based on a loss function, gradient descent, and backpropagation. This is a heavy process requiring time and resources.

In inference, everything is based on what is fed into the context window.

Training requires not only weights but also gradients, activations, and an optimizer such as Adam. Therefore, VRAM requirements in training may be 3 to 5 times higher than in inference. This is an important nuance to know.

7.2 PEFT and Transfer Learning

Sometimes we want to take an existing model and adapt it - for example, a model that knows how to classify horses vs dogs, and adapt it to classify tables vs chairs.

Instead of training billions of parameters, PEFT (Parameter-Efficient Fine-Tuning) freezes most of the model and trains only small components. This approach preserves the base knowledge and dramatically reduces memory usage.

7.3 The Idea Behind LoRA

In LoRA (Low-Rank Adaptation), we take the weight matrix $$W0$$ - all the weights of the model, say 70 billion parameters - and freeze it completely, while adding two small low-dimensional matrices, $$A$$ and $$B$$.

We train only these small matrices in the relevant places:

==> picture intentionally omitted <==

This yields orders-of-magnitude savings in the number of trainable parameters.

A Real Example

A student wanted to train a Text-to-Image model in a black-and-white Japanese art style. Using LoRA and only about 900 images, he built an excellent model that worked very well.

The training was performed on a computer with a 16GB graphics card costing about 6,000 NIS, and took only about half an hour.

Training Skeleton for PEFT with LoRA

`python
from peft import LoraConfig, getpeftmodel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.frompretrained("meta-llama/Llama-3-8B")
config = LoraConfig(
    r=8,  # rank of A and B matrices
    loraalpha=16,
    targetmodules=["qproj", "vproj"],
    loradropout=0.05,
)

model = getpeftmodel(base, config)
model.printtrainableparameters()  # only A,B are trainable
W0 is frozen
`

7.4 QLoRA and OLoRA

QLoRA combines quantization with LoRA: not only savings in matrices, but also reducing the precision of the matrices themselves, which makes it possible to take standard models and run them with reasonable compute power.

In regular LoRA, initialization of the matrices $$A$$ and $$B$$ is random, so the beginning of training may be unstable.

OLoRA uses orthonormal initialization via QR decomposition. The result is smoother and more stable training.

7.5 Extension - The Mathematical Basis of LoRA and OLoRA

LoRA was introduced by Microsoft researchers (Hu et al., ICLR 2022). Given an original weight matrix $$W0$$ of size $$d \times k$$, the method freezes it and adds a parallel learning path through two small matrices: $$B$$ of size $$d \times r$$ and $$A$$ of size $$r \times k$$, where $$r \ll \min(d, k)$$.

The computation is:

$$W = W0 + BA$$

At the end of training, the matrices are multiplied and merged back into the weights, so there is no penalty in inference latency.

OLoRA (Buyukakyuz, 2024) improves the initialization stage. Instead of random values, a QR decomposition of the weight matrix is performed, $$W = QR$$, where $$Q$$ is orthogonal and $$R$$ is upper triangular, and the matrices are initialized from these spaces.

This orthonormal initialization ensures that learning begins in a well-conditioned subspace, leading to faster and more stable convergence. Since the QR decomposition is performed only once during initialization, its cost is amortized and becomes negligible over tens of thousands of training cycles.

AirLLM: Running Massive Models on CPU
8.1 The Idea

AirLLM is an open-source project intended mainly for research, which makes it possible to run heavy models on CPU (and also GPU). Instead of loading all weights into VRAM at once, AirLLM works layer by layer: it brings in one layer, computes, gets the result, brings in the next layer, and so on.

The idea is based on virtual memory and paging from operating systems, where a page corresponds here to a layer.

8.2 Reminder: Virtual Memory from Operating Systems

Virtual memory makes the computer "think" it has much more memory than it physically does - for example, 1TB instead of 32GB - while at any given moment only the relevant information is brought in.

Key principles:

• Pages and Page Table - virtual memory is divided into fixed-size pages; the page table translates between a virtual page and a physical frame in RAM.
• MMU - the unit that performs the translation in real time.
• Hit / Miss - if the page exists in RAM, this is a hit; otherwise it must be brought from disk and another page must be evicted.
• Memory hierarchy - the closer the memory is to the processor, the faster and more expensive it is: Registers -> Cache -> RAM -> NVMe/SSD.
• Principle of locality - there is a high probability that the next operation will be close to the current one; cache is based on this.
• mmap - allows a file to be treated as if it were a region in memory; the operating system loads pages only when they are accessed.

AirLLM implements a similar logic for large language models: not all information must be in RAM at the same time.

8.3 The Price: Latency vs Memory

The real bottleneck is I/O - the time needed to bring in and evict a page. Therefore, AirLLM enables operation with much smaller memory, but at the price of high latency: lower throughput and slower response times.

It is suitable for experimentation, research, and accessibility - less so for busy real-time systems. Without paying tokens and expensive compute power, one can do things "at the price of electricity" and waiting time.

8.4 Extension - How AirLLM Uses mmap

The practical implementation of AirLLM relies on mmap together with the flat SafeTensors format. When the engine reads a layer, the operating system creates only a logical mapping in virtual address space, without loading the bits immediately.

When the engine accesses a layer to compute its forward pass, a page fault occurs, and the kernel loads the required block into the page cache, without unnecessary copying into user space (zero-copy).

After computation, the GPU keeps in VRAM only the small hidden states, while the weights of the previous layer are automatically freed in favor of the next layer loaded using the same technique - exactly like the paging principle in operating systems.

8.5 Advanced Topic - Research Frontier: FlexGen, LLM in a Flash, and PagedAttention

The analogy to operating systems underlies additional leading studies, showing how much this topic is at the research frontier.

• FlexGen (ICML 2023) built a mathematical model based on Linear Programming for optimal scheduling of loading and eviction among GPU, memory, and disk, and made it possible to run a 175-billion-parameter model on a single 16GB home GPU.

• LLM in a Flash (Apple, 2024) optimized staged loading from disk to flash by predicting which neurons would become zero due to ReLU sparsity, and therefore read only the relevant memory columns.

• PagedAttention (SOSP 2023) - the basis of the vLLM library - applies the paging principle and page tables directly to the KV Cache in VRAM. This eliminated fragmentation, which had wasted up to 60% of memory, enabled page sharing across requests with a common prompt, and increased total throughput by 2x to 4x.

The Practical Assignment
9.1 What Needs to Be Done

The goal is to prove that AirLLM helps in certain cases. The assignment stages are:

Choose a task and choose a model on Hugging Face, adapted to the computer's specification.
Install Ollama and the model, and run a basic execution to verify that everything works.
Take a model that is too large to fit in RAM/GPU, and show that it fails or is extremely slow - this is the baseline.
Run the same model using AirLLM on CPU, and show that this time it does run - at the cost of latency.
Measure response time, memory consumption, and runtime, and compare between GPU, CPU, and AirLLM.

Tips and Recommendations
• Create a virtual environment (venv; preferably with uv`).
• Do not work with the newest Python version - many packages are still not adapted to it.
• Secure the Hugging Face token; do not keep it exposed.
• Start with a small, even irrelevant model, and set a low maximum token count, just to verify that the pipeline works.
• Allocate enough disk space before downloading large models.

9.2 Summary of the Journey

We went through a complete journey: from hardware (CPU/GPU), through CUDA/PTX/SASS, the stages of inference (Prefill/Decode) and VRAM, through choosing models in Hugging Face, licensing, formats (SafeTensors/GGUF), quantization, local execution with Ollama, efficient training with LoRA/QLoRA/OLoRA, and finally running huge models on CPU using AirLLM and virtual-memory principles.

All of these together expand the boundary of what can be run and trained locally.

