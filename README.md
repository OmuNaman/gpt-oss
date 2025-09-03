# GPT-OSS: The Missing Open-Source Training Code

<p align="center">
  <img alt="gpt-oss-20b" src="https://raw.githubusercontent.com/openai/gpt-oss/main/docs/gpt-oss-20b.svg" width="400">
</p>
<p align="center">
  <strong>A complete, open-source framework to train gpt-oss-style models from scratch.</strong>
</p>

<p align="center">
  <a href="https://github.com/OmuNaman/gpt-oss/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://huggingface.co/omunaman/Open_Source_GPT_OSS_20B"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo%20Model-orange" alt="Hugging Face"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/Built%20with-PyTorch-red" alt="PyTorch"></a>
</p>

---

## The Mission: Truly Open-Source AI

When OpenAI released its `gpt-oss` models, it provided the community with powerful **open-weights**. However, "open-weights" is not the same as open-source code. The crucial tools to replicate, understand, and build upon these models—the training and inference framework—were not included.

**This repository provides the missing piece.**

We have created a clean, high-performance, and fully open-source system that implements the `gpt-oss-20b` architecture. Our goal is to empower the community to train these models from the ground up, fostering true innovation and transparency.

This is not just a model; it's a complete toolkit.

## Core Features of this Framework

This codebase is not a toy. It's a production-grade framework for training multi-billion parameter models, built with best practices for scale and efficiency.

*   **🚀 High-Performance Distributed Training:** Built on PyTorch's **FSDP (Fully Sharded Data Parallel)** for training massive models that don't fit on a single GPU.
*   **🧠 Advanced Model Architecture:** A faithful implementation of `gpt-oss` features:
    *   **Mixture-of-Experts (MoE)** using efficient `einsum` operations.
    *   **Grouped-Query Attention (GQA)** for faster inference.
    *   **Sliding Window Attention** and **Attention Sinks** for long-context efficiency.
    *   **Rotary Position Embeddings (RoPE)** with YaRN-style scaling.
*   **💾 Memory-Efficient Initialization:** Uses `meta` device initialization to instantiate 20B+ parameter models on machines with limited CPU RAM.
*   **⚡️ Scalable Sharded Checkpointing:** Saves and resumes training for both model and optimizer states in a sharded format, avoiding memory bottlenecks on a single node.
*   **🌍 Hugging Face Integration:** Includes a simple script to convert native FSDP checkpoints into the standard `safetensors` format for easy sharing and use with the `transformers` library.

## Project Structure

The repository is organized for clarity and maintainability:

-   `prepare.py`: A utility to download and tokenize a dataset into a memory-mapped binary format for efficient loading.
-   `model.py`: The heart of the project. Contains the complete definition of the Transformer architecture, including all layers like MoE, GQA, etc.
-   `train.py`: The main script for launching a distributed training job using FSDP.
-   `sample.py`: A multi-GPU, FSDP-aware script for generating text from a trained checkpoint.
-   `export_to_safetensors.py`: The script to convert internal training checkpoints to a Hugging Face-compatible format.

---

## Getting Started: Train Your Own 20B Model

Follow these steps to train a `gpt-oss-20b` model from scratch.

### Step 1: Setup

First, clone the repository and install the required dependencies.

```bash
git clone https://github.com/OmuNaman/gpt-oss.git
cd gpt-oss
pip install -r requirements.txt # (Assuming you create a requirements.txt with torch, tiktoken, etc.)
```

### Step 2: Prepare the Dataset

We use the TinyStories dataset as an example. The `prepare.py` script will automatically download it from Hugging Face, tokenize it with the `o200k_harmony` tokenizer, and create `train.bin` and `val.bin` files in the specified directory.

```bash
python prepare.py --out_dir data/tinystories
```

### Step 3: Launch Training

The following command launches a distributed training run for the 20B model on 5 GPUs. It is the exact command used to train our proof-of-concept model.

```bash
torchrun --nproc_per_node=5 train.py \
    --model_size="20b" \
    --out_dir="out-20b-h200-stable" \
    --data_dir="data/tinystories" \
    --batch_size=1 \
    --grad_accum_steps=8 \
    --block_size=512 \
    --max_iters=5000 \
    --lr=3e-4 \
    --min_lr=3e-5 \
    --warmup_iters=100 \
    --lr_decay_iters=5000 \
    --weight_decay=0.1 \
    --beta1=0.9 \
    --beta2=0.95 \
    --dtype="bfloat16" \
    --log_interval=10 \
    --eval_interval=100 \
    --save_every=500 \
    --sample_every=100
```
**Note:** The `bfloat16` dtype is highly recommended for modern GPUs (NVIDIA Ampere/Hopper). For older GPUs, you may need to use `float16`.

---

## Using Your Trained Model

Once training is running, you'll have checkpoints in your `--out_dir`. Here’s how to use them.

### Running Inference from Checkpoints

Use the `sample.py` script to generate text. This script correctly handles the FSDP sharded checkpoint format and runs inference in a distributed, deadlock-free manner.

```bash
torchrun --nproc_per_node=5 sample.py \
    --out_dir out-20b-h200-stable \
    --ckpt_prefix ckpt \
    --prompt "Once upon a time there was a " \
    --max_new_tokens 200 \
    --temperature 0.8 \
    --top_k 200 \
    --dtype bfloat16
```

### Exporting to Hugging Face `safetensors`

To share your model with the world, convert the sharded FSDP checkpoints into the standard `safetensors` format.

This script gathers the full model weights onto rank 0's CPU memory and re-shards them into files of a maximum size (e.g., 5GB), creating the necessary `index.json` file for `transformers`.

```bash
torchrun --nproc_per_node=5 export_to_safetensors.py \
  --in_dir out-20b-h200-stable \
  --ckpt_prefix ckpt \
  --max_shard_size 5GB \
  --release_dir /workspace/20b-release
```
The resulting files in `/workspace/20b-release` can then be uploaded directly to the Hugging Face Hub.

## Our Proof-of-Concept Model

To demonstrate that our codebase works, we trained a model with the commands above and have shared it on the Hugging Face Hub.

**➡️ [omunaman/Open_Source_GPT_OSS_20B](https://huggingface.co/omunaman/Open_Source_GPT_OSS_20B)**

This model is a checkpoint from a very early stage of training (**only 1900 iterations**). Its primary purpose is to serve as a tangible validation of this open-source code.

## Roadmap & Contributing

This project is just the beginning. We welcome contributions from the community! Our current roadmap includes:
- [ ] Training a model on a larger, more diverse dataset.
- [ ] Adding support for more quantization techniques (e.g., GGUF, AWQ).
- [ ] Writing detailed technical blog posts explaining the framework.
- [ ] Improving documentation and adding more examples.

Feel free to open an issue or submit a pull request!

## License
This project is licensed under the **Apache 2.0 License**. See the [LICENSE](LICENSE) file for details.

## Citation
If you use this codebase in your research or work, please consider citing our repository:

```bibtex
@software{Vizuara_GPT-OSS_Replication_2025,
  author = {Naman and Dr. Raj Dandekar,
  title = {{An Open-Source Implementation of gpt-oss-20b}},
  month = {September},
  year = {2025},
  url = {https://github.com/OmuNaman/gpt-oss}
}
```
