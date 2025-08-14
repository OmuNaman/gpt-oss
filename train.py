"""
A from-scratch training script for the GPT-OSS model.

This script is designed in the spirit of nanoGPT: a single, clear file
that handles all aspects of training, from data loading to checkpointing
and logging, while being easy to modify.

It supports training models of different sizes and can be pointed to any
pre-tokenized dataset.

Key Features:
-   Configurable model sizes (e.g., a 155M test model).
-   Efficient data loading using memory-mapped numpy arrays.
-   Seamless checkpointing and resumption of training.
-   Periodic evaluation on a validation set.
-   Live text generation to monitor model progress.
-   Standard training utilities: AdamW, gradient accumulation, cosine LR decay.

To run on a pre-tokenized dataset (like TinyStories):
1.  Prepare the data:
    python data/tinystories/prepare.py

2.  Run the training script:
    python train.py --model_size=gpt2-124m --data_dir=data/tinystories
"""
import os
import time
import math
from contextlib import nullcontext

import numpy as np
import torch
import tiktoken

# (Make sure model.py is in the same directory)
from model import ModelConfig, GptOssModel, calculate_params

# --- Configuration -----------------------------------------------------------------

# Training hyperparameters
model_size = 'gpt2-124m' # or 'gpt-oss-120b'
data_dir = 'data/tinystories'
out_dir = 'out'
# ---
eval_interval = 250       # How often to run validation
log_interval = 10         # How often to print training loss
save_interval = 1000      # How often to save a checkpoint
eval_iters = 200          # Number of batches for validation loss estimation
always_save_checkpoint = True # if True, always save a checkpoint at the end of eval

# Dataloader
batch_size = 12
block_size = 256 # Context length

# AdamW optimizer
learning_rate = 6e-4
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate decay settings
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5

# System
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False # PyTorch 2.0 compilation (use for speedup)

# -----------------------------------------------------------------------------
config = {}
# exec(open('configurator.py').read()) # Overrides from command line
# -----------------------------------------------------------------------------

# Create output directory
os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337)

# --- Data Loading ------------------------------------------------------------
def get_batch(split):
    # Use memory-mapped files for efficient access
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# --- Model Configurations ----------------------------------------------------
def get_config(size: str) -> ModelConfig:
    """
    Returns a ModelConfig instance for a specified model size.
    """
    if size == 'gpt2-124m':
        # This is a config for a 124M parameter model, very close to GPT-2 small.
        # It's a non-MoE, non-GQA model for simple and fast testing.
        return ModelConfig(
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=12,   # No GQA, so n_kv_head == n_head
            hidden_size=768,
            head_dim=64,              # hidden_size / num_attention_heads = 768 / 12 = 64
            intermediate_size=3072,   # 4 * hidden_size, standard for GPT-2
            num_local_experts=1,      # A "1-expert MoE" is just a standard dense FFN
            experts_per_token=1,
            vocab_size=50257,         # Standard GPT-2 vocab size
            max_position_embeddings=1024, # Standard GPT-2 context length
            block_size=1024,          # Use the full context length for this model
            # For a non-SWA model, all layers are full attention
            layer_types=['full_attention'] * 12,
        )
    elif size == 'gpt2-155m':
        # This is a slightly larger GPT-2 Small-ish config (kept for reference)
        return ModelConfig(
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=12,
            hidden_size=768,
            head_dim=64,
            intermediate_size=3072,
            num_local_experts=1,
            experts_per_token=1,
            vocab_size=50257,
            max_position_embeddings=1024,
            sliding_window=1024,
            layer_types=['full_attention'] * 12,
            block_size=1024,
        )
    elif size == 'gpt-oss-120b':
        # The full model config from model.py
        # WARNING: Do not attempt to train this without a large cluster.
        return ModelConfig()
    else:
        raise ValueError(f"Unknown model size: {size}")

# --- Main Training Script ----------------------------------------------------
if __name__ == "__main__":
    # --- Initialization ---
    
    # Get model config
    model_config = get_config(model_size)
    print(f"--- Training model: {model_size} ---")
    
    # Calculate and print parameter count
    param_stats = calculate_params(model_config)
    print(f"Total Parameters: {param_stats['total_B']:.2f}B")
    print(f"Active Parameters: {param_stats['active_B']:.2f}B")
    print("-" * 30)

    # Gradient accumulation
    gradient_accumulation_steps = 8
    
    # Initialize model
    model = GptOssModel(model_config)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)

    # Check for a checkpoint to resume from
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    iter_num = 0
    best_val_loss = 1e9
    if os.path.exists(ckpt_path):
        print("Resuming training from checkpoint.")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']

    # Learning rate scheduler
    def get_lr(it):
        if not decay_lr:
            return learning_rate
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)
        
    # Mixed precision training
    pt_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=pt_dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # For decoding generated text
    enc = tiktoken.get_encoding("gpt2")
    
    # --- Training Loop ---
    X, Y = get_batch('train') # Fetch first batch
    t0 = time.time()
    
    while iter_num < max_iters:
        # Update learning rate
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # --- Evaluation (Periodic) ---
        if iter_num % eval_interval == 0:
            model.eval()
            print("\n--- Running Validation ---")
            
            # Estimate validation loss
            with torch.no_grad():
                losses = torch.zeros(eval_iters)
                for k in range(eval_iters):
                    X_val, Y_val = get_batch('val')
                    with ctx:
                        logits, outputs = model(X_val, labels=Y_val)
                    losses[k] = outputs['loss'].item()
                val_loss = losses.mean()
            
            print(f"iter {iter_num}: val loss {val_loss:.4f}")

            # --- Generate Sample Text ---
            print("--- Generating Sample Text ---")
            with torch.no_grad():
                with ctx:
                    start_ids = torch.zeros((1, 1), dtype=torch.long, device=device) # Start with token 0 (e.g., <|endoftext|>)
                    generated_tokens = model.generate(start_ids, max_new_tokens=100, temperature=0.8, top_k=200)
                    decoded_text = enc.decode(generated_tokens[0].tolist())
                    print(decoded_text)
            print("-" * 30 + "\n")

            # Save checkpoint if it's the best so far
            if val_loss < best_val_loss or always_save_checkpoint:
                best_val_loss = val_loss
                if iter_num > 0:
                    print(f"Saving checkpoint to {out_dir}")
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'model_config': model_config,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                    }
                    torch.save(checkpoint, ckpt_path)
            model.train()

        # --- Training Step ---
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits, outputs = model(X, labels=Y)
                loss = outputs['loss'] / gradient_accumulation_steps
            
            # Fetch next batch right away to overlap compute and data transfer
            X, Y = get_batch('train')
            scaler.scale(loss).backward()
            
        # Clip gradients
        if grad_clip > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Log training progress
        if iter_num % log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            lossf = loss.item() * gradient_accumulation_steps # last micro-step's loss
            print(f"iter {iter_num}: train loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.6f}")

        iter_num += 1

    print("--- Training Finished ---")