"""
Generates text from a trained GPT-OSS model.

This script loads a model checkpoint and its configuration, then uses it to
generate text based on a starting prompt. It's designed to be simple and
showcase the model's capabilities.

Example Usage:
1.  Generate from scratch (unconditional):
    python sample.py

2.  Provide a starting prompt:
    python sample.py --start="Once upon a time"

3.  Generate more text and more samples:
    python sample.py --num_samples=5 --max_new_tokens=200
"""
import os
import torch
import tiktoken
from contextlib import nullcontext

# Make sure model.py is in the same directory
from model import Transformer, ModelConfig

# --- Configuration -----------------------------------------------------------------
out_dir = 'out'           # Checkpoint directory
start = "\n"              # Or "<|endoftext|>" or "Hello, world!" etc.
num_samples = 3           # Number of samples to generate
max_new_tokens = 256      # Number of tokens to generate in each sample
temperature = 0.8         # 1.0 = no change, < 1.0 = less random, > 1.0 = more random
top_k = 200               # Retain only the top_k most likely tokens, clamp others to -inf
seed = 1337

# System
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False # Use PyTorch 2.0 compilation for potential speedup

# -----------------------------------------------------------------------------
# exec(open('configurator.py').read()) # Overrides from command line
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed) # if you are using GPU

# --- Main Sampling Script ----------------------------------------------------
if __name__ == "__main__":
    # --- 1. Load Checkpoint and Configuration ---
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    print(f"Loading checkpoint from: {ckpt_path}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{ckpt_path}'. "
            "Ensure you have run train.py to create a checkpoint."
        )

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # It's crucial to load the config from the checkpoint to ensure the model
    # architecture matches the trained weights.
    model_config = checkpoint['model_config']
    print("Model configuration loaded from checkpoint.")

    # --- 2. Build the Model ---
    model = Transformer(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set the model to evaluation mode
    model.to(device)

    if compile:
        print("Compiling the model... (requires PyTorch 2.0)")
        model = torch.compile(model)

    # --- 3. Initialize Tokenizer ---
    # The tokenizer must match the one used during training.
    enc = tiktoken.get_encoding("o200k_base")
    
    # Encode the starting prompt
    start_ids = enc.encode(start, allowed_special={'<|endoftext|>'})
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # --- 4. Generate Text ---
    print(f"Starting generation with prompt:\n'{start}'")
    
    # Setup mixed precision context
    pt_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=pt_dtype)
    
    # Run generation
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                print("\n" + "="*20 + f" SAMPLE {k+1} " + "="*20)
                generated_tokens = model.generate(
                    x,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k
                )
                decoded_text = enc.decode(generated_tokens[0].tolist())
                print(decoded_text)
    
    print("\n" + "="*50)