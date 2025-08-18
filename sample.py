"""
sample.py — simple inference script (no CLI).
Edit the constants below and run:  python sample.py
"""

import json
import os
from contextlib import nullcontext

import torch
import torch.nn.functional as F

try:
    import tiktoken
except ImportError as e:
    raise SystemExit("Please `pip install tiktoken` first.") from e

from model import Transformer, ModelConfig

# ----------------- EDIT THESE -----------------
OUT_DIR = "out"
DATA_DIR = "data/tinystories"   # for fallback meta.json
PROMPT = "Once upon a time,"
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.8
TOP_K = 200
DTYPE = "bfloat16"              # "float32" | "bfloat16" | "float16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK_SIZE = 1024               # truncate context during generation
# ---------------------------------------------


def load_tokenizer_from_meta(data_dir: str):
    meta_path = os.path.join(data_dir, "meta.json")
    tok_name = "o200k_harmony"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        tok_name = meta.get("tokenizer", tok_name)
    try:
        enc = tiktoken.get_encoding(tok_name)
    except Exception:
        enc = tiktoken.get_encoding("o200k_base")
        tok_name = "o200k_base"
        print(f"[sample] WARNING: tokenizer '{tok_name}' unavailable; using 'o200k_base'.")
    vocab_size = getattr(enc, "n_vocab", 201_088)
    return enc, tok_name, vocab_size


@torch.no_grad()
def generate_tokens(model, input_ids, max_new_tokens=64, temperature=1.0, top_k=None,
                    eos_token_id=None, device="cuda", block_size=1024):
    model.eval()
    tokens = input_ids.to(device)
    for _ in range(max_new_tokens):
        inp = tokens[:, -block_size:]
        logits, _ = model(inp, labels=None)
        next_logits = logits[:, -1, :]
        if temperature != 1.0:
            next_logits = next_logits / max(1e-6, temperature)
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(next_logits, top_k)
            next_logits[next_logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
        if eos_token_id is not None and (next_token.squeeze(-1) == eos_token_id).all():
            break
    return tokens


def main():
    ckpt_path = os.path.join(OUT_DIR, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}. Train first (see train.py).")

    # If you trust your own checkpoint, set weights_only=False to avoid PyTorch 2.6 pickle guard.
    # This also handles older checkpoints that stored a ModelConfig object.
    payload = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    # Tokenizer preference: checkpoint → meta.json
    saved_tok = payload.get("tokenizer", None)
    if saved_tok is not None:
        try:
            enc = tiktoken.get_encoding(saved_tok)
            tok_name = saved_tok
        except Exception:
            enc, tok_name, _ = load_tokenizer_from_meta(DATA_DIR)
    else:
        enc, tok_name, _ = load_tokenizer_from_meta(DATA_DIR)

    # Recover config (support dict or object)
    cfg = None
    if "model_config_dict" in payload:
        cfg = ModelConfig(**payload["model_config_dict"])
    elif "model_config" in payload and isinstance(payload["model_config"], ModelConfig):
        cfg = payload["model_config"]
    else:
        # fallback to tokenizer vocab
        cfg = ModelConfig(vocab_size=getattr(enc, "n_vocab", 201_088))

    # Make sure vocab matches tokenizer
    cfg.vocab_size = getattr(enc, "n_vocab", cfg.vocab_size)

    # AMP context
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    amp_dtype = dtype_map[DTYPE]
    ctx = nullcontext() if DEVICE == "cpu" else torch.amp.autocast("cuda", dtype=amp_dtype)

    # Build & load model
    model = Transformer(cfg).to(DEVICE)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    # Encode prompt
    start_ids = enc.encode(PROMPT, allowed_special=set()) or [enc.encode("\n")[0]]
    x = torch.tensor([start_ids], dtype=torch.long, device=DEVICE)

    with torch.no_grad(), ctx:
        out_ids = generate_tokens(
            model,
            x,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            eos_token_id=getattr(model.config, "eos_token_id", None),
            device=DEVICE,
            block_size=BLOCK_SIZE,
        )

    print("\n" + "="*80)
    print(enc.decode(out_ids[0].tolist()))
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
