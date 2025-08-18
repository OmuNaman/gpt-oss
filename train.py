#!/usr/bin/env python3
"""
train.py — Training harness for GPT-OSS (toy or 20B config).

- Loads NanoGPT-style memmaps created by prepare.py (train.bin / val.bin / meta.json)
- Uses tokenizer from meta.json (o200k_harmony preferred; falls back to o200k_base)
- Saves a checkpoint every N iters (default 100)
- Prints a short sample every N iters (default 100)
- Evaluates periodically and saves on best val
- Resumes from out/ckpt.pt if available
- Uses modern AMP APIs (torch.amp.*)
"""
import argparse
import json
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

try:
    import tiktoken
except ImportError as e:
    raise SystemExit("Please `pip install tiktoken` first.") from e

# Your model must be in the same folder
from model import Transformer, ModelConfig, gpt_oss_20b_config


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/tinystories")
    ap.add_argument("--out_dir", type=str, default="out")
    ap.add_argument("--model_size", type=str, choices=["toy", "20b"], default="20b")

    # Training hyperparams
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--block_size", type=int, default=512)    # train-time context (≤ model max)
    ap.add_argument("--max_iters", type=int, default=2000)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--eval_interval", type=int, default=500)
    ap.add_argument("--eval_iters", type=int, default=200)

    # Periodic save + sample
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--sample_every", type=int, default=100)
    ap.add_argument("--sample_tokens", type=int, default=120)
    ap.add_argument("--top_k", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)

    # Optim & schedule
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--decay_lr", action="store_true", default=True)
    ap.add_argument("--warmup_iters", type=int, default=2000)
    ap.add_argument("--lr_decay_iters", type=int, default=600000)
    ap.add_argument("--min_lr", type=float, default=6e-5)

    # System
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dtype", type=str, choices=["float32", "bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--compile", action="store_true", default=False)
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# Data loader
# --------------------------------------------------------------------------- #

class BinLoader:
    def __init__(self, data_dir: str, split: str, block_size: int, batch_size: int, device: str):
        path = os.path.join(data_dir, f"{split}.bin")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Did you run prepare.py?")
        self.data = np.memmap(path, dtype=np.uint32, mode="r")
        self.block = block_size
        self.bs = batch_size
        self.device = device

    def get_batch(self):
        # Random chunked batches
        ix = np.random.randint(0, len(self.data) - self.block - 1, size=(self.bs,))
        X = np.stack([self.data[i:i+self.block].astype(np.int64) for i in ix])
        Y = np.stack([self.data[i+1:i+self.block+1].astype(np.int64) for i in ix])
        X = torch.from_numpy(X).to(self.device)
        Y = torch.from_numpy(Y).to(self.device)
        return X, Y


# --------------------------------------------------------------------------- #
# Tokenizer helpers
# --------------------------------------------------------------------------- #

def load_tokenizer(meta_path: str):
    with open(meta_path, "r") as f:
        meta = json.load(f)
    tok_name = meta.get("tokenizer", "o200k_harmony")
    try:
        enc = tiktoken.get_encoding(tok_name)
    except Exception:
        enc = tiktoken.get_encoding("o200k_base")
        tok_name = "o200k_base"
        print(f"[train] WARNING: tokenizer '{meta.get('tokenizer')}' not available. Using 'o200k_base'.")
    vocab_size = int(meta.get("vocab_size", getattr(enc, "n_vocab", 201_088)))
    return enc, tok_name, vocab_size


# --------------------------------------------------------------------------- #
# Model configs
# --------------------------------------------------------------------------- #

def build_config(name: str) -> ModelConfig:
    if name == "20b":
        return gpt_oss_20b_config()
    # Tiny “toy” config: same motifs (GQA, MoE, SWA) but small dims
    return ModelConfig(
        vocab_size=200_019,      # overwritten by meta.json later
        hidden_size=256,
        num_hidden_layers=8,
        head_dim=32,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        sliding_window=64,
        num_local_experts=4,
        experts_per_token=2,
        intermediate_size=256,
        rope_theta=150_000.0,
        enable_sink_logit=False,
        tie_word_embeddings=False,
    )


# --------------------------------------------------------------------------- #
# Generation helper (no KV cache; keeps it simple)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def generate_tokens(model, input_ids, max_new_tokens=64, temperature=1.0, top_k=None,
                    eos_token_id=None, device="cuda", block_size=512):
    model.eval()
    tokens = input_ids.to(device)
    for _ in range(max_new_tokens):
        # Truncate context to training block for speed/mem
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


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # tokenizer & vocab
    meta_path = os.path.join(args.data_dir, "meta.json")
    enc, tok_name, vocab_size = load_tokenizer(meta_path)

    # model config
    cfg = build_config(args.model_size)
    cfg.vocab_size = vocab_size
    if args.block_size > cfg.max_position_embeddings:
        print(f"[train] Reducing block_size from {args.block_size} to model max {cfg.max_position_embeddings}")
        args.block_size = cfg.max_position_embeddings

    # model
    model = Transformer(cfg).to(device)
    if args.compile and device == "cuda":
        print("[train] torch.compile() on…")
        model = torch.compile(model)

    # params summary
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 60)
    print(f"Model: {args.model_size}   device: {device}")
    print(f"Tokenizer: {tok_name}  vocab: {vocab_size}")
    print(f"Params: total {total/1e6:.1f}M  trainable {trainable/1e6:.1f}M")
    print(f"Context: train block {args.block_size} (model max {cfg.max_position_embeddings})")
    print("=" * 60)

    # data
    train_loader = BinLoader(args.data_dir, "train", args.block_size, args.batch_size, device)
    val_loader   = BinLoader(args.data_dir, "val",   args.block_size, args.batch_size, device)

    # AMP
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    amp_dtype = dtype_map[args.dtype]
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast("cuda", dtype=amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and args.dtype == "float16"))

    # optim
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

    # resume
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    iter_num = 0
    best_val = float("inf")
    if os.path.exists(ckpt_path):
        print(f"[train] Resuming from {ckpt_path}")
        payload = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        opt.load_state_dict(payload["optimizer_state_dict"])
        iter_num = int(payload.get("iter_num", 0))
        best_val = float(payload.get("best_val_loss", best_val))
        # Update cfg if you want to hard-override from checkpoint:
        # cfg = payload.get("model_config", cfg)

    # lr schedule
    def get_lr(it: int) -> float:
        if not args.decay_lr:
            return args.lr
        if it < args.warmup_iters:
            return args.lr * it / max(1, args.warmup_iters)
        if it > args.lr_decay_iters:
            return args.min_lr
        decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return args.min_lr + coeff * (args.lr - args.min_lr)

    # small eval helper
    def evaluate() -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(args.eval_iters):
                Xv, Yv = val_loader.get_batch()
                with ctx:
                    _, out = model(Xv, labels=Yv)
                losses.append(out["loss"].item())
        model.train()
        return sum(losses) / len(losses)

    # sampling helper (uses the generation loop above)
    def sample(n_tokens: int):
        model.eval()
        with torch.no_grad():
            start_tok = enc.encode("\n")[0]
            start_ids = torch.tensor([[start_tok]], device=device, dtype=torch.long)
            out_ids = generate_tokens(model, start_ids,
                                      max_new_tokens=n_tokens,
                                      temperature=args.temperature,
                                      top_k=args.top_k,
                                      eos_token_id=getattr(model.config, "eos_token_id", None),
                                      device=device,
                                      block_size=args.block_size)
            print("\n--- SAMPLE ---")
            print(enc.decode(out_ids[0].tolist()))
            print("--------------\n")
        model.train()

    # training loop
    t0 = time.time()
    X, Y = train_loader.get_batch()
    while iter_num < args.max_iters:
        # set lr
        lr = get_lr(iter_num)
        for g in opt.param_groups:
            g["lr"] = lr

        # fw/bw
        with ctx:
            _, out = model(X, labels=Y)
            loss = out["loss"]

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
        opt.zero_grad(set_to_none=True)

        # prefetch next batch
        X, Y = train_loader.get_batch()

        # logs
        if iter_num % args.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(f"iter {iter_num:06d}  loss {loss.item():.4f}  lr {lr:.6e}  {dt*1000:.1f} ms/it")

        # periodic eval (+ save on best)
        if args.eval_interval > 0 and iter_num > 0 and iter_num % args.eval_interval == 0:
            val = evaluate()
            print(f"[eval] iter {iter_num}  val_loss {val:.4f}")
            if val < best_val:
                best_val = val
                print(f"[ckpt] new best ({best_val:.4f}) → saving {ckpt_path}")
                payload = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "model_config": cfg,
                    "iter_num": iter_num,
                    "best_val_loss": best_val,
                    "tokenizer": tok_name,
                }
                torch.save(payload, ckpt_path)

        # periodic sample
        if args.sample_every > 0 and iter_num % args.sample_every == 0:
            sample(args.sample_tokens)

        # periodic save
        if args.save_every > 0 and iter_num > 0 and iter_num % args.save_every == 0:
            print(f"[ckpt] saving (periodic) at iter {iter_num} → {ckpt_path}")
            payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "model_config": cfg,
                "iter_num": iter_num,
                "best_val_loss": best_val,
                "tokenizer": tok_name,
            }
            torch.save(payload, ckpt_path)

        iter_num += 1

    print("[train] done.")


if __name__ == "__main__":
    main()