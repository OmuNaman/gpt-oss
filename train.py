#!/usr/bin/env python3
"""
train.py — FSDP (ZeRO-3) trainer for GPT-OSS (toy or 20B).

- Loads memmaps from prepare.py (train.bin / val.bin / meta.json)
- Proper multi-GPU via torch.distributed + FSDP
- Rank-0 logging, eval, sampling, checkpointing (full state on CPU)
- Resumes cleanly from out/ckpt.pt
- Works on single GPU too (FSDP still ok with world_size=1)

Launch (single node, 8 GPUs):
  torchrun --standalone --nproc_per_node=8 train.py \
    --data_dir data/tinystories --out_dir out_toy --model_size toy \
    --batch_size 4 --grad_accum_steps 4 --block_size 512 --dtype bfloat16
"""
import argparse
import dataclasses
import json
import math
import os
import time
from contextlib import nullcontext
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

try:
    import tiktoken
except ImportError as e:
    raise SystemExit("Please `pip install tiktoken` first.") from e

# FSDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    StateDictType,
    FullStateDictConfig,
    FullOptimStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision

# Your model (must define Transformer, ModelConfig, gpt_oss_20b_config, TransformerBlock)
from model import Transformer, ModelConfig, gpt_oss_20b_config, TransformerBlock


# ============================== ARGS =========================================

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/tinystories")
    ap.add_argument("--out_dir", type=str, default="out")
    ap.add_argument("--model_size", type=str, choices=["toy", "20b"], default="toy")

    # training
    ap.add_argument("--batch_size", type=int, default=2, help="per-rank microbatch")
    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--max_iters", type=int, default=2000)
    ap.add_argument("--grad_accum_steps", type=int, default=8)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--eval_interval", type=int, default=500)
    ap.add_argument("--eval_iters", type=int, default=100)

    # periodic save + sample
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--sample_every", type=int, default=100)
    ap.add_argument("--sample_tokens", type=int, default=120)
    ap.add_argument("--top_k", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)

    # optim & schedule
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--decay_lr", action="store_true", default=True)
    ap.add_argument("--warmup_iters", type=int, default=2000)
    ap.add_argument("--lr_decay_iters", type=int, default=600000)
    ap.add_argument("--min_lr", type=float, default=6e-5)

    # system
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dtype", type=str, choices=["float32", "bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--compile", action="store_true", default=False,
                    help="not recommended with FSDP; ignored")
    return ap.parse_args()


# =========================== HELPERS =========================================

def is_dist() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1

def rank0_print(*args, **kwargs):
    if not is_dist() or dist.get_rank() == 0:
        print(*args, **kwargs)

class BinLoader:
    def __init__(self, data_dir: str, split: str, block_size: int, batch_size: int, device: str, seed: int):
        path = os.path.join(data_dir, f"{split}.bin")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Did you run prepare.py?")
        self.data = np.memmap(path, dtype=np.uint32, mode="r")
        self.block = block_size
        self.bs = batch_size
        self.device = device
        self.rng = np.random.RandomState(seed)

    def get_batch(self):
        N = len(self.data)
        ixs = self.rng.randint(0, N - self.block - 1, size=(self.bs,))
        X = np.stack([self.data[i:i+self.block].astype(np.int64) for i in ixs])
        Y = np.stack([self.data[i+1:i+self.block+1].astype(np.int64) for i in ixs])
        return torch.from_numpy(X).to(self.device), torch.from_numpy(Y).to(self.device)


def load_tokenizer(meta_path: str):
    with open(meta_path, "r") as f:
        meta = json.load(f)
    tok_name = meta.get("tokenizer", "o200k_harmony")
    try:
        enc = tiktoken.get_encoding(tok_name)
    except Exception:
        enc = tiktoken.get_encoding("o200k_base")
        tok_name = "o200k_base"
        rank0_print(f"[train] WARNING: tokenizer '{meta.get('tokenizer')}' not available. Using 'o200k_base'.")
    vocab_size = int(meta.get("vocab_size", getattr(enc, "n_vocab", 201_088)))
    return enc, tok_name, vocab_size


def build_config(name: str) -> ModelConfig:
    if name == "20b":
        return gpt_oss_20b_config()
    # tiny toy config (keeps GQA + MoE shape)
    return ModelConfig(
        vocab_size=200_019,      # overwritten by meta.json
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


@torch.no_grad()
def sample_text(model, enc, device, n_tokens=120, temperature=0.8, top_k=200,
                block_size=512, amp_dtype=torch.bfloat16):
    model_was_training = model.training
    model.eval()

    start_tok = enc.encode("\n")[0]
    tokens = torch.tensor([[start_tok]], device=device, dtype=torch.long)

    # use the same autocast dtype as training (bf16)
    ctx = (nullcontext() if "cpu" in str(device)
           else torch.amp.autocast("cuda", dtype=amp_dtype))

    with ctx:
        for _ in range(n_tokens):
            inp = tokens[:, -block_size:]
            logits, _ = model(inp, labels=None)
            next_logits = logits[:, -1, :]

            if temperature != 1.0:
                next_logits = next_logits / max(1e-6, temperature)
            if top_k and top_k > 0:
                v, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < v[:, [-1]]] = -float("inf")

            probs = torch.softmax(next_logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, nxt], dim=1)

    if model_was_training:
        model.train()

    return enc.decode(tokens[0].tolist())


# ============================ MAIN ==========================================

def main():
    # allocation hint to reduce fragmentation on long runs
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    args = get_args()

    # --- distributed init & device ---
    if is_dist():
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rank = 0
        world_size = 1

    # sanity note for 20B
    if args.model_size == "20b" and world_size < 8:
        rank0_print("WARNING: 20B training needs many GPUs with FSDP/ZeRO-3 "
                    "(e.g., 8×A100-80GB or 4×H200). Toy works anywhere; "
                    "20B on 1 GPU will OOM.")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # tokenizer & vocab
    meta_path = os.path.join(args.data_dir, "meta.json")
    enc, tok_name, vocab_size = load_tokenizer(meta_path)

    # model config
    cfg = build_config(args.model_size)
    cfg.vocab_size = vocab_size
    if args.block_size > cfg.max_position_embeddings:
        rank0_print(f"[train] Reducing block_size from {args.block_size} to model max {cfg.max_position_embeddings}")
        args.block_size = cfg.max_position_embeddings

    # base model
    base_model = Transformer(cfg).to(device)

    # FSDP wrap with correct policy (this is the FIX)
    auto_wrap = partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})
    mp_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    mp = MixedPrecision(param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype)

    model = FSDP(
        base_model,
        auto_wrap_policy=auto_wrap,
        device_id=torch.device(device),
        mixed_precision=mp,
        use_orig_params=True,  # ZeRO-3-style param management
    )

    if args.compile and "cuda" in device:
        rank0_print("[train] Skipping torch.compile with FSDP (not reliably supported).")

    # params (rank0)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print("=" * 60)
    rank0_print(f"Model: {args.model_size}   device: {device}   world_size: {world_size}")
    rank0_print(f"Tokenizer: {tok_name}  vocab: {vocab_size}")
    rank0_print(f"Params: total {total/1e6:.1f}M  trainable {trainable/1e6:.1f}M")
    rank0_print(f"Context: train block {args.block_size} (model max {cfg.max_position_embeddings})")
    rank0_print("=" * 60)

    # data (rank-distinct RNG)
    train_loader = BinLoader(args.data_dir, "train", args.block_size, args.batch_size, device, seed=args.seed+rank)
    val_loader   = BinLoader(args.data_dir, "val",   args.block_size, args.batch_size, device, seed=args.seed+1234+rank)

    # AMP
    ctx = nullcontext() if "cpu" in device else torch.amp.autocast("cuda", dtype=mp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=("cuda" in device and args.dtype == "float16"))

    # optim
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

    # FSDP wants explicit state dict type
    FSDP.set_state_dict_type(model, StateDictType.FULL_STATE_DICT)

    # resume
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    iter_num = 0
    best_val = float("inf")
    if os.path.exists(ckpt_path):
        if rank == 0:
            rank0_print(f"[train] Resuming from {ckpt_path}")
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if is_dist():
            dist.barrier()
        # broadcast-aware resume (simple path: let every rank read payload from disk on shared FS)
        if rank != 0:
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # load model full → sharded
        model.load_state_dict(payload["model_state_dict"])
        # optimizer: map full → sharded
        shard_optim = FSDP.optim_state_dict_to_load(model, opt, payload["optimizer_state_dict"])
        opt.load_state_dict(shard_optim)
        iter_num = int(payload.get("iter_num", 0))
        best_val = float(payload.get("best_val_loss", best_val))
        if is_dist():
            dist.barrier()
        rank0_print(f"[train] Resumed at iter {iter_num} (best val {best_val:.4f})")

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

    # eval
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
        val = torch.tensor([sum(losses) / max(1, len(losses))], device=device)
        if is_dist():
            dist.all_reduce(val, op=dist.ReduceOp.AVG)
        return float(val.item())

    # save (rank0 full, CPU)
    def save_ckpt(path, iter_num, best_val):
        if dist.is_initialized() and dist.get_rank() != 0:
            if is_dist():
                dist.barrier()
            return
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            full_model_sd = model.state_dict()
            full_optim_sd = FSDP.optim_state_dict(model, opt)
        payload = {
            "model_state_dict": full_model_sd,
            "optimizer_state_dict": full_optim_sd,
            "model_config_dict": dataclasses.asdict(model.module.config if hasattr(model, "module") else model.config),
            "iter_num": iter_num,
            "best_val_loss": best_val,
            "tokenizer": tok_name,
        }
        torch.save(payload, path)
        rank0_print(f"[ckpt] saved {path}")
        if is_dist():
            dist.barrier()

    # training loop
    t0 = time.time()
    X, Y = train_loader.get_batch()
    while iter_num < args.max_iters:
        lr = get_lr(iter_num)
        for g in opt.param_groups:
            g["lr"] = lr

        # gradient accumulation
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        for micro in range(args.grad_accum_steps):
            with ctx:
                _, out = model(X, labels=Y)
                loss = out["loss"] / args.grad_accum_steps
            total_loss += loss.detach().item()
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            # next micro
            X, Y = train_loader.get_batch()

        if args.grad_clip > 0:
            if scaler.is_enabled():
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        if scaler.is_enabled():
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        # logs
        if iter_num % args.log_interval == 0:
            loss_t = torch.tensor([total_loss], device=device)
            if is_dist():
                dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
            dt = time.time() - t0
            t0 = time.time()
            rank0_print(f"iter {iter_num:06d}  loss {loss_t.item():.4f}  lr {lr:.6e}  {dt*1000:.1f} ms/it "
                        f"(G: {world_size*args.batch_size*args.grad_accum_steps})")

        # eval + save-on-best
        if args.eval_interval > 0 and iter_num > 0 and (iter_num % args.eval_interval == 0):
            val = evaluate()
            rank0_print(f"[eval] iter {iter_num}  val_loss {val:.4f}")
            if val < best_val:
                best_val = val
                rank0_print(f"[ckpt] new best {best_val:.4f}")
                save_ckpt(ckpt_path, iter_num, best_val)

        # periodic sample (rank0 only)
        if args.sample_every > 0 and iter_num % args.sample_every == 0 and (not is_dist() or dist.get_rank() == 0):
            try:
                txt = sample_text(model, enc, device, n_tokens=args.sample_tokens,
                                  temperature=args.temperature, top_k=args.top_k,
                                  block_size=args.block_size, amp_dtype=mp_dtype)
                print("\n--- SAMPLE ---")
                print(txt)
                print("--------------\n")
            except RuntimeError as e:
                print(f"[sample] skipped due to error: {e}")

        # periodic save
        if args.save_every > 0 and iter_num > 0 and (iter_num % args.save_every == 0):
            save_ckpt(ckpt_path, iter_num, best_val)

        iter_num += 1

    rank0_print("[train] done.")

    # cleanup
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()