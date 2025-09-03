#!/usr/bin/env python3
import argparse, dataclasses, json, math, os, time
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

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy

from model import Transformer, ModelConfig, gpt_oss_20b_config, TransformerBlock

# ------------------------------- args ----------------------------------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/tinystories")
    ap.add_argument("--out_dir", type=str, default="out")
    ap.add_argument("--model_size", type=str, choices=["toy", "20b"], default="toy")
    # training
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--max_iters", type=int, default=2000)
    ap.add_argument("--grad_accum_steps", type=int, default=8)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--eval_interval", type=int, default=500)
    ap.add_argument("--eval_iters", type=int, default=100)
    # save + sample
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--sample_every", type=int, default=100)
    ap.add_argument("--sample_tokens", type=int, default=120)
    ap.add_argument("--top_k", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    # optim
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
    ap.add_argument("--compile", action="store_true", default=False)
    # wrap policy
    ap.add_argument("--wrap_policy", type=str, choices=["transformer", "size"], default="transformer")
    ap.add_argument("--min_params", type=int, default=2_000_000)
    return ap.parse_args()

# ------------------------------ helpers --------------------------------------
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
        enc = tiktoken.get_encoding("o200k_base"); tok_name = "o200k_base"
        rank0_print(f"[train] WARNING: tokenizer '{meta.get('tokenizer')}' not available. Using 'o200k_base'.")
    vocab_size = int(meta.get("vocab_size", getattr(enc, "n_vocab", 201_088)))
    return enc, tok_name, vocab_size

def build_config(name: str) -> ModelConfig:
    if name == "20b":
        return gpt_oss_20b_config()
    return ModelConfig(
        vocab_size=200_019, hidden_size=256, num_hidden_layers=8, head_dim=32,
        num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=2048,
        sliding_window=64, num_local_experts=4, experts_per_token=2, intermediate_size=256,
        rope_theta=150_000.0, enable_sink_logit=False, tie_word_embeddings=False,
    )

@torch.no_grad()
def sample_text(model, enc, device, n_tokens=120, temperature=0.8, top_k=200,
                block_size=512, amp_dtype=torch.bfloat16):
    # Use FSDP's recommended context manager for inference
    # This gathers all the parameters on rank 0 without OOMing on other ranks
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
    
    # We must be on rank 0 to sample
    if is_dist() and dist.get_rank() != 0:
        return ""

    was_training = model.training
    model.eval()

    start_tok = enc.encode("\n")[0]
    tokens = torch.tensor([[start_tok]], device=device, dtype=torch.long)
    
    # Use bfloat16 for inference
    ctx = nullcontext() if "cpu" in str(device) else torch.amp.autocast("cuda", dtype=amp_dtype)

    # FSDP's summon_full_params is the safe way to get the full model for inference
    # It offloads to CPU to avoid VRAM spikes and only does this on rank 0.
    with FSDP.summon_full_params(model, state_dict_type=StateDictType.FULL_STATE_DICT, offload_to_cpu=True, rank0_only=True):
        with ctx:
            for _ in range(n_tokens):
                inp = tokens[:, -block_size:]
                # The model is now on CPU because of offload_to_cpu=True, so move input to CPU
                logits, _ = model(inp.to('cpu'), labels=None)
                
                # Move logits to the target device for sampling
                nxt_logits = logits[:, -1, :].to(device)

                if temperature != 1.0: nxt_logits = nxt_logits / max(1e-6, temperature)
                if top_k and top_k > 0:
                    v, _ = torch.topk(nxt_logits, top_k)
                    nxt_logits[nxt_logits < v[:, [-1]]] = -float("inf")
                
                probs = torch.softmax(nxt_logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
                tokens = torch.cat([tokens, nxt], dim=1)

    if was_training:
        model.train()
    
    return enc.decode(tokens[0].tolist())

# -------------------------------- main ---------------------------------------
def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = get_args()

    # dist
    if is_dist():
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        rank = dist.get_rank(); world_size = dist.get_world_size()
    else:
        local_rank = 0
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rank = 0; world_size = 1

    if args.model_size == "20b" and world_size < 8:
        rank0_print("WARNING: 20B ideally needs many GPUs (e.g., 8×A100-80GB).")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed + rank); np.random.seed(args.seed + rank)

    # tokenizer & cfg
    enc, tok_name, vocab_size = load_tokenizer(os.path.join(args.data_dir, "meta.json"))
    cfg = build_config(args.model_size); cfg.vocab_size = vocab_size
    if args.block_size > cfg.max_position_embeddings:
        rank0_print(f"[train] Reducing block_size from {args.block_size} to model max {cfg.max_position_embeddings}")
        args.block_size = cfg.max_position_embeddings

    # --- build on meta on ALL ranks ---
    if hasattr(torch, "set_default_device"): torch.set_default_device("meta")
    base_model = Transformer(cfg)
    if hasattr(torch, "set_default_device"): torch.set_default_device("cpu")

    if rank == 0:
        total_params = sum(p.numel() for p in base_model.parameters())
        trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        rank0_print(f"[params] Global: total {total_params/1e9:.2f}B  trainable {trainable_params/1e9:.2f}B")

    # wrap policy
    if args.wrap_policy == "transformer":
        auto_wrap = partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})
    else:
        auto_wrap = partial(size_based_auto_wrap_policy, min_num_params=max(1_000_000, args.min_params))

    # mixed precision
    mp_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    mp = MixedPrecision(param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype)

    # --- materialize & init from meta safely ---
    def _fsdp_param_init_fn(m: torch.nn.Module):
        # 1) allocate storage on right CUDA device (no data copy from meta)
        m.to_empty(device=torch.device(device))
        # 2) init parameters (custom + standard layers)
        std = getattr(getattr(m, "config", None), "initializer_range", 0.02)

        for mod in m.modules():
            # nn.Linear
            if isinstance(mod, torch.nn.Linear):
                if mod.weight is not None and mod.weight.is_meta is False:
                    torch.nn.init.normal_(mod.weight, mean=0.0, std=std)
                if mod.bias is not None and mod.bias.is_meta is False:
                    torch.nn.init.zeros_(mod.bias)
            # nn.Embedding
            elif isinstance(mod, torch.nn.Embedding):
                if mod.weight is not None and mod.weight.is_meta is False:
                    torch.nn.init.normal_(mod.weight, mean=0.0, std=std)
            # RMSNorm (custom)
            elif mod.__class__.__name__ == "RMSNorm":
                if hasattr(mod, "weight") and mod.weight is not None and mod.weight.is_meta is False:
                    torch.nn.init.ones_(mod.weight)
            # MoE (custom)
            elif mod.__class__.__name__ == "MoE":
                if hasattr(mod, "W_in") and not mod.W_in.is_meta: torch.nn.init.normal_(mod.W_in, 0.0, std)
                if hasattr(mod, "W_out") and not mod.W_out.is_meta: torch.nn.init.normal_(mod.W_out, 0.0, std)
                if hasattr(mod, "b_in") and not mod.b_in.is_meta: torch.nn.init.zeros_(mod.b_in)
                if hasattr(mod, "b_out") and not mod.b_out.is_meta: torch.nn.init.zeros_(mod.b_out)
                if hasattr(mod, "router"):
                    if hasattr(mod.router, "weight") and not mod.router.weight.is_meta:
                        torch.nn.init.normal_(mod.router.weight, 0.0, std)
                    if hasattr(mod.router, "bias") and mod.router.bias is not None and not mod.router.bias.is_meta:
                        torch.nn.init.zeros_(mod.router.bias)
            # MultiheadSelfAttention (custom)
            elif mod.__class__.__name__ == "MultiheadSelfAttention":
                for name in ("q","k","v","o"):
                    lin = getattr(mod, name, None)
                    if isinstance(lin, torch.nn.Linear):
                        if lin.weight is not None and not lin.weight.is_meta:
                            torch.nn.init.normal_(lin.weight, 0.0, std)
                        if lin.bias is not None and not lin.bias.is_meta:
                            torch.nn.init.zeros_(lin.bias)
                if getattr(mod, "use_sink", False) and hasattr(mod, "sink_logit") and not mod.sink_logit.is_meta:
                    with torch.no_grad():
                        mod.sink_logit.fill_(float(getattr(getattr(m, "config", None), "sink_logit_init", 4.0)))
        # RotaryEmbedding holds buffers; they’re computed on first forward.

    # FSDP (no device_id, no sync_module_states)
    model = FSDP(
        base_model,
        auto_wrap_policy=auto_wrap,
        device_id=None,
        mixed_precision=mp,
        use_orig_params=True,
        limit_all_gathers=True,
        param_init_fn=_fsdp_param_init_fn,
    )

    if args.compile and "cuda" in device:
        rank0_print("[train] Skipping torch.compile with FSDP.")

    shard_params = sum(p.numel() for p in model.parameters())
    rank0_print("=" * 60)
    rank0_print(f"Model: {args.model_size}   device: {device}   world_size: {world_size}")
    rank0_print(f"Tokenizer: {tok_name}  vocab: {vocab_size}")
    rank0_print(f"[params] Per-rank shard ~ {shard_params/1e9:.2f}B")
    rank0_print(f"Context: train block {args.block_size} (model max {cfg.max_position_embeddings})")
    rank0_print("=" * 60)

    # data
    train_loader = BinLoader(args.data_dir, "train", args.block_size, args.batch_size, device, seed=args.seed+rank)
    val_loader   = BinLoader(args.data_dir, "val",   args.block_size, args.batch_size, device, seed=args.seed+1234+rank)

    # AMP
    ctx = nullcontext() if "cpu" in device else torch.amp.autocast("cuda", dtype=mp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=("cuda" in device and args.dtype == "float16"))

    # optim
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

    # resume
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    iter_num = 0; best_val = float("inf")
    if os.path.exists(ckpt_path):
        if rank == 0:
            rank0_print(f"[train] Resuming from {ckpt_path}")
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if is_dist(): dist.barrier()
        if rank != 0: payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            model.load_state_dict(payload["model_state_dict"])
            shard_optim = FSDP.optim_state_dict_to_load(model, opt, payload["optimizer_state_dict"])
        opt.load_state_dict(shard_optim)
        iter_num = int(payload.get("iter_num", 0))
        best_val = float(payload.get("best_val_loss", best_val))
        if is_dist(): dist.barrier()
        rank0_print(f"[train] Resumed at iter {iter_num} (best val {best_val:.4f})")

    # LR sched
    def get_lr(it: int) -> float:
        if not args.decay_lr: return args.lr
        if it < args.warmup_iters: return args.lr * it / max(1, args.warmup_iters)
        if it > args.lr_decay_iters: return args.min_lr
        decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return args.min_lr + coeff * (args.lr - args.min_lr)

    # eval
    def evaluate() -> float:
        model.eval(); losses = []
        with torch.no_grad():
            for _ in range(args.eval_iters):
                Xv, Yv = val_loader.get_batch()
                with ctx: _, out = model(Xv, labels=Yv)
                losses.append(out["loss"].item())
        model.train()
        val = torch.tensor([sum(losses)/max(1, len(losses))], device=device)
        if is_dist(): dist.all_reduce(val, op=dist.ReduceOp.AVG)
        return float(val.item())

    # save (rank0)
    def save_ckpt(path, iter_n, best_v):
        if dist.is_initialized() and dist.get_rank() != 0:
            if is_dist(): dist.barrier()
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
            "iter_num": iter_n, "best_val_loss": best_v, "tokenizer": tok_name,
        }
        torch.save(payload, path)
        rank0_print(f"[ckpt] saved {path}")
        if is_dist(): dist.barrier()

    # train loop
    t0 = time.time()
    X, Y = train_loader.get_batch()
    while iter_num < args.max_iters:
        lr = get_lr(iter_num)
        for g in opt.param_groups: g["lr"] = lr
        opt.zero_grad(set_to_none=True); total_loss = 0.0

        for _ in range(args.grad_accum_steps):
            with ctx: _, out = model(X, labels=Y); loss = out["loss"] / args.grad_accum_steps
            total_loss += float(loss.detach().item())
            if scaler.is_enabled(): scaler.scale(loss).backward()
            else: loss.backward()
            X, Y = train_loader.get_batch()

        if args.grad_clip > 0:
            if scaler.is_enabled(): scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if scaler.is_enabled(): scaler.step(opt); scaler.update()
        else: opt.step()

        if iter_num % args.log_interval == 0:
            if "cuda" in device: torch.cuda.synchronize()
            loss_t = torch.tensor([total_loss], device=device)
            if is_dist(): dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
            dt = time.time() - t0; t0 = time.time()
            rank0_print(f"iter {iter_num:06d}  loss {loss_t.item():.4f}  lr {lr:.6e}  {dt*1000:.1f} ms/it "
                        f"(G: {world_size*args.batch_size*args.grad_accum_steps})")

        if args.eval_interval > 0 and iter_num > 0 and (iter_num % args.eval_interval == 0):
            val = evaluate()
            rank0_print(f"[eval] iter {iter_num}  val_loss {val:.4f}")
            if val < best_val:
                best_val = val; rank0_print(f"[ckpt] new best {best_val:.4f}")
                save_ckpt(ckpt_path, iter_num, best_val)

        if (args.sample_every > 0 and iter_num > 0 and (iter_num % args.sample_every == 0)
                and (not is_dist() or dist.get_rank() == 0)):
            try:
                txt = sample_text(model, enc, device,
                                  n_tokens=args.sample_tokens, temperature=args.temperature,
                                  top_k=args.top_k, block_size=args.block_size, amp_dtype=mp_dtype)
                print("\n--- SAMPLE ---"); print(txt); print("--------------\n")
            except RuntimeError as e:
                print(f"[sample] skipped due to error: {e}")

        if args.save_every > 0 and iter_num > 0 and (iter_num % args.save_every == 0):
            save_ckpt(ckpt_path, iter_num, best_val)

        iter_num += 1

    rank0_print("[train] done.")
    if dist.is_initialized():
        dist.barrier(); dist.destroy_process_group()

if __name__ == "__main__":
    main()