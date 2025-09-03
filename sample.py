#!/usr/bin/env python3
# sample.py — Multi-GPU FSDP-aware text generation from sharded/full checkpoints

import argparse, os, datetime
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace

import torch
import torch.distributed as dist
import numpy as np

try:
    import tiktoken
except ImportError as e:
    raise SystemExit("Please `pip install tiktoken` first.") from e

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy

# Your model code (same as training)
from model import Transformer, ModelConfig, TransformerBlock

# ----------------------------- Args -----------------------------
def get_args():
    ap = argparse.ArgumentParser(description="FSDP sharded sampling")
    # ckpt options
    ap.add_argument("--out_dir", type=str, default="out",
                    help="folder containing ckpts (for sharded: <out_dir>/<prefix>_rank00000.pt ...)")
    ap.add_argument("--ckpt_prefix", type=str, default="ckpt",
                    help="prefix for sharded files: <out_dir>/ckpt_rank00000.pt ...")
    ap.add_argument("--ckpt_full", type=str, default="",
                    help="optional single full ckpt file path (fallback if sharded not found)")
    # sampling
    ap.add_argument("--prompt", type=str, default="", help="prompt string (UTF-8)")
    ap.add_argument("--prompt_file", type=str, default="", help="read prompt from file if set")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=200)
    ap.add_argument("--top_p", type=float, default=0.0, help="nucleus sampling; 0 disables")
    ap.add_argument("--repetition_penalty", type=float, default=1.0, help=">1.0 penalizes repeats")
    ap.add_argument("--eos_id", type=int, default=-1,
                    help="-1 to use id from ckpt config if present; otherwise ignore EOS")
    ap.add_argument("--block_size", type=int, default=0, help="0 => use model max")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dtype", type=str, choices=["float32","bfloat16","float16"], default="bfloat16")
    ap.add_argument("--wrap_policy", type=str, choices=["transformer","size"], default="transformer")
    ap.add_argument("--min_params", type=int, default=2_000_000)
    ap.add_argument("--print_tokens", action="store_true", help="print token ids as they are generated")
    return ap.parse_args()

# ----------------------- Dist / Helpers -------------------------
def is_dist():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1

def rank0_print(*args, **kwargs):
    if not is_dist() or dist.get_rank() == 0:
        print(*args, **kwargs)

def init_dist():
    if is_dist():
        timeout = datetime.timedelta(minutes=60)
        dist.init_process_group(backend="nccl", timeout=timeout)
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        rank = dist.get_rank()
        world = dist.get_world_size()
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rank, world = 0, 1
    return device, rank, world

def sharded_path(out_dir, prefix, rank):
    return os.path.join(out_dir, f"{prefix}_rank{rank:05d}.pt")

def top_k_filter(logits, k):
    if k <= 0: return logits
    v, _ = torch.topk(logits, min(k, logits.size(-1)))
    logits[logits < v[..., [-1]]] = -float("inf")
    return logits

def top_p_filter(logits, top_p):
    if top_p <= 0.0 or top_p >= 1.0:
        return logits
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    mask = cum > top_p
    # keep at least the first token
    mask[..., 0] = False
    logits_filtered = logits.clone()
    logits_filtered[..., sorted_idx[mask]] = -float("inf")
    return logits_filtered

def apply_repetition_penalty(logits, generated_ids, penalty):
    if penalty <= 1.0:
        return logits
    with torch.no_grad():
        vocab = logits.size(-1)
        seen = torch.bincount(generated_ids.flatten(), minlength=vocab).clamp(max=1).to(logits.dtype)
        # subtract a small constant from seen tokens' logits
        logits = logits - (penalty - 1.0) * 5.0 * seen
    return logits

# --------- Config massaging (fix rope_scaling dict -> object) ----------
def _massage_cfg_for_model(cfg: dict) -> dict:
    """Normalize nested fields so model.py (RotaryEmbedding etc.) gets attribute-like objects."""
    cfg = dict(cfg) if cfg is not None else {}
    r = cfg.get("rope_scaling", None)
    if isinstance(r, dict):
        # Support 'scale' alias used by some configs
        if "factor" not in r and "scale" in r:
            r["factor"] = r.pop("scale")
        cfg["rope_scaling"] = SimpleNamespace(**r)
    return cfg

# ---------------------- Build & Load Model ----------------------
def build_model_from_config(cfg: dict, device: str, args) -> FSDP:
    # Normalize nested bits (e.g., rope_scaling)
    cfg = _massage_cfg_for_model(cfg)

    # Rehydrate dataclass
    model_cfg = ModelConfig(**cfg)

    # Cap block size if user asked bigger than model max
    if args.block_size <= 0 or args.block_size > model_cfg.max_position_embeddings:
        args.block_size = int(model_cfg.max_position_embeddings)

    # wrapping policy
    if args.wrap_policy == "transformer":
        auto_wrap = partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})
    else:
        auto_wrap = partial(size_based_auto_wrap_policy, min_num_params=max(1_000_000, args.min_params))

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    mp = MixedPrecision(param_dtype=dtype_map[args.dtype],
                        reduce_dtype=dtype_map[args.dtype],
                        buffer_dtype=dtype_map[args.dtype])

    # Construct on meta, then allocate directly to the correct device
    if hasattr(torch, "set_default_device"): torch.set_default_device("meta")
    base = Transformer(model_cfg)
    if hasattr(torch, "set_default_device"): torch.set_default_device("cpu")

    def _param_init_fn(m: torch.nn.Module):
        m.to_empty(device=torch.device(device))
        # weights will be overwritten by checkpoint load

    model = FSDP(
        base,
        auto_wrap_policy=auto_wrap,
        device_id=None,
        mixed_precision=mp,
        use_orig_params=True,
        limit_all_gathers=True,
        param_init_fn=_param_init_fn,
    )
    return model

# --------------------------- Sampling ---------------------------
@torch.no_grad()
def generate_collective(model: FSDP, enc, device, args, eos_id: int):
    """
    All ranks participate in forward passes (FSDP requirement).
    Rank 0 samples next token id and broadcasts to all ranks so sequences stay in sync.
    """
    rank = dist.get_rank() if is_dist() else 0

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    amp_ctx = nullcontext() if "cuda" not in str(device) else torch.amp.autocast(device_type="cuda",
                                                                                  dtype=dtype_map[args.dtype])

    # prompt
    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()
    if prompt is None: prompt = ""
    input_ids = enc.encode(prompt) if prompt.strip() else enc.encode("\n")

    # same seed across ranks (for reproducibility if needed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tokens = torch.tensor([input_ids], device=device, dtype=torch.long)  # [1, T]

    model.eval()
    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            inp = tokens[:, -args.block_size:]
            with amp_ctx:
                logits, _ = model(inp, labels=None)   # [1, L, vocab]
                next_logits = logits[:, -1, :].squeeze(0)  # [vocab]

            # sampling transforms
            if args.repetition_penalty > 1.0:
                next_logits = apply_repetition_penalty(next_logits, tokens, args.repetition_penalty)
            if args.temperature != 1.0:
                next_logits = next_logits / max(1e-6, args.temperature)
            if args.top_k > 0:
                next_logits = top_k_filter(next_logits, args.top_k)
            if args.top_p > 0.0:
                next_logits = top_p_filter(next_logits, args.top_p)

            # rank 0 samples, then broadcast
            if (not is_dist()) or rank == 0:
                probs = torch.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)  # [1]
            else:
                next_id = torch.empty((1,), device=device, dtype=torch.long)

            if is_dist():
                dist.broadcast(next_id, src=0)

            tokens = torch.cat([tokens, next_id.view(1, 1)], dim=1)

            if args.print_tokens and (rank == 0):
                print(int(next_id.item()), end=" ", flush=True)

            # stop at EOS
            if eos_id is not None and eos_id >= 0 and int(next_id.item()) == eos_id:
                break

    if rank == 0:
        return enc.decode(tokens[0].tolist())
    return ""

# ----------------------------- Main -----------------------------
def main():
    args = get_args()
    device, rank, world = init_dist()

    # decide checkpoint style
    sharded_first = sharded_path(args.out_dir, args.ckpt_prefix, 0)
    using_sharded = os.path.exists(sharded_first)

    if not using_sharded and args.ckpt_full:
        if not os.path.exists(args.ckpt_full):
            raise FileNotFoundError(f"--ckpt_full not found: {args.ckpt_full}")
    elif not using_sharded and not args.ckpt_full:
        raise SystemExit(f"No sharded files like {sharded_first} and no --ckpt_full provided.")

    # Load payload metadata first to get config/tokenizer
    if using_sharded:
        path = sharded_path(args.out_dir, args.ckpt_prefix, rank)
        payload = torch.load(path, map_location="cpu", weights_only=False)
    else:
        payload = torch.load(args.ckpt_full, map_location="cpu", weights_only=False)

    cfg_dict = payload.get("model_config_dict", {})
    tok_name = payload.get("tokenizer", "o200k_harmony")

    # build model using payload config
    model = build_model_from_config(cfg_dict, device, args)

    # load weights
    if using_sharded:
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            model.load_state_dict(payload["model_state_dict"])
        iter_num = int(payload.get("iter_num", 0))
        best_val = float(payload.get("best_val_loss", float("inf")))
        if is_dist(): dist.barrier()
        if rank == 0:
            rank0_print(f"[load] loaded SHARDED ckpt @ iter {iter_num} best {best_val:.4f}")
    else:
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            model.load_state_dict(payload["model_state_dict"])
        iter_num = int(payload.get("iter_num", 0))
        best_val = float(payload.get("best_val_loss", float("inf")))
        if rank == 0:
            rank0_print(f"[load] loaded FULL ckpt @ iter {iter_num} best {best_val:.4f}")

    # tokenizer
    try:
        enc = tiktoken.get_encoding(tok_name)
    except Exception:
        enc = tiktoken.get_encoding("o200k_base")
        rank0_print(f"[sample] WARNING: tokenizer '{tok_name}' not found; using 'o200k_base'.")

    # eos id (optional)
    eos_id = args.eos_id
    if eos_id < 0:
        eos_id = cfg_dict.get("eos_token_id", None)
        if eos_id is None:
            eos_id = -1

    text = generate_collective(model, enc, device, args, eos_id)
    if (not is_dist()) or rank == 0:
        print("\n--- OUTPUT ---")
        print(text)
        print("--------------")

    if dist.is_initialized():
        dist.barrier(); dist.destroy_process_group()

if __name__ == "__main__":
    main()