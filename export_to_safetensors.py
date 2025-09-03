#!/usr/bin/env python3
"""
Export FSDP-sharded training checkpoints (.pt with optimizer) -> HuggingFace-style weights-only .safetensors

- NO transformers dependency (we implement simple sharding here)
- Loads metadata only on rank 0 to avoid ShardedTensor local-rank mismatch
- Writes to a NEW folder (default under /workspace)

Run (same world size as training, e.g. 5):
  torchrun --nproc_per_node=5 export_to_safetensors.py \
    --in_dir out-20b-h200-stable \
    --ckpt_prefix ckpt \
    --max_shard_size 5GB \
    --release_dir /workspace/20b-release

Requires:
  pip install safetensors
"""

import os, json, argparse, datetime
import torch
import torch.distributed as dist
from functools import partial

from safetensors.torch import save_file

# import your model bits
from model import Transformer, ModelConfig, TransformerBlock, RopeScalingConfig

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy


# --------------------------- args ---------------------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True, help="Folder with sharded .pt files")
    ap.add_argument("--ckpt_prefix", type=str, default="ckpt", help="Prefix of sharded training ckpts")
    ap.add_argument("--release_dir", type=str, default="", help="Absolute output dir (default: auto under /workspace)")
    ap.add_argument("--wrap_policy", type=str, choices=["transformer","size"], default="transformer")
    ap.add_argument("--min_params", type=int, default=2_000_000)
    ap.add_argument("--dtype", type=str, choices=["float32","bfloat16","float16"], default="bfloat16")
    ap.add_argument("--max_shard_size", type=str, default="5GB", help='e.g. "2GB", "5GB", "10GB"')
    return ap.parse_args()


# --------------------- misc helpers ---------------------
def is_dist() -> bool:
    return int(os.environ.get("WORLD_SIZE","1")) > 1

def rank0_print(*a, **k):
    if (not is_dist()) or dist.get_rank() == 0:
        print(*a, **k)

def sharded_path(in_dir, prefix, rank):
    return os.path.join(in_dir, f"{prefix}_rank{rank:05d}.pt")

_UNITS = {"B":1, "KB":1024, "MB":1024**2, "GB":1024**3, "TB":1024**4}
def parse_size(s: str) -> int:
    if isinstance(s, (int, float)): return int(s)
    s = s.strip().upper().replace(" ", "")
    for u in ["TB","GB","MB","KB","B"]:
        if s.endswith(u):
            return int(float(s[:-len(u)]) * _UNITS[u])
    return int(s)  # raw bytes

# allowlist ShardedTensor in case safe loader is used elsewhere
try:
    from torch.distributed._shard.sharded_tensor.api import ShardedTensor
    import torch.serialization as _ser
    _ser.add_safe_globals([ShardedTensor])
except Exception:
    pass


# ---------------------- config normalization ----------------------
def _massage_cfg_for_model(cfg_dict: dict) -> dict:
    """Turn nested dicts into expected types (e.g., rope_scaling dict -> RopeScalingConfig)."""
    if cfg_dict is None:
        return {}
    cfg = dict(cfg_dict)
    rs = cfg.get("rope_scaling", None)
    if isinstance(rs, dict):
        # support alias 'scale' -> 'factor'
        if "factor" not in rs and "scale" in rs:
            rs["factor"] = rs.pop("scale")
        cfg["rope_scaling"] = RopeScalingConfig(**rs)
    return cfg


# ---------------------- build FSDP model ----------------------
def build_model_from_cfg(cfg_dict: dict, device: str, args) -> FSDP:
    cfg = _massage_cfg_for_model(cfg_dict)
    model_cfg = ModelConfig(**cfg)

    if args.wrap_policy == "transformer":
        auto_wrap = partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})
    else:
        auto_wrap = partial(size_based_auto_wrap_policy, min_num_params=max(1_000_000, args.min_params))

    dmap = {"float32":torch.float32,"bfloat16":torch.bfloat16,"float16":torch.float16}
    mp = MixedPrecision(param_dtype=dmap[args.dtype], reduce_dtype=dmap[args.dtype], buffer_dtype=dmap[args.dtype])

    # meta build -> to_empty on correct device
    if hasattr(torch, "set_default_device"): torch.set_default_device("meta")
    base = Transformer(model_cfg)
    if hasattr(torch, "set_default_device"): torch.set_default_device("cpu")

    def _param_init_fn(m: torch.nn.Module):
        m.to_empty(device=torch.device(device))  # weights will be overwritten by ckpt load

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


# --------------------- sharding (no transformers) ---------------------
def shard_state_dict_for_hf(state: dict, max_shard_bytes: int, base_name="pytorch_model"):
    """
    Split a full weights dict into shards <= max_shard_bytes.
    Returns:
      shards: { filename -> {param_name: tensor, ...} }
      index:  {"metadata":{"total_size":bytes},"weight_map":{param_name:filename}}
    """
    shards_list = []   # list[dict[param_name->tensor]]
    shard_sizes = []   # bytes per shard
    weight_map = {}
    total = 0

    current = {}
    cur_size = 0

    for name, tensor in state.items():
        if not torch.is_tensor(tensor):
            continue
        t_bytes = tensor.element_size() * tensor.numel()
        if current and cur_size + t_bytes > max_shard_bytes:
            shards_list.append(current)
            shard_sizes.append(cur_size)
            current, cur_size = {}, 0
        current[name] = tensor
        cur_size += t_bytes
        total += t_bytes
    if current:
        shards_list.append(current)
        shard_sizes.append(cur_size)

    n = len(shards_list)
    shards = {}
    for i, shard in enumerate(shards_list, start=1):
        fname = f"{base_name}-{i:05d}-of-{n:05d}.safetensors"
        shards[fname] = shard
        for k in shard.keys():
            weight_map[k] = fname

    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    return shards, index


# --------------------------- main ---------------------------
def main():
    args = get_args()

    # init dist
    if is_dist():
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=60))
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        rank = dist.get_rank(); world = dist.get_world_size()
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rank, world = 0, 1

    # sanity: shards exist
    for r in range(world):
        p = sharded_path(args.in_dir, args.ckpt_prefix, r)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing shard for rank {r}: {p}")

    # ---------- load metadata ONLY on rank 0, then broadcast ----------
    if (not is_dist()) or rank == 0:
        meta = torch.load(sharded_path(args.in_dir, args.ckpt_prefix, 0),
                          map_location="cpu", weights_only=False)  # trusted file you created
        cfg_dict = meta.get("model_config_dict", {})
        iter_num = int(meta.get("iter_num", 0))
        tok_name = meta.get("tokenizer", "o200k_harmony")
    else:
        cfg_dict, iter_num, tok_name = None, 0, ""

    if is_dist():
        obj = [cfg_dict, iter_num, tok_name]
        dist.broadcast_object_list(obj, src=0)
        cfg_dict, iter_num, tok_name = obj

    # decide output dir (outside training folder by default)
    if args.release_dir:
        release_dir = args.release_dir
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        release_dir = f"/workspace/release-iter{iter_num:06d}-{stamp}"
    if (not is_dist()) or rank == 0:
        os.makedirs(release_dir, exist_ok=True)
        rank0_print(f"[export] writing HF weights to: {release_dir}")
    if is_dist(): dist.barrier()

    # ---------- build model & load THIS rank's shard ----------
    model = build_model_from_cfg(cfg_dict, device, args)
    my_path = sharded_path(args.in_dir, args.ckpt_prefix, rank)
    payload = torch.load(my_path, map_location="cpu", weights_only=False)  # trusted file

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        model.load_state_dict(payload["model_state_dict"])

    if is_dist(): dist.barrier()

    # ---------- gather full weights to rank 0 (CPU) ----------
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT,
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        full_state = model.state_dict()

    # ---------- rank 0: save shards + index + config ----------
    if (not is_dist()) or rank == 0:
        max_bytes = parse_size(args.max_shard_size)
        shards, index = shard_state_dict_for_hf(full_state, max_bytes, base_name="pytorch_model")

        # write shards
        for fname, shard in shards.items():
            save_file(shard, os.path.join(release_dir, fname))

        # write index
        with open(os.path.join(release_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)

        # minimal config.json (adjust fields if your config differs)
        cfg = _massage_cfg_for_model(cfg_dict)
        hf_cfg = dict(
            model_type="gpt_oss_like",
            vocab_size=int(cfg["vocab_size"]),
            hidden_size=int(cfg["hidden_size"]),
            num_hidden_layers=int(cfg["num_hidden_layers"]),
            num_attention_heads=int(cfg["num_attention_heads"]),
            num_key_value_heads=int(cfg.get("num_key_value_heads", cfg["num_attention_heads"])),
            max_position_embeddings=int(cfg["max_position_embeddings"]),
            rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-5)),
            rope_theta=float(cfg.get("rope_theta", 1e5)),
            attention_bias=bool(cfg.get("attention_bias", True)),
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
            sliding_window=int(cfg.get("sliding_window", 0)),
            torch_dtype=args.dtype,
            eos_token_id=cfg.get("eos_token_id", None),
        )
        with open(os.path.join(release_dir, "config.json"), "w") as f:
            json.dump(hf_cfg, f, indent=2)

        # tiny readme
        with open(os.path.join(release_dir, "README_EXPORT.md"), "w") as f:
            f.write(
                f"# Exported Weights\n\n"
                f"- Source shards: {args.in_dir}/{args.ckpt_prefix}_rankXXXXX.pt\n"
                f"- Iteration: {iter_num}\n"
                f"- Tokenizer hint: {tok_name}\n"
                f"- max_shard_size: {args.max_shard_size}\n"
                f"- Generated: {datetime.datetime.now().isoformat()}\n"
            )

        rank0_print(f"[export] done. Files in: {release_dir}")

    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()