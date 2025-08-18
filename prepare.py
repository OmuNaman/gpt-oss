#!/usr/bin/env python3
"""
prepare.py — Tokenize TinyStories with o200k_harmony and write NanoGPT-style memmaps.

Output layout:
  <out_dir>/
    train.bin  (uint32 token ids)
    val.bin    (uint32 token ids)
    meta.json  (tokenizer + vocab info)
"""
import argparse, json, os, sys
from typing import List

import numpy as np
from datasets import load_dataset

try:
    import tiktoken
except ImportError:
    raise SystemExit("Please `pip install tiktoken datasets` first.")

# ---- tokenizer helpers ------------------------------------------------------

def get_o200k_harmony_tokenizer():
    """
    Try to get 'o200k_harmony'. If not present, fall back to 'o200k_base'
    and warn, but write the actual tokenizer name into meta.json so the
    training script can adjust vocab_size to match.
    """
    preferred = "o200k_harmony"
    fallback = "o200k_base"
    try:
        enc = tiktoken.get_encoding(preferred)
        return enc, preferred
    except Exception:
        try:
            enc = tiktoken.get_encoding(fallback)
            print(f"[prepare] WARNING: '{preferred}' not found, using '{fallback}' instead.", file=sys.stderr)
            return enc, fallback
        except Exception as e:
            raise SystemExit(
                "Could not load tiktoken encodings 'o200k_harmony' or 'o200k_base'. "
                "Upgrade tiktoken (`pip install -U tiktoken`)."
            )

# ---- io helpers -------------------------------------------------------------

def encode_corpus(texts: List[str], enc) -> np.ndarray:
    # Encode as a single flat stream of tokens (with newlines between samples)
    ids: List[int] = []
    for s in texts:
        # TinyStories has small snippets; a newline separator is enough
        ids.extend(enc.encode(s))
        ids.append(enc.eot_token if hasattr(enc, "eot_token") and enc.eot_token is not None else enc.encode("\n")[0])
    return np.array(ids, dtype=np.uint32)

def write_memmap(path: str, tokens: np.ndarray):
    arr = np.memmap(path, dtype=np.uint32, mode="w+", shape=(tokens.size,))
    arr[:] = tokens
    arr.flush()

# ---- main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/tinystories", help="where to write memmaps")
    ap.add_argument("--val_ratio", type=float, default=0.1, help="fraction for validation split")
    ap.add_argument("--dataset", type=str, default="roneneldan/TinyStories")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # load TinyStories (has 'train' split only; we create train/val)
    print(f"[prepare] loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset)
    if "train" not in ds:
        raise SystemExit("Dataset has no 'train' split.")
    train_all = ds["train"]

    # simple split
    val_count = int(len(train_all) * args.val_ratio)
    val_ds = train_all.select(range(val_count))
    train_ds = train_all.select(range(val_count, len(train_all)))

    enc, tok_name = get_o200k_harmony_tokenizer()
    # tiktoken enc has n_vocab
    vocab_size = getattr(enc, "n_vocab", None)
    if vocab_size is None:
        # last resort for very old tiktoken; shouldn't happen
        print("[prepare] WARNING: tokenizer has no n_vocab; assuming 201088", file=sys.stderr)
        vocab_size = 201_088

    # encode
    print(f"[prepare] encoding {len(train_ds)} train / {len(val_ds)} val with '{tok_name}' (vocab={vocab_size})")
    train_texts = [ex["text"] for ex in train_ds]
    val_texts = [ex["text"] for ex in val_ds]

    train_ids = encode_corpus(train_texts, enc)
    val_ids = encode_corpus(val_texts, enc)

    # write memmaps
    train_bin = os.path.join(args.out_dir, "train.bin")
    val_bin   = os.path.join(args.out_dir, "val.bin")
    print(f"[prepare] writing {train_bin} ({train_ids.size} tokens)")
    write_memmap(train_bin, train_ids)
    print(f"[prepare] writing {val_bin} ({val_ids.size} tokens)")
    write_memmap(val_bin, val_ids)

    # meta
    meta = {
        "tokenizer": tok_name,
        "vocab_size": vocab_size,
        "train_tokens": int(train_ids.size),
        "val_tokens": int(val_ids.size),
        "dataset": args.dataset,
    }
    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[prepare] wrote {meta_path}")
    print("[prepare] done.")

if __name__ == "__main__":
    main()