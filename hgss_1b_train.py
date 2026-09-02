#!/usr/bin/env python3
"""
HGSS ~1B pretraining helper for a single 16 GB NVIDIA GPU.

Designed for the user's HGSS files:
  - model(...).py
  - triton_scan(...).py

Dataset:
  IFM/CrystalCoderDatasets, already tokenized and packed to 2048-token HDF5 samples.
  Default mix: 70% SlimPajama text (excluding SlimPajama GitHub) + 30% StarCoder FIM.
  Default training budget: 10B tokens.

Examples
--------
1) Download ~10B already-tokenized tokens:
   python hgss_1b_train.py download --data-dir ./crystal10b

2) Train:
   python hgss_1b_train.py train \
       --data-dir ./crystal10b \
       --model-file "./model(20260902-145037).py" \
       --triton-file "./triton_scan(5).py" \
       --out-dir ./hgss1b_run

Notes
-----
- The HDF5 files contain `data` with shape [N, 3, 2048].
- We use data[:, 0, :] as token IDs and create local next-token labels.
  The final token in each 2048-token sample is ignored so that samples can
  be shuffled safely without depending on cross-sample labels.
- EOS id 2 is used as HGSS boundary_token_id, matching Crystal preprocessing.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import os
import random
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import h5py
import numpy as np
import requests
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Direct URLs: already-tokenized CrystalCoder / SlimPajama + StarCoder shards
# ---------------------------------------------------------------------------

HF_DATA_BASE = (
    "https://huggingface.co/datasets/IFM/CrystalCoderDatasets/resolve/main"
)

TOKENIZER_URL = (
    "https://raw.githubusercontent.com/LLM360/"
    "crystalcoder-data-prep/master/tokenizer.json"
)

# SlimPajama source proportions in the original SlimPajama mixture.
# GitHub is intentionally excluded because StarCoder supplies the code portion.
_RAW_SLIM_WEIGHTS = {
    "commoncrawl": 0.522,
    "c4": 0.267,
    "book": 0.042,
    "arxiv": 0.046,
    "wikipedia": 0.038,
    "stackexchange": 0.033,
}
_SLIM_NORM = sum(_RAW_SLIM_WEIGHTS.values())
SLIM_WEIGHTS = {k: v / _SLIM_NORM for k, v in _RAW_SLIM_WEIGHTS.items()}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    url_template: str
    start_index: int
    step: int


SOURCES: Dict[str, SourceSpec] = {
    # phase2/SlimPajama/*part1of2 contains odd-numbered shards.
    "commoncrawl": SourceSpec(
        "commoncrawl",
        HF_DATA_BASE
        + "/phase2/SlimPajama/CommonCrawl_train_packed_part1of2/data-{index:05d}.h5",
        1,
        2,
    ),
    "c4": SourceSpec(
        "c4",
        HF_DATA_BASE
        + "/phase2/SlimPajama/C4_train_packed_part1of2/data-{index:05d}.h5",
        1,
        2,
    ),
    "book": SourceSpec(
        "book",
        HF_DATA_BASE
        + "/phase2/SlimPajama/Book_train_packed_part1of2/data-{index:05d}.h5",
        1,
        2,
    ),
    "arxiv": SourceSpec(
        "arxiv",
        HF_DATA_BASE
        + "/phase2/SlimPajama/ArXiv_train_packed_part1of2/data-{index:05d}.h5",
        1,
        2,
    ),
    "wikipedia": SourceSpec(
        "wikipedia",
        HF_DATA_BASE
        + "/phase2/SlimPajama/Wikipedia_train_packed_part1of2/data-{index:05d}.h5",
        1,
        2,
    ),
    "stackexchange": SourceSpec(
        "stackexchange",
        HF_DATA_BASE
        + "/phase2/SlimPajama/StackExchange_train_packed_part1of2/data-{index:05d}.h5",
        1,
        2,
    ),
    # StarCoder FIM shards are sequentially numbered.
    "starcoder": SourceSpec(
        "starcoder",
        HF_DATA_BASE
        + "/phase2/StarCoder_fim_shuffled/data-{index:05d}.h5",
        0,
        1,
    ),
}


# ---------------------------------------------------------------------------
# Download / manifest
# ---------------------------------------------------------------------------

def format_tokens(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.3f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def target_source_tokens(total_tokens: int, code_fraction: float) -> Dict[str, int]:
    if not 0.0 <= code_fraction <= 1.0:
        raise ValueError("--code-fraction must be in [0, 1]")
    code = int(round(total_tokens * code_fraction))
    text = total_tokens - code

    targets = {
        name: int(round(text * weight))
        for name, weight in SLIM_WEIGHTS.items()
    }
    # Remove integer-rounding drift from one text source.
    drift = text - sum(targets.values())
    targets["commoncrawl"] += drift
    targets["starcoder"] = code
    return targets


def inspect_h5(path: Path) -> Tuple[int, int, int]:
    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise ValueError(f"{path}: missing HDF5 dataset 'data'")
        ds = f["data"]
        if ds.ndim != 3 or ds.shape[1] < 1:
            raise ValueError(
                f"{path}: expected data shape [N,3,T], got {tuple(ds.shape)}"
            )
        n_examples = int(f.attrs.get("n_examples", ds.shape[0]))
        seq_len = int(ds.shape[-1])
        if n_examples != ds.shape[0]:
            # Do not fail on metadata mismatch; count what is physically present.
            n_examples = int(ds.shape[0])
        return n_examples, seq_len, n_examples * seq_len


def download_file(url: str, dest: Path, retries: int = 4) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    headers = {"User-Agent": "hgss-pretrain/1.0"}
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(
                url,
                stream=True,
                timeout=(20, 180),
                allow_redirects=True,
                headers=headers,
            ) as r:
                if r.status_code == 404:
                    raise FileNotFoundError(f"404: {url}")
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                mode = "ab" if tmp.exists() else "wb"
                already = tmp.stat().st_size if tmp.exists() else 0

                # If a previous partial exists but server did not honor Range,
                # restart cleanly. For simplicity we request from scratch here.
                if already:
                    tmp.unlink()
                    mode = "wb"
                    already = 0

                with open(tmp, mode) as fh, tqdm(
                    total=total or None,
                    unit="B",
                    unit_scale=True,
                    desc=dest.name,
                    leave=False,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            fh.write(chunk)
                            bar.update(len(chunk))
            tmp.replace(dest)
            return
        except FileNotFoundError:
            raise
        except Exception as exc:
            last_exc = exc
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"failed downloading {url}: {last_exc}")


def load_manifest(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "format": 1,
        "dataset": "IFM/CrystalCoderDatasets",
        "tokenizer_url": TOKENIZER_URL,
        "entries": [],
    }


def save_manifest(path: Path, manifest: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    tmp.replace(path)


def cmd_download(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "manifest.json"
    manifest = load_manifest(manifest_path)

    targets = target_source_tokens(args.tokens, args.code_fraction)
    manifest["requested_tokens"] = int(args.tokens)
    manifest["code_fraction"] = float(args.code_fraction)
    manifest["source_targets"] = targets
    manifest["source_url_templates"] = {
        name: spec.url_template for name, spec in SOURCES.items()
    }

    existing_by_source: Dict[str, int] = {name: 0 for name in SOURCES}
    existing_indices: Dict[str, List[int]] = {name: [] for name in SOURCES}
    valid_entries = []

    # Validate / reuse already-downloaded files.
    for entry in manifest.get("entries", []):
        p = data_dir / entry["path"]
        if not p.exists():
            continue
        n_examples, seq_len, tokens = inspect_h5(p)
        entry["n_examples"] = n_examples
        entry["seq_len"] = seq_len
        entry["tokens"] = tokens
        valid_entries.append(entry)
        existing_by_source[entry["source"]] += tokens
        existing_indices[entry["source"]].append(int(entry["index"]))

    manifest["entries"] = valid_entries
    save_manifest(manifest_path, manifest)

    print("Target mix:")
    for source, target in targets.items():
        print(
            f"  {source:14s} {format_tokens(target):>9s} "
            f"(already {format_tokens(existing_by_source[source])})"
        )

    for source, target in targets.items():
        spec = SOURCES[source]
        have = existing_by_source[source]
        if have >= target:
            continue

        if existing_indices[source]:
            index = max(existing_indices[source]) + spec.step
        else:
            index = spec.start_index

        while have < target:
            url = spec.url_template.format(index=index)
            rel = Path(source) / f"data-{index:05d}.h5"
            dest = data_dir / rel

            print(
                f"[{source}] {format_tokens(have)}/{format_tokens(target)} -> {url}"
            )

            if not dest.exists():
                try:
                    download_file(url, dest)
                except FileNotFoundError as exc:
                    raise RuntimeError(
                        f"source {source!r} ran out of shards at index {index}. "
                        f"Last URL: {url}"
                    ) from exc

            n_examples, seq_len, shard_tokens = inspect_h5(dest)
            if seq_len != 2048:
                print(
                    f"WARNING: {dest} has seq_len={seq_len}; "
                    "training script supports it, but default HGSS plan assumes 2048."
                )

            entry = {
                "source": source,
                "index": index,
                "path": str(rel),
                "url": url,
                "n_examples": n_examples,
                "seq_len": seq_len,
                "tokens": shard_tokens,
            }
            # Avoid duplicates on resume.
            manifest["entries"] = [
                e
                for e in manifest["entries"]
                if not (e["source"] == source and int(e["index"]) == index)
            ]
            manifest["entries"].append(entry)
            have += shard_tokens
            existing_by_source[source] = have
            save_manifest(manifest_path, manifest)

            print(
                f"  + {format_tokens(shard_tokens)} tokens "
                f"({n_examples} x {seq_len}); source total={format_tokens(have)}"
            )
            index += spec.step

    manifest["downloaded_tokens"] = int(
        sum(int(e["tokens"]) for e in manifest["entries"])
    )

    # Keep the exact tokenizer used by the Crystal data-prep repository next
    # to the shards. Training only needs token IDs, but this is useful for
    # decoding/evaluation later.
    tokenizer_path = data_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        print(f"[tokenizer] {TOKENIZER_URL}")
        try:
            download_file(TOKENIZER_URL, tokenizer_path)
        except Exception as exc:
            print(f"WARNING: tokenizer download failed: {exc}")

    save_manifest(manifest_path, manifest)

    print()
    print(f"Manifest: {manifest_path}")
    print(f"Downloaded usable tokens: {format_tokens(manifest['downloaded_tokens'])}")
    print(
        "The downloader intentionally overshoots each source target by at most "
        "one shard; training stops at the exact requested global budget."
    )


# ---------------------------------------------------------------------------
# HDF5 source streams
# ---------------------------------------------------------------------------

class SourceStream:
    def __init__(
        self,
        data_dir: Path,
        entries: List[dict],
        seed: int,
        expected_seq_len: int,
    ):
        self.data_dir = data_dir
        self.entries = list(entries)
        if not self.entries:
            raise ValueError("empty source stream")
        self.rng = random.Random(seed)
        self.expected_seq_len = expected_seq_len
        self._file_order: List[int] = []
        self._file_cursor = 0
        self._row_order: np.ndarray | None = None
        self._row_cursor = 0
        self._h5 = None
        self._ds = None
        self._reshuffle_files()

    def _reshuffle_files(self) -> None:
        self._file_order = list(range(len(self.entries)))
        self.rng.shuffle(self._file_order)
        self._file_cursor = 0

    def _close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
        self._h5 = None
        self._ds = None
        self._row_order = None
        self._row_cursor = 0

    def _open_next_file(self) -> None:
        self._close()
        if self._file_cursor >= len(self._file_order):
            self._reshuffle_files()

        entry = self.entries[self._file_order[self._file_cursor]]
        self._file_cursor += 1
        path = self.data_dir / entry["path"]

        self._h5 = h5py.File(path, "r")
        self._ds = self._h5["data"]
        if int(self._ds.shape[-1]) != self.expected_seq_len:
            raise ValueError(
                f"{path}: seq_len={self._ds.shape[-1]}, "
                f"expected {self.expected_seq_len}"
            )
        n = int(self._ds.shape[0])
        self._row_order = np.arange(n, dtype=np.int64)
        # NumPy shuffle is faster and more compact than a Python list.
        np_rng = np.random.default_rng(self.rng.randrange(2**63))
        np_rng.shuffle(self._row_order)
        self._row_cursor = 0

    def next_ids(self) -> np.ndarray:
        if self._ds is None or self._row_cursor >= len(self._row_order):
            self._open_next_file()
        idx = int(self._row_order[self._row_cursor])
        self._row_cursor += 1

        # Crystal H5 layout: [sample, feature, token].
        # feature 0 = input token ids.
        arr = np.asarray(self._ds[idx, 0, :], dtype=np.int64)
        return arr


class MixedTokenStream:
    def __init__(
        self,
        data_dir: Path,
        manifest: dict,
        seed: int,
        seq_len: int,
        code_fraction: float,
    ):
        by_source: Dict[str, List[dict]] = {}
        for entry in manifest["entries"]:
            by_source.setdefault(entry["source"], []).append(entry)

        requested_targets = target_source_tokens(1_000_000_000, code_fraction)
        names = [name for name, t in requested_targets.items() if t > 0]
        missing = [name for name in names if not by_source.get(name)]
        if missing:
            raise ValueError(
                f"manifest is missing required sources: {', '.join(missing)}"
            )

        self.names = names
        self.weights = [requested_targets[name] for name in names]
        self.streams = {
            name: SourceStream(
                data_dir,
                by_source[name],
                seed=seed + 1009 * i,
                expected_seq_len=seq_len,
            )
            for i, name in enumerate(names)
        }
        self.rng = random.Random(seed + 99991)

    def next_ids(self) -> Tuple[str, np.ndarray]:
        source = self.rng.choices(self.names, weights=self.weights, k=1)[0]
        return source, self.streams[source].next_ids()


# ---------------------------------------------------------------------------
# Dynamic loading of the user's HGSS files
# ---------------------------------------------------------------------------

def load_user_hgss(model_file: str, triton_file: str):
    model_path = Path(model_file).resolve()
    triton_path = Path(triton_file).resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not triton_path.exists():
        raise FileNotFoundError(triton_path)

    # The user's model uses `from .triton_scan import ...`. Build a tiny
    # in-memory package so this relative import works even when the files have
    # names like model(....).py and triton_scan(5).py.
    pkg_name = "_local_hgss_pkg"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(model_path.parent)]
    pkg.__package__ = pkg_name
    sys.modules[pkg_name] = pkg

    triton_name = f"{pkg_name}.triton_scan"
    triton_spec = importlib.util.spec_from_file_location(triton_name, triton_path)
    if triton_spec is None or triton_spec.loader is None:
        raise ImportError(f"cannot load {triton_path}")
    triton_mod = importlib.util.module_from_spec(triton_spec)
    sys.modules[triton_name] = triton_mod
    triton_spec.loader.exec_module(triton_mod)

    model_name = f"{pkg_name}.model"
    model_spec = importlib.util.spec_from_file_location(model_name, model_path)
    if model_spec is None or model_spec.loader is None:
        raise ImportError(f"cannot load {model_path}")
    model_mod = importlib.util.module_from_spec(model_spec)
    model_mod.__package__ = pkg_name
    sys.modules[model_name] = model_mod
    model_spec.loader.exec_module(model_mod)

    if not hasattr(model_mod, "HGSS") or not hasattr(model_mod, "HGSSConfig"):
        raise ImportError(f"{model_path} does not expose HGSS / HGSSConfig")
    return model_mod.HGSS, model_mod.HGSSConfig


def estimate_hgss_params(
    vocab_size: int,
    dim: int,
    layers: int,
    heads: int,
    d_k: int,
    d_v: int,
    d_conv: int,
    full_quaternion_read: bool = False,
) -> int:
    # Exact for the user's current model.py layout with tied embedding/head.
    H, K, V, D, C = heads, d_k, d_v, dim, d_conv
    proj_out = 12 * H * K + 4 * H * V
    read_mult = 4 if full_quaternion_read else 1

    per_block = 0
    per_block += D                             # RMSNorm
    per_block += D * proj_out                  # proj
    per_block += H * K                         # decay
    per_block += H                             # key_scale
    per_block += D * C + D                     # depthwise conv + bias
    per_block += D * (H * V)                   # gate
    per_block += D * (H * V * read_mult)       # out

    base = vocab_size * D + D                   # tied emb/head + final norm
    return base + layers * per_block


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def cosine_lr(
    step: int,
    max_steps: int,
    warmup_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if max_steps <= warmup_steps:
        return min_lr
    ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    ratio = min(max(ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (max_lr - min_lr)


def forward_loss_checkpointed(
    model,
    ids: torch.Tensor,
    use_block_checkpoint: bool,
    checkpoint_stride: int = 1,
):
    x = model.emb(ids)
    reset_mask = (
        (ids == model.cfg.boundary_token_id)
        if model.cfg.boundary_token_id is not None
        else None
    )

    checkpoint_stride = max(1, int(checkpoint_stride))
    for block_index, block in enumerate(model.blocks):
        checkpoint_this_block = (
            use_block_checkpoint
            and torch.is_grad_enabled()
            and block_index % checkpoint_stride == 0
        )
        if checkpoint_this_block:
            def block_fn(t, block=block):
                return block(t, reset_mask=reset_mask)

            x = checkpoint(
                block_fn,
                x,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            x = block(x, reset_mask=reset_mask)

    logits = model.head(model.norm(x))

    # Shuffle-safe local next-token objective:
    # use all within-window transitions and ignore the last prediction.
    targets = torch.empty_like(ids)
    targets[:, :-1] = ids[:, 1:]
    targets[:, -1] = -100

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
    )
    return loss


def save_checkpoint(
    out_dir: Path,
    model,
    optimizer,
    opt_step: int,
    tokens_seen: int,
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"checkpoint_step_{opt_step:07d}.pt"

    # argparse also stores the selected command handler in ``func``. Pickling
    # that function ties the checkpoint to a script running as __main__, which
    # prevents inference tools from loading it normally.
    checkpoint_args = {
        key: value for key, value in vars(args).items() if not callable(value)
    }
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "opt_step": opt_step,
        "tokens_seen": tokens_seen,
        "args": checkpoint_args,
    }
    torch.save(state, path)

    latest = out_dir / "latest.txt"
    latest.write_text(path.name + "\n", encoding="utf-8")
    print(f"saved: {path}")


def load_checkpoint(path: Path, model, optimizer) -> Tuple[int, int]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("opt_step", 0)), int(ckpt.get("tokens_seen", 0))


def cmd_train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the Triton HGSS backend")

    data_dir = Path(args.data_dir).resolve()
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. Run the 'download' command first."
        )
    manifest = load_manifest(manifest_path)

    HGSS, HGSSConfig = load_user_hgss(args.model_file, args.triton_file)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    use_bf16 = torch.cuda.is_bf16_supported() and not args.force_fp16
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    precision_name = "bf16" if use_bf16 else "fp16"

    estimated = estimate_hgss_params(
        vocab_size=args.vocab_size,
        dim=args.dim,
        layers=args.layers,
        heads=args.heads,
        d_k=args.d_k,
        d_v=args.d_v,
        d_conv=args.d_conv,
        full_quaternion_read=True,
    )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"precision: {precision_name}")
    print(f"estimated parameters: {estimated:,} ({estimated / 1e9:.3f}B)")
    print(
        f"HGSS config: D={args.dim}, L={args.layers}, H={args.heads}, "
        f"K={args.d_k}, V={args.d_v}, seq={args.seq_len}"
    )

    cfg = HGSSConfig(
        vocab_size=args.vocab_size,
        dim=args.dim,
        layers=args.layers,
        heads=args.heads,
        d_k=args.d_k,
        d_v=args.d_v,
        d_conv=args.d_conv,
        memory_min=args.memory_min,
        memory_max=args.memory_max,
        boundary_token_id=args.boundary_token_id,
        scan_backend="triton",
        scan_chunk_size=args.scan_chunk_size,
        scan_checkpoint=True,
        scan_compile=False,
        scan_triton=True,
        full_quaternion_read=True,
    )

    # Initialize directly on CUDA in FP32 so the model's decay initialization
    # keeps good precision, then cast trainable weights to BF16/FP16.
    with torch.device("cuda"):
        model = HGSS(cfg)
    model = model.to(dtype=amp_dtype)
    model.train()

    actual = sum(p.numel() for p in model.parameters())
    print(f"actual trainable parameters: {actual:,} ({actual / 1e9:.3f}B)")

    try:
        import bitsandbytes as bnb
    except ImportError as exc:
        raise RuntimeError(
            "bitsandbytes is required for the default 8-bit AdamW optimizer. "
            "Install it with: pip install -U bitsandbytes"
        ) from exc

    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
        min_8bit_size=4096,
    )

    stream = MixedTokenStream(
        data_dir=data_dir,
        manifest=manifest,
        seed=args.seed,
        seq_len=args.seq_len,
        code_fraction=args.code_fraction,
    )

    tokens_per_micro = args.micro_batch * args.seq_len
    tokens_per_opt = tokens_per_micro * args.grad_accum
    max_steps = math.ceil(args.tokens / tokens_per_opt)

    print(
        f"micro batch={args.micro_batch}, grad_accum={args.grad_accum}, "
        f"effective tokens/optimizer-step={tokens_per_opt:,}"
    )
    print(
        f"token budget={args.tokens:,} ({args.tokens/1e9:.3f}B), "
        f"planned optimizer steps~={max_steps:,}"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(amp_dtype == torch.float16),
    )

    opt_step = 0
    tokens_seen = 0
    if args.resume:
        resume_path = Path(args.resume).resolve()
        opt_step, tokens_seen = load_checkpoint(resume_path, model, optimizer)
        print(
            f"resumed {resume_path}: step={opt_step:,}, "
            f"tokens={tokens_seen:,}"
        )

    optimizer.zero_grad(set_to_none=True)
    out_dir = Path(args.out_dir).resolve()

    start_time = time.time()
    last_log_time = start_time
    last_log_tokens = tokens_seen
    source_counts: Dict[str, int] = {name: 0 for name in SOURCES}

    while tokens_seen < args.tokens:
        current_lr = cosine_lr(
            opt_step,
            max_steps,
            args.warmup_steps,
            args.lr,
            args.min_lr,
        )
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        loss_acc = 0.0
        micros_done = 0

        for _ in range(args.grad_accum):
            batches = []
            batch_sources = []
            for _b in range(args.micro_batch):
                source, arr = stream.next_ids()
                batches.append(arr)
                batch_sources.append(source)

            ids_np = np.stack(batches, axis=0)
            ids = torch.from_numpy(ids_np).to(
                device="cuda",
                dtype=torch.long,
                non_blocking=False,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=True,
            ):
                loss = forward_loss_checkpointed(
                    model,
                    ids,
                    use_block_checkpoint=not args.no_block_checkpoint,
                    checkpoint_stride=args.checkpoint_stride,
                )
                scaled_loss = loss / args.grad_accum

            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            loss_acc += float(loss.detach())
            micros_done += 1
            used = int(ids.numel())
            tokens_seen += used

            per_sample = args.seq_len
            for source in batch_sources:
                source_counts[source] += per_sample

            del ids, loss, scaled_loss

            if tokens_seen >= args.tokens:
                # Finish this partial accumulation with correct gradient scale.
                # We already divided by full grad_accum; compensate below.
                break

        if micros_done < args.grad_accum:
            correction = args.grad_accum / max(1, micros_done)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(correction)

        if scaler.is_enabled():
            scaler.unscale_(optimizer)

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        opt_step += 1

        if opt_step % args.log_every == 0 or tokens_seen >= args.tokens:
            now = time.time()
            dt = max(now - last_log_time, 1e-6)
            dtok = tokens_seen - last_log_tokens
            tok_s = dtok / dt
            avg_loss = loss_acc / max(1, micros_done)
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            print(
                f"step {opt_step:7d}/{max_steps:7d} | "
                f"tokens {tokens_seen/1e9:7.4f}B | "
                f"loss {avg_loss:.4f} | lr {current_lr:.3e} | "
                f"{tok_s:,.0f} tok/s | "
                f"VRAM {allocated:.2f}G alloc {reserved:.2f}G reserved"
            )
            last_log_time = now
            last_log_tokens = tokens_seen

        if (
            args.save_every > 0
            and opt_step % args.save_every == 0
            and tokens_seen < args.tokens
        ):
            save_checkpoint(
                out_dir,
                model,
                optimizer,
                opt_step,
                tokens_seen,
                args,
            )

    save_checkpoint(
        out_dir,
        model,
        optimizer,
        opt_step,
        tokens_seen,
        args,
    )

    elapsed = time.time() - start_time
    print()
    print(f"done: {tokens_seen:,} tokens in {elapsed/3600:.2f} h")
    total_counted = sum(source_counts.values())
    if total_counted:
        print("observed source mix:")
        for source, count in sorted(source_counts.items()):
            if count:
                print(
                    f"  {source:14s} {count/total_counted:7.2%} "
                    f"({format_tokens(count)})"
                )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download tokenized Crystal shards and train the user's HGSS."
    )
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="download already-tokenized HDF5 shards")
    d.add_argument("--data-dir", default="./crystal10b")
    d.add_argument("--tokens", type=int, default=10_000_000_000)
    d.add_argument(
        "--code-fraction",
        type=float,
        default=0.30,
        help="StarCoder fraction; remaining tokens are SlimPajama text",
    )
    d.set_defaults(func=cmd_download)

    t = sub.add_parser("train", help="train HGSS on downloaded shards")
    t.add_argument("--data-dir", default="./crystal10b")
    t.add_argument("--model-file", required=True)
    t.add_argument("--triton-file", required=True)
    t.add_argument("--out-dir", default="./hgss1b_run")
    t.add_argument("--resume", default=None)

    # Token budget / data mix.
    t.add_argument("--tokens", type=int, default=10_000_000_000)
    t.add_argument("--code-fraction", type=float, default=0.30)
    t.add_argument("--seq-len", type=int, default=2048)
    t.add_argument("--vocab-size", type=int, default=32032)
    t.add_argument("--boundary-token-id", type=int, default=2)

    # ~0.972B parameters with the user's scalar-read architecture.
    t.add_argument("--dim", type=int, default=2048)
    t.add_argument("--layers", type=int, default=18)
    t.add_argument("--heads", type=int, default=16)
    t.add_argument("--d-k", type=int, default=64)
    t.add_argument("--d-v", type=int, default=128)
    t.add_argument("--d-conv", type=int, default=4)
    t.add_argument("--memory-min", type=float, default=4.0)
    t.add_argument("--memory-max", type=float, default=32768.0)
    t.add_argument(
        "--scan-chunk-size",
        type=int,
        default=0,
        help="0 = use automatic chunk size from the user's Triton kernel",
    )

    # 16 GB-oriented defaults.
    t.add_argument("--micro-batch", type=int, default=1)
    t.add_argument("--grad-accum", type=int, default=64)
    t.add_argument("--no-block-checkpoint", action="store_true")
    t.add_argument("--force-fp16", action="store_true")

    # Optimizer / schedule.
    t.add_argument("--lr", type=float, default=2.0e-4)
    t.add_argument("--min-lr", type=float, default=2.0e-5)
    t.add_argument("--warmup-steps", type=int, default=1000)
    t.add_argument("--beta1", type=float, default=0.9)
    t.add_argument("--beta2", type=float, default=0.95)
    t.add_argument("--eps", type=float, default=1.0e-8)
    t.add_argument("--weight-decay", type=float, default=0.1)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument(
        "--checkpoint-stride",
        type=int,
        default=1,
        help=(
            "activation-checkpoint every Nth block (1=all blocks; 2=alternating); "
            "ignored with --no-block-checkpoint"
        ),
    )

    t.add_argument("--seed", type=int, default=1337)
    t.add_argument("--log-every", type=int, default=10)
    t.add_argument(
        "--save-every",
        type=int,
        default=1000,
        help="optimizer steps; checkpoint includes 8-bit optimizer state",
    )
    t.set_defaults(func=cmd_train)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
