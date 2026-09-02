#!/usr/bin/env python3
"""Local Gradio UI for HGSS checkpoints."""

from __future__ import annotations

import argparse
import threading
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch


CRYSTAL_FIM_TOKENS = [
    "<fim_prefix>",
    "<fim_middle>",
    "<fim_suffix>",
    "<fim_pad>",
    "<filename>",
    "<gh_stars>",
    "<issue_start>",
    "<issue_comment>",
    "<issue_closed>",
    "<jupyter_start>",
    "<jupyter_text>",
    "<jupyter_code>",
    "<jupyter_output>",
    "<empty_output>",
    "<commit_before>",
    "<commit_msg>",
    "<commit_after>",
    "<reponame>",
]


# Compatibility for checkpoints produced before callable argparse values were
# excluded. Those files contain a pickle reference to ``__main__.cmd_train``.
def cmd_train(*_args, **_kwargs):
    raise RuntimeError("training handlers cannot be invoked while serving")


def find_checkpoint(run_dir: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return path

    latest = run_dir / "latest.txt"
    if latest.is_file():
        name = latest.read_text(encoding="utf-8").strip()
        path = run_dir / name
        if path.is_file():
            return path.resolve()

    candidates = sorted(run_dir.glob("checkpoint_step_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint found in {run_dir.resolve()}")
    return candidates[-1].resolve()


def load_checkpoint_file(path: Path) -> dict[str, Any]:
    # weights_only=False is required for legacy checkpoints containing an
    # argparse.Namespace. Only load checkpoints created locally/trusted by you.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"unsupported checkpoint format: {path}")
    return checkpoint


def make_model(checkpoint: dict[str, Any]):
    from hgss.model import HGSS, HGSSConfig

    saved_args = checkpoint.get("args", {})
    if isinstance(saved_args, argparse.Namespace):
        saved_args = vars(saved_args)
    if not isinstance(saved_args, dict):
        raise ValueError("checkpoint args must be a dictionary or Namespace")

    required = ("vocab_size", "dim", "layers", "heads", "d_k", "d_v", "d_conv")
    missing = [name for name in required if name not in saved_args]
    if missing:
        raise ValueError(f"checkpoint lacks model settings: {', '.join(missing)}")

    state_dict = checkpoint["model"]
    out_weight = state_dict.get("blocks.0.out.weight")
    scalar_width = int(saved_args["heads"]) * int(saved_args["d_v"])
    if out_weight is None:
        raise ValueError("checkpoint lacks blocks.0.out.weight")
    if out_weight.shape[1] == scalar_width * 4:
        full_read = True
    elif out_weight.shape[1] == scalar_width:
        full_read = False
    else:
        raise ValueError(f"cannot infer read mode from out width {out_weight.shape[1]}")

    valid_names = {field.name for field in fields(HGSSConfig)}
    config_values = {key: value for key, value in saved_args.items() if key in valid_names}
    config_values.update(
        scan_backend="triton",
        scan_checkpoint=False,
        scan_compile=False,
        scan_triton=True,
        full_quaternion_read=full_read,
    )
    config = HGSSConfig(**config_values)

    with torch.device("cuda"):
        model = HGSS(config)
    model = model.to(dtype=torch.bfloat16)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, config, full_read


def format_chat_prompt(message: str, history: list[Any], include_history: bool) -> str:
    if not include_history:
        return message

    parts: list[str] = []
    for item in history or []:
        if isinstance(item, dict):
            role, content = item.get("role"), item.get("content")
            if role in ("user", "assistant") and isinstance(content, str):
                label = "User" if role == "user" else "Assistant"
                parts.append(f"{label}: {content}")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            if item[0]:
                parts.append(f"User: {item[0]}")
            if item[1]:
                parts.append(f"Assistant: {item[1]}")
    parts.append(f"User: {message}")
    parts.append("Assistant:")
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="./hgss100m_run")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tokenizer", default="./crystal10b/tokenizer.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    try:
        import gradio as gr
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing UI dependencies. Run: pip install gradio tokenizers"
        ) from exc

    run_dir = Path(args.run_dir).expanduser().resolve()
    checkpoint_path = find_checkpoint(run_dir, args.checkpoint)
    tokenizer_path = Path(args.tokenizer).expanduser().resolve()
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"tokenizer not found: {tokenizer_path}")

    print(f"Loading checkpoint: {checkpoint_path}", flush=True)
    checkpoint = load_checkpoint_file(checkpoint_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    # CrystalCoder's text shards use the base 32k LLaMA tokenizer. Its
    # StarCoder preprocessing appends these 18 tokens in this exact order.
    # The distributed tokenizer.json contains only the base vocabulary.
    tokenizer.add_special_tokens(CRYSTAL_FIM_TOKENS)
    fim_ids = [tokenizer.token_to_id(token) for token in CRYSTAL_FIM_TOKENS]
    expected_fim_ids = list(range(32000, 32018))
    if fim_ids != expected_fim_ids:
        raise ValueError(
            f"unexpected Crystal FIM token IDs: {fim_ids}; expected {expected_fim_ids}"
        )
    model, config, full_read = make_model(checkpoint)
    if tokenizer.get_vocab_size() > config.vocab_size:
        raise ValueError(
            f"tokenizer has {tokenizer.get_vocab_size()} tokens but model has "
            f"only {config.vocab_size} embeddings"
        )
    lock = threading.Lock()
    max_context = int(checkpoint.get("args", {}).get("seq_len", 2048))
    eos_id = 2

    def respond(message, history, max_new_tokens, temperature, top_k, include_history):
        prompt = format_chat_prompt(str(message), history, bool(include_history))
        token_ids = tokenizer.encode(prompt, add_special_tokens=True).ids
        if len(token_ids) > max_context:
            # Preserve BOS while retaining the newest prompt tokens.
            token_ids = [token_ids[0], *token_ids[-(max_context - 1):]]
        ids = torch.tensor([token_ids], dtype=torch.long, device="cuda")

        with lock, torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            output = model.generate(
                ids,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_k=int(top_k),
                use_cuda_graph=False,
            )
        generated = output[0, ids.size(1):].tolist()
        if eos_id in generated:
            generated = generated[:generated.index(eos_id)]
        # 0/1 are UNK/BOS and 32018..32031 are embedding padding slots, not
        # decodable vocabulary. FIM controls 32000..32017 are skipped by the
        # tokenizer because they were registered as special tokens above.
        generated = [token for token in generated if token not in (0, 1) and token < 32018]
        return tokenizer.decode(generated, skip_special_tokens=True)

    step = int(checkpoint.get("opt_step", 0))
    tokens_seen = int(checkpoint.get("tokens_seen", 0))
    mode = "full quaternion" if full_read else "scalar"
    description = (
        f"Checkpoint: `{checkpoint_path.name}` · step {step:,} · "
        f"{tokens_seen / 1e6:.1f}M training tokens · {mode} read · "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters. "
        "This is a base next-token model, not an instruction-tuned assistant."
    )

    chatbot = gr.Chatbot(height=520)
    app = gr.ChatInterface(
        fn=respond,
        chatbot=chatbot,
        title="HGSS local model",
        description=description,
        additional_inputs=[
            gr.Slider(1, 512, value=128, step=1, label="Max new tokens"),
            gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="Temperature"),
            gr.Slider(0, 200, value=40, step=1, label="Top-k (0 = disabled)"),
            gr.Checkbox(value=False, label="Format and include chat history"),
        ],
        cache_examples=False,
    )
    app.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
