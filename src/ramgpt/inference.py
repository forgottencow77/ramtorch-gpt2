from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .config import TrainingConfig
from .model import GPT2RamLMHead
from .utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a RamGPT checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, world",
        help="Prompt text",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Optional tokenizer directory (defaults to checkpoint/tokenizer)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    state = load_checkpoint(checkpoint_path, map_location=device)
    cfg_dict = state.get("config")
    if cfg_dict is None:
        raise RuntimeError("Checkpoint missing config. Re-run training with a newer script.")
    cfg = TrainingConfig.from_dict(cfg_dict)

    model = GPT2RamLMHead(cfg.model)
    model.load_state_dict(state["model_state"], strict=True)
    model.to(device)
    model.eval()

    tokenizer_dir = (
        Path(args.tokenizer)
        if args.tokenizer
        else checkpoint_path.parent / "tokenizer"
    )
    if tokenizer_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(
        args.prompt,
        return_tensors="pt",
        return_attention_mask=False,
    )
    input_ids = encoded["input_ids"].to(device)
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    completion = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(completion)


if __name__ == "__main__":
    main()
