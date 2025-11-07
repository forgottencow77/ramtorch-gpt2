from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

try:  # optional dependency
    import wandb
except ImportError:  # pragma: no cover - optional
    wandb = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def human_readable_params(num: int) -> str:
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    if num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    if num >= 1000:
        return f"{num / 1000:.2f}K"
    return str(num)


def get_amp_dtype(cfg, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if cfg.bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if cfg.fp16:
        return torch.float16
    return torch.float32


def maybe_init_wandb(cfg, config_dict: Dict[str, Any]) -> Optional[Any]:
    if not cfg.logging.use_wandb:
        return None
    if wandb is None:
        raise RuntimeError("wandb is not installed but use_wandb=True")
    run = wandb.init(
        project=cfg.logging.wandb_project,
        name=cfg.logging.wandb_run_name,
        config=config_dict,
        mode="offline" if os.environ.get("WANDB_MODE") == "offline" else None,
    )
    if cfg.logging.wandb_watch:
        wandb.watch_called = False
    return run


def log_metrics(run, metrics: Dict[str, Any], step: int) -> None:
    if wandb is not None and run is not None:
        wandb.log(metrics, step=step)


def save_checkpoint(state: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def rotate_checkpoints(ckpt_dir: Path, keep_last: int) -> None:
    checkpoints = sorted(ckpt_dir.glob("step_*.pt"))
    if keep_last <= 0:
        return
    for extra in checkpoints[:-keep_last]:
        extra.unlink(missing_ok=True)


def maybe_push_to_hf(folder: Path, repo_id: Optional[str], token: Optional[str], message: str) -> None:
    if not repo_id:
        return
    from huggingface_hub import HfApi

    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(folder),
        repo_type="model",
        commit_message=message,
        allow_patterns=["*.pt", "*.safetensors", "*.json", "*.md", "*.yaml"],
    )


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


def cosine_lr(base_lr: float, min_lr_ratio: float, warmup_steps: int, max_steps: int, step: int) -> float:
    if step < warmup_steps:
        return base_lr * (step / max(1, warmup_steps))
    progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
    min_lr = base_lr * min_lr_ratio
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
