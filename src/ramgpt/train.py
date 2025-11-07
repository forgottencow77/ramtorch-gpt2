from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from tqdm import tqdm

from .config import TrainingConfig
from .data import create_dataloader
from .model import GPT2RamLMHead
from .utils import (
    cosine_lr,
    count_parameters,
    get_amp_dtype,
    human_readable_params,
    load_checkpoint,
    log_metrics,
    maybe_init_wandb,
    maybe_push_to_hf,
    rotate_checkpoints,
    save_checkpoint,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GPT-2 (~1B) with RamTorch")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_1b.yaml",
        help="Path to YAML config",
    )
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only")
    parser.add_argument(
        "--micro-test",
        action="store_true",
        help="Shrink the model for a quick CPU-friendly OOM smoke test",
    )
    return parser.parse_args()


def apply_micro_test_overrides(cfg: TrainingConfig) -> None:
    cfg.model.n_layer = min(cfg.model.n_layer, 2)
    cfg.model.n_head = min(cfg.model.n_head, 8)
    cfg.model.n_embd = min(cfg.model.n_embd, 512)
    cfg.data.seq_len = min(cfg.data.seq_len, 128)
    cfg.model.max_position_embeddings = max(cfg.data.seq_len, 128)
    cfg.data.micro_batch_size = min(cfg.data.micro_batch_size, 2)
    cfg.data.gradient_accumulation = 1
    cfg.scheduler.max_steps = min(cfg.scheduler.max_steps, 20)
    cfg.scheduler.warmup_steps = min(cfg.scheduler.warmup_steps, 5)
    cfg.logging.eval_interval = 5
    cfg.checkpoint.save_interval_steps = 10
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(model, loader, device, amp_dtype, steps: int = 20) -> float:
    model.eval()
    losses = []
    iterator = iter(loader)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    amp_enabled = device.type == "cuda" and amp_dtype != torch.float32
    with torch.no_grad():
        for _ in range(steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            input_ids = batch.to(device)
            with autocast(
                autocast_device,
                enabled=amp_enabled,
                dtype=amp_dtype if amp_enabled else None,
            ):
                logits = model(input_ids)
                shift_logits = logits[:, :-1, :]
                shift_labels = input_ids[:, 1:]
                loss = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                )
            losses.append(loss.item())
            if len(losses) >= steps:
                break
    model.train()
    return sum(losses) / max(1, len(losses))


def main() -> None:
    args = parse_args()
    cfg = TrainingConfig.from_yaml(args.config)
    if args.micro_test:
        apply_micro_test_overrides(cfg)

    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(cfg.checkpoint.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_loader, tokenizer = create_dataloader(cfg.data, cfg.data.train_split, cfg.seed)
    eval_loader = None
    try:
        eval_loader, _ = create_dataloader(cfg.data, cfg.data.eval_split, cfg.seed + 1)
    except Exception:
        eval_loader = None

    model = GPT2RamLMHead(cfg.model)
    model = model.to(device)
    total_params = count_parameters(model)
    print(f"Model params: {human_readable_params(total_params)} ({total_params:,})")

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optim.learning_rate,
        betas=(cfg.optim.beta1, cfg.optim.beta2),
        eps=cfg.optim.eps,
        weight_decay=cfg.optim.weight_decay,
    )
    amp_dtype = get_amp_dtype(cfg, device)
    amp_enabled = device.type == "cuda" and amp_dtype != torch.float32
    use_scaler = device.type == "cuda" and amp_dtype == torch.float16
    scaler = GradScaler(enabled=use_scaler)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    global_step = 0
    (ckpt_dir / "tokenizer").mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(ckpt_dir / "tokenizer")

    if args.resume or cfg.checkpoint.resume_path:
        ckpt_path = Path(args.resume or cfg.checkpoint.resume_path)
        if ckpt_path.exists():
            state = load_checkpoint(ckpt_path, map_location=device)
            model.load_state_dict(state["model_state"])
            optimizer.load_state_dict(state["optimizer_state"])
            if "scaler_state" in state:
                scaler.load_state_dict(state["scaler_state"])
            global_step = state.get("step", 0)
            print(f"Resumed from {ckpt_path} at step {global_step}")

    if args.eval_only:
        if eval_loader is None:
            raise RuntimeError("Evaluation split not available")
        eval_loss = evaluate(model, eval_loader, device, amp_dtype)
        print(f"Eval loss: {eval_loss:.4f}")
        return

    wandb_run = maybe_init_wandb(cfg, cfg.to_dict())

    train_iterator = iter(train_loader)
    grad_accum = cfg.data.gradient_accumulation
    tokens_per_step = cfg.data.seq_len * cfg.data.micro_batch_size * grad_accum
    pbar = tqdm(total=cfg.scheduler.max_steps, initial=global_step, desc="steps")

    while global_step < cfg.scheduler.max_steps:
        start_time = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for micro_step in range(grad_accum):
            batch, train_iterator = get_micro_batch(train_loader, train_iterator, device)
            with autocast(
                autocast_device,
                enabled=amp_enabled,
                dtype=amp_dtype if amp_enabled else None,
            ):
                logits = model(batch)
                shift_logits = logits[:, :-1, :]
                shift_labels = batch[:, 1:]
                loss = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                )
                loss = loss / grad_accum
            step_loss += loss.item()
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if cfg.optim.grad_clip > 0:
            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)

        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        global_step += 1
        pbar.update(1)
        current_lr = cosine_lr(
            cfg.optim.learning_rate,
            cfg.scheduler.min_lr_ratio,
            cfg.scheduler.warmup_steps,
            cfg.scheduler.max_steps,
            global_step,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        elapsed = time.perf_counter() - start_time
        metrics = {
            "train/loss": step_loss,
            "train/lr": current_lr,
            "speed/tokens_per_s": tokens_per_step / max(1e-6, elapsed),
        }
        if global_step % cfg.logging.log_interval == 0:
            print(
                f"step {global_step}: loss={step_loss:.4f} lr={current_lr:.6f} "
                f"tokens/s={metrics['speed/tokens_per_s']:.0f}"
            )
            log_metrics(wandb_run, metrics, global_step)

        if (
            eval_loader is not None
            and cfg.logging.eval_interval > 0
            and global_step % cfg.logging.eval_interval == 0
        ):
            eval_loss = evaluate(model, eval_loader, device, amp_dtype)
            log_metrics(wandb_run, {"eval/loss": eval_loss}, global_step)
            print(f"Eval loss @ step {global_step}: {eval_loss:.4f}")

        if global_step % cfg.checkpoint.save_interval_steps == 0:
            ckpt_path = ckpt_dir / f"step_{global_step:07d}.pt"
            save_checkpoint(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "step": global_step,
                    "config": cfg.to_dict(),
                },
                ckpt_path,
            )
            rotate_checkpoints(ckpt_dir, cfg.checkpoint.keep_last_n)
            maybe_push_to_hf(
                ckpt_dir,
                cfg.checkpoint.hf_repo_id,
                cfg.checkpoint.hf_token,
                message=f"Step {global_step}",
            )

    pbar.close()
    if wandb_run is not None:
        wandb_run.finish()


def get_micro_batch(loader, iterator, device):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch.to(device), iterator


if __name__ == "__main__":
    main()
