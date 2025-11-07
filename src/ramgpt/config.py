from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class OptimizerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0


@dataclass
class SchedulerConfig:
    warmup_steps: int = 2000
    max_steps: int = 600_000
    min_lr_ratio: float = 0.1


@dataclass
class DataConfig:
    dataset_name: Optional[str] = "wikitext"
    dataset_config: Optional[str] = "wikitext-103-raw-v1"
    dataset_path: Optional[str] = None
    text_column: str = "text"
    train_split: str = "train"
    eval_split: str = "validation"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    seq_len: int = 2048
    micro_batch_size: int = 1
    gradient_accumulation: int = 64
    num_workers: int = 2
    streaming: bool = True
    buffer_size: int = 4096
    pretokenized_dir: Optional[str] = None
    tokenizer_name: str = "gpt2"
    tokenizer_vocab_size: int = 50257
    tokenizer_pad_to_multiple_of: int = 128
    pack_sequences: bool = True


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    n_layer: int = 36
    n_head: int = 20
    n_embd: int = 3072
    max_position_embeddings: int = 2048
    rotary_pct: Optional[float] = None
    activation_fn: str = "gelu"
    resid_pdrop: float = 0.0
    embd_pdrop: float = 0.01
    attn_pdrop: float = 0.01
    layer_norm_epsilon: float = 1e-5
    use_flash_attn: bool = True
    gradient_checkpointing: bool = True
    ram_offload_device: str = "cuda"
    ram_cpu_buffer: int = 4
    dtype: str = "bfloat16"


@dataclass
class CheckpointConfig:
    output_dir: str = "checkpoints"
    save_interval_steps: int = 1000
    keep_last_n: int = 3
    resume_path: Optional[str] = None
    hf_repo_id: Optional[str] = None
    hf_token: Optional[str] = None
    push_to_hub_interval: int = 10_000


@dataclass
class LoggingConfig:
    log_interval: int = 10
    eval_interval: int = 1000
    use_wandb: bool = False
    wandb_project: str = "ramgpt"
    wandb_run_name: Optional[str] = None
    wandb_watch: bool = False


@dataclass
class TrainingConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 42
    device: str = "cuda"
    compile: bool = False
    bf16: bool = True
    fp16: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        raw = Path(path).read_text()
        data = yaml.safe_load(raw) or {}
        instance = cls()
        _apply_updates(instance, data)
        return instance

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        instance = cls()
        _apply_updates(instance, data)
        return instance


def _apply_updates(target: Any, updates: Dict[str, Any]) -> Any:
    for key, value in updates.items():
        if not hasattr(target, key):
            continue
        attr = getattr(target, key)
        if dataclasses.is_dataclass(attr) and isinstance(value, dict):
            _apply_updates(attr, value)
        else:
            setattr(target, key, value)
    return target
