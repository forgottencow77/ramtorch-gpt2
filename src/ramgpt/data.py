from __future__ import annotations

import itertools
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, Optional

import torch
from datasets import IterableDatasetDict, load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from .config import DataConfig


class PackedIterableDataset(IterableDataset):
    def __init__(
        self,
        iterator_fn: Callable[[], Iterable[Dict[str, str]]],
        tokenizer: PreTrainedTokenizer,
        seq_len: int,
        buffer_size: int,
        text_column: str,
    ) -> None:
        self.iterator_fn = iterator_fn
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.buffer_size = max(seq_len, buffer_size)
        self.text_column = text_column

    def __iter__(self) -> Iterator[torch.Tensor]:
        token_buffer: list[int] = []
        # re-create iterator each time DataLoader worker starts consuming
        for sample in self.iterator_fn():
            text = sample.get(self.text_column) if isinstance(sample, dict) else None
            if not text:
                continue
            encoded = self.tokenizer.encode(str(text), add_special_tokens=False)
            token_buffer.extend(encoded)
            token_buffer.append(self.tokenizer.eos_token_id)

            while len(token_buffer) >= self.seq_len:
                chunk = token_buffer[: self.seq_len]
                token_buffer = token_buffer[self.seq_len :]
                yield torch.tensor(chunk, dtype=torch.long)

            if len(token_buffer) > self.buffer_size:
                # prevent unbounded growth when encountering very long samples
                token_buffer = token_buffer[-self.seq_len :]


def build_tokenizer(cfg: DataConfig) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, use_fast=True)
    tokenizer.model_max_length = max(cfg.seq_len, 1_000_000)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "left"
    return tokenizer


def _line_iterator(path: Path, text_column: str) -> Iterator[Dict[str, str]]:
    for file in sorted(path.rglob("*.txt")):
        with file.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield {text_column: line}


def _dataset_iterator(cfg: DataConfig, split: str, seed: int):
    limit = None
    if split == cfg.train_split:
        limit = cfg.max_train_samples
    elif split == cfg.eval_split:
        limit = cfg.max_eval_samples

    if cfg.dataset_path:
        base = Path(cfg.dataset_path)
        if not base.exists():
            raise FileNotFoundError(f"Dataset path {base} does not exist")

        def iterator() -> Iterator[Dict[str, str]]:
            stream = _line_iterator(base, cfg.text_column)
            if limit is not None:
                stream = itertools.islice(stream, limit)
            return stream

        return iterator

    if not cfg.dataset_name:
        raise ValueError("Either dataset_name or dataset_path must be provided")

    dataset = load_dataset(
        cfg.dataset_name,
        cfg.dataset_config,
        split=split,
        streaming=cfg.streaming,
    )

    if not cfg.streaming:
        dataset = dataset.shuffle(seed=seed)
        if limit is not None:
            limit = min(limit, len(dataset))
            dataset = dataset.select(range(limit))

        def iterator() -> Iterator[Dict[str, str]]:
            yield from dataset

        return iterator

    def iterator() -> Iterator[Dict[str, str]]:
        iterable = dataset
        if limit is not None:
            iterable = itertools.islice(iterable, limit)
        yield from iterable

    return iterator


def create_dataloader(cfg: DataConfig, split: str, seed: int) -> tuple[DataLoader, PreTrainedTokenizer]:
    tokenizer = build_tokenizer(cfg)
    iterator_fn = _dataset_iterator(cfg, split, seed)
    dataset = PackedIterableDataset(
        iterator_fn=iterator_fn,
        tokenizer=tokenizer,
        seq_len=cfg.seq_len,
        buffer_size=cfg.buffer_size,
        text_column=cfg.text_column,
    )

    num_workers = 0 if cfg.streaming else max(0, cfg.num_workers)
    loader = DataLoader(
        dataset,
        batch_size=cfg.micro_batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, tokenizer
