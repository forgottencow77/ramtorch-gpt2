from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint as activation_checkpoint

try:
    from ramtorch.modules import Linear as RamLinear
except Exception:  # pragma: no cover - fallback when ramtorch unavailable
    RamLinear = None

from .config import ModelConfig


def _make_linear(in_features: int, out_features: int, cfg: ModelConfig, bias: bool = True) -> nn.Module:
    use_ramtorch = RamLinear is not None and torch.cuda.is_available()
    if use_ramtorch:
        return RamLinear(
            in_features,
            out_features,
            bias=bias,
            device=cfg.ram_offload_device,
            dtype=torch.float32,
        )
    layer = nn.Linear(in_features, out_features, bias=bias)
    return layer


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "Embedding dim must be divisible by heads"
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn = _make_linear(cfg.n_embd, 3 * cfg.n_embd, cfg)
        self.c_proj = _make_linear(cfg.n_embd, cfg.n_embd, cfg)
        self.attn_dropout = nn.Dropout(cfg.attn_pdrop)
        self.resid_dropout = nn.Dropout(cfg.resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.cfg.use_flash_attn and hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            attn_weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            mask = torch.tril(
                torch.ones((T, T), device=attn_weights.device, dtype=torch.bool)
            )
            attn_weights = attn_weights.masked_fill(~mask, float("-inf"))
            attn_probs = F.softmax(attn_weights, dim=-1)
            attn_probs = self.attn_dropout(attn_probs)
            y = attn_probs @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        inner_dim = 4 * cfg.n_embd
        self.c_fc = _make_linear(cfg.n_embd, inner_dim, cfg)
        self.act = nn.GELU()
        self.c_proj = _make_linear(inner_dim, cfg.n_embd, cfg)
        self.dropout = nn.Dropout(cfg.resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2RamLMHead(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.max_position_embeddings, cfg.n_embd)
        self.drop = nn.Dropout(cfg.embd_pdrop)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.gradient_checkpointing = cfg.gradient_checkpointing
        self.apply(self._init_weights)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        if T > self.cfg.max_position_embeddings:
            raise ValueError(
                f"Sequence length {T} exceeds max_position_embeddings={self.cfg.max_position_embeddings}"
            )
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        pos = pos.expand(B, T)
        hidden_states = self.wte(input_ids) + self.wpe(pos)
        hidden_states = self.drop(hidden_states)

        for block in self.blocks:
            if (
                self.gradient_checkpointing
                and self.training
                and hidden_states.requires_grad
            ):
                hidden_states = activation_checkpoint.checkpoint(
                    block,
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(hidden_states)

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
        if RamLinear is not None and isinstance(module, RamLinear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = 0,
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids
        for _ in range(max_new_tokens):
            logits = self(generated)
            next_token_logits = logits[:, -1, :]
            next_token_logits = next_token_logits / max(temperature, 1e-4)
            if top_k and top_k > 0:
                values, indices = torch.topk(next_token_logits, top_k)
                mask = torch.full_like(next_token_logits, float("-inf"))
                mask.scatter_(1, indices, values)
                next_token_logits = mask
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if generated.size(1) >= self.cfg.max_position_embeddings:
                break
        return generated
