"""Decoder-only GPT for next-token prediction and autoregressive generation."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 4096
    context_length: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 192

    def __post_init__(self):
        sizes = (self.vocab_size, self.context_length, self.n_layers, self.n_heads, self.d_model)
        if any(size <= 0 for size in sizes):
            raise ValueError("All GPT dimensions must be positive.")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads.")


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)
        mask = torch.ones(config.context_length, config.context_length, dtype=torch.bool).tril()
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x):
        batch_size, length, width = x.shape
        projected = self.qkv(x).reshape(batch_size, length, 3, self.n_heads, self.head_dim)
        query, key, value = projected.permute(2, 0, 3, 1, 4).unbind(0)

        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.causal_mask[:length, :length], float("-inf"))
        weights = scores.softmax(dim=-1)
        combined = (weights @ value).transpose(1, 2).reshape(batch_size, length, width)
        return self.output(combined)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.d_model
        self.attention_norm = nn.LayerNorm(width)
        self.attention = CausalSelfAttention(config)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.feedforward(self.feedforward_norm(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.blocks = nn.Sequential(*(TransformerBlock(config) for _ in range(config.n_layers)))
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._initialize)
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def _hidden(self, tokens):
        if (
            tokens.ndim != 2
            or tokens.shape[0] == 0
            or not 1 <= tokens.shape[1] <= self.config.context_length
        ):
            raise ValueError("Expected [batch, time] tokens within the context length.")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        return self.final_norm(self.blocks(hidden))

    def next_token_logits(self, tokens, last_positions):
        """Project only the final prompt position of each row to the vocabulary."""
        hidden = self._hidden(tokens)
        rows = torch.arange(tokens.shape[0], device=tokens.device)
        return self.lm_head(hidden[rows, last_positions])

    def forward(self, tokens, targets=None):
        """Return logits and the optional loss for already-shifted targets."""
        if targets is not None and targets.shape != tokens.shape:
            raise ValueError("Targets must match the shape of the input tokens.")
        logits = self.lm_head(self._hidden(tokens))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens, eos_id=None, *, temperature=0.0, top_k=None, generator=None):
        """Extend a single prompt greedily or by sampling, stopping at EOS or the context limit."""
        if prompt.ndim != 1 or not 1 <= prompt.numel() <= self.config.context_length:
            raise ValueError("Expected a nonempty 1-D prompt within the context length.")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative.")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative.")
        if top_k is not None and not 1 <= top_k <= self.config.vocab_size:
            raise ValueError("top_k must be within the vocabulary size.")

        tokens = prompt
        remaining = self.config.context_length - prompt.numel()
        for _ in range(min(max_new_tokens, remaining)):
            logits, _ = self(tokens.unsqueeze(0))
            scores = logits[0, -1]
            if temperature == 0:
                next_token = scores.argmax().reshape(1)
            elif top_k is None:
                probabilities = (scores / temperature).softmax(dim=-1)
                next_token = torch.multinomial(probabilities, 1, generator=generator)
            else:
                values, indices = scores.topk(top_k)
                probabilities = (values / temperature).softmax(dim=-1)
                next_token = indices[torch.multinomial(probabilities, 1, generator=generator)]
            tokens = torch.cat((tokens, next_token))
            if eos_id is not None and next_token.item() == eos_id:
                break
        return tokens
