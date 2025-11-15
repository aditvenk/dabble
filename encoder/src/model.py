# model.py

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# -------------------------
# Config
# -------------------------


@dataclass
class TransformerConfig:
    vocab_size: int = 5000
    max_len: int = 128
    d_model: int = 256
    n_heads: int = 4
    num_layers: int = 4
    dim_feedforward: int = 512
    num_classes: int = 10
    dropout: float = 0.1
    use_checkpoint: bool = (
        False  # activation checkpointing over encoder layers
    )


# -------------------------
# Building blocks
# -------------------------


class PositionalEmbedding(nn.Module):
    """Learned positional embeddings: (B, L) -> (B, L, D)."""

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len) token ids (only used for shape)
        returns: (batch, seq_len, d_model)
        """
        bsz, seq_len = x.size()
        device = x.device
        positions = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        )
        return self.pos_embed(positions)


# -------------------------
# Encoder-only Transformer classifier
# -------------------------


class TransformerClassifier(nn.Module):
    """
    Encoder-only Transformer for sequence classification.

    - Input:  (B, L) integer token ids
    - Output: (B, C) log-probabilities over classes
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = PositionalEmbedding(cfg.max_len, cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,  # (B, L, D)
        )

        # We keep layers explicitly (instead of nn.TransformerEncoder)
        # so we can apply checkpoint() per layer.
        self.layers = nn.ModuleList(
            [encoder_layer for _ in range(cfg.num_layers)]
        )

        self.dropout = nn.Dropout(cfg.dropout)
        self.fc_out = nn.Linear(cfg.d_model, cfg.num_classes)

        self.use_checkpoint = cfg.use_checkpoint

    def _encode_layer(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Helper to run a single encoder layer, optionally under checkpoint.
        src_key_padding_mask: (B, L) bool, True = pad.
        """

        if self.use_checkpoint and self.training:
            # checkpoint doesn't support kwargs, so wrap layer call.
            def layer_fn(t, mask):
                return layer(t, src_key_padding_mask=mask)

            # use_reentrant=False is the modern recommended behavior.
            return checkpoint(
                layer_fn, x, src_key_padding_mask, use_reentrant=False
            )
        else:
            return layer(x, src_key_padding_mask=src_key_padding_mask)

    def forward(
        self,
        tokens: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        train: bool = True,
    ) -> torch.Tensor:
        """
        tokens: (B, L) int64 token ids
        src_key_padding_mask: (B, L) bool, True = pad position
        returns: (B, C) log-probabilities
        """
        # Embeddings
        x_tok = self.token_embed(tokens)  # (B, L, D)
        x_pos = self.pos_embed(tokens)  # (B, L, D)
        x = x_tok + x_pos  # (B, L, D)

        # Encoder stack
        for layer in self.layers:
            x = self._encode_layer(layer, x, src_key_padding_mask)

        # Pool over sequence to get a single representation per example
        if src_key_padding_mask is not None:
            # mask: True = pad, False = real token
            mask = ~src_key_padding_mask  # (B, L)
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)

            # Mean pooling only over the non-pad tokens
            x = (x * mask.unsqueeze(-1)).sum(dim=1) / lengths  # (B, D)
        else:
            x = x.mean(dim=1)  # (B, D)

        if train:
            x = self.dropout(x)

        logits = self.fc_out(x)  # (B, C)
        return F.log_softmax(logits.float(), dim=-1)


# -------------------------
# Convenience builder
# -------------------------


def build_transformer_classifier(
    vocab_size: int,
    num_classes: int,
    max_len: int = 128,
    d_model: int = 256,
    n_heads: int = 4,
    num_layers: int = 4,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
    use_checkpoint: bool = False,
) -> TransformerClassifier:
    cfg = TransformerConfig(
        vocab_size=vocab_size,
        max_len=max_len,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        num_classes=num_classes,
        dropout=dropout,
        use_checkpoint=use_checkpoint,
    )
    return TransformerClassifier(cfg)
