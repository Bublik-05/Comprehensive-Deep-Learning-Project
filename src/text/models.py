from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128,
                 num_layers: int = 1, dropout: float = 0.3, rnn_type: str = "lstm",
                 num_classes: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)
        rnn_cls = nn.LSTM if rnn_type.lower() == "lstm" else nn.GRU
        # Note: dropout inside RNN applies between layers, so only effective if num_layers>1
        self.rnn = rnn_cls(embed_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
                           bidirectional=False)
        self.out_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # x: (B, T)
        emb = self.embed_dropout(self.embedding(x))  # (B, T, D)
        out, _ = self.rnn(emb)  # (B, T, H)
        # take last timestep (works because we pad to fixed length)
        last = out[:, -1, :]
        logits = self.fc(self.out_dropout(last))
        return logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        # x: (B, T, D)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerClassifier(nn.Module):
    """Encoder-only Transformer classifier (no pretrained weights)."""
    def __init__(self, vocab_size: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 3, dim_feedforward: int = 128, dropout: float = 0.1,
                 num_classes: int = 2, max_len: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # x: (B, T), mask: (B, T) 1 for tokens, 0 for pad
        emb = self.embedding(x)
        emb = self.pos(emb)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask == 0)  # True for pads
        h = self.encoder(emb, src_key_padding_mask=key_padding_mask)  # (B, T, D)
        if mask is None:
            pooled = h.mean(dim=1)
        else:
            m = mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(dim=1) / (m.sum(dim=1).clamp_min(1.0))
        logits = self.fc(self.dropout(pooled))
        return logits
