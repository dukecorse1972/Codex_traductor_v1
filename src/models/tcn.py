from __future__ import annotations

import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.final_act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)
        out = self.drop1(self.act1(self.norm1(self.conv1(x))))
        out = self.drop2(self.act2(self.norm2(self.conv2(out))))
        return self.final_act(out + residual)


class TCNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 300,
        channels: tuple[int, ...] = (256, 256, 256, 256),
        kernel_size: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        blocks = []
        in_ch = input_dim
        for i, ch in enumerate(channels):
            blocks.append(TemporalBlock(in_ch, ch, kernel_size, dilation=2**i, dropout=dropout))
            in_ch = ch
        self.network = nn.Sequential(*blocks)
        self.classifier = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B,T,D] -> [B,D,T]
        x = x.transpose(1, 2)
        h = self.network(x)
        h = h.transpose(1, 2)  # [B,T,C]

        if mask is None:
            pooled = h.mean(dim=1)
        else:
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / denom

        return self.classifier(pooled)


class GRUBaseline(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 300, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.gru(x)
        if mask is None:
            pooled = out.mean(dim=1)
        else:
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (out * mask.unsqueeze(-1)).sum(dim=1) / denom
        return self.head(pooled)
