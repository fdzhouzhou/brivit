"""2D-CNN and CRNN baselines (log-Mel spectrogram input)."""
from __future__ import annotations

import torch
import torch.nn as nn


class CNN2D(nn.Module):
    """Simple 3-stage VGG-ish 2D CNN over log-Mel spectrograms."""
    def __init__(self, channels=(32, 64, 128), n_classes: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        layers = []
        in_c = 1
        for c in channels:
            layers += [
                nn.Conv2d(in_c, c, 3, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 3, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_c = c
        self.feat = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(channels[-1], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feat(x)
        x = self.pool(x).flatten(1)
        return self.head(self.drop(x))


class CRNN(nn.Module):
    """Optional CNN-LSTM baseline (#28)."""
    def __init__(self, cnn_channels=(32, 64), rnn_hidden: int = 128,
                 n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        layers = []
        in_c = 1
        for c in cnn_channels:
            layers += [
                nn.Conv2d(in_c, c, 3, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 2)),
            ]
            in_c = c
        self.cnn = nn.Sequential(*layers)
        self.rnn = nn.LSTM(input_size=cnn_channels[-1],
                           hidden_size=rnn_hidden,
                           batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(rnn_hidden * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, F, T)
        h = self.cnn(x)                                 # (B, C, F', T')
        # Collapse freq dim by mean, treat time as sequence
        h = h.mean(-2).transpose(1, 2)                  # (B, T', C)
        o, _ = self.rnn(h)
        o = o.mean(1)                                   # temporal mean-pool
        return self.head(self.drop(o))
