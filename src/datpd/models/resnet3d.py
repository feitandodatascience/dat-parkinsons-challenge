"""A small 3D residual CNN. VENDORED INTO submission.zip.

Architecture (documented in docs/02, "A concrete first architecture"):

    input  (B, 1, 96, 96, 96)
      stem     Conv3d k5 s2 -> GroupNorm -> SiLU      (B,  16, 48, 48, 48)
      stage 1  ResBlock s2, ResBlock                  (B,  32, 24, 24, 24)
      stage 2  ResBlock s2, ResBlock                  (B,  64, 12, 12, 12)
      stage 3  ResBlock s2, ResBlock                  (B, 128,  6,  6,  6)
      pool     AdaptiveAvgPool3d(1)                   (B, 128)
      head     Dropout -> Linear                      (B,) logits

Around 1-2 M parameters. Small on purpose: with a few thousand samples a bigger model mostly
buys overfitting. Get this working end to end and submitted, *then* scale up.

**GroupNorm, not BatchNorm** (docs/02 Delta 3), and this is a competition-rules matter rather
than a tuning preference. BatchNorm mixes samples inside a batch; at inference that would make
one examination's prediction depend on which other examinations shared its batch, violating the
per-sample independence rule (CLAUDE.md 1.1). GroupNorm normalises within a single sample, so
independence is structurally true rather than promised - and 3D volumes force batch sizes small
enough that BatchNorm statistics would be noisy anyway.
`test_eval_mode_is_batch_size_independent` is the standing check on that.
"""

from __future__ import annotations

from math import gcd

import torch
import torch.nn as nn

# Target number of GroupNorm groups. The actual count is gcd(this, channels) so it always
# divides the channel count, whatever widths a config asks for.
GROUPNORM_GROUPS = 8


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(max(1, gcd(GROUPNORM_GROUPS, channels)), channels)


class ResBlock(nn.Module):
    """Two 3x3x3 convolutions with a residual connection, optionally downsampling by 2.

    SiLU rather than ReLU: it is smooth, costs nothing extra here, and tends to train slightly
    more stably at the small batch sizes 3D volumes force.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = _norm(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.norm2 = _norm(out_channels)
        self.act = nn.SiLU(inplace=True)

        # Identity shortcut where shapes already agree; a 1x1x1 projection otherwise.
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                _norm(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + identity)


class ResNet3D(nn.Module):
    """Binary classifier over a single-channel 3D volume. Returns raw logits of shape (B,).

    Logits, not probabilities: `BCEWithLogitsLoss` fuses the sigmoid with the loss and is
    numerically stable where a separate sigmoid is not. The sigmoid is applied once, at
    prediction time.
    """

    def __init__(
        self,
        widths: tuple[int, ...] = (16, 32, 64, 128),
        dropout: float = 0.3,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        if len(widths) < 2:
            raise ValueError("widths needs a stem width plus at least one stage width")

        stem_width, stage_widths = widths[0], widths[1:]

        # A k5 s2 stem halves the volume immediately. That single stride is worth a lot of
        # memory: activations here are cubic in resolution (docs/02 Delta 2).
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, stem_width, 5, stride=2, padding=2, bias=False),
            _norm(stem_width),
            nn.SiLU(inplace=True),
        )

        stages, prev = [], stem_width
        for width in stage_widths:
            stages.append(ResBlock(prev, width, stride=2))
            stages.append(ResBlock(width, width, stride=1))
            prev = width
        self.stages = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stages(self.stem(x))
        x = self.pool(x).flatten(1)
        return self.head(self.dropout(x)).squeeze(-1)
