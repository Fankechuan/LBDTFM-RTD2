# src/nn/backbone/ghostnet.py
from src.core import register
import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_divisible(v, divisor=4, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1, act=nn.SiLU):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            act()
        )

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_kernel_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        init_channels = int(out_channels / ratio)
        new_channels = out_channels - init_channels

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.SiLU() if relu else nn.Identity(),
        )

        self.cheap_conv = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_kernel_size, 1, dw_kernel_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.SiLU() if relu else nn.Identity(),
        )

    def forward(self, x):
        primary = self.primary_conv(x)
        cheap = self.cheap_conv(primary)
        return torch.cat([primary, cheap], dim=1)

class GhostBottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        self.stride = stride

        self.ghost1 = GhostModule(in_channels, mid_channels, relu=True)
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, kernel_size//2, groups=mid_channels, bias=False)
        self.bn = nn.BatchNorm2d(mid_channels)
        self.se = SELayer(mid_channels) if use_se else nn.Identity()
        self.ghost2 = GhostModule(mid_channels, out_channels, relu=False)

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, stride, kernel_size//2, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)

        x = self.ghost1(x)
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.se(x)
        x = self.ghost2(x)

        return x + residual

@register(name="GhostNetV2")
class GhostNetV2(nn.Module):
    def __init__(self, width=1.0, out_channels=256):
        super(GhostNetV2, self).__init__()
        self.conv_stem = ConvBNAct(3, _make_divisible(16 * width, 4), 3, 2)
        in_channels = _make_divisible(16 * width, 4)

        self.cfgs = [
            # k,  exp,   c,     se,  s
            [3,   16,    16,    0,   1],
            [3,   48,    24,    0,   2],
            [3,   72,    24,    0,   1],
            [5,   72,    40,    1,   2],
            [5,  120,    40,    1,   1],
            [3,  240,    80,    0,   2],
            [3,  200,    80,    0,   1],
            [3,  184,    80,    0,   1],
            [3,  184,    80,    0,   1],
            [3,  480,   112,    1,   1],
            [3,  672,   112,    1,   1],
            [5,  672,   160,    1,   2],
            [5,  960,   160,    0,   1],
            [5,  960,   160,    1,   1],
        ]

        self.stages = nn.ModuleList()
        stage_channels = []
        stage = []
        output_ids = [4, 8, 13]  # P3, P4, P5 对应的 stage 索引

        for idx, (k, exp, c, se, s) in enumerate(self.cfgs):
            out_c = _make_divisible(c * width, 4)
            exp_c = _make_divisible(exp * width, 4)
            stage.append(GhostBottleneck(in_channels, exp_c, out_c, k, s, se))
            in_channels = out_c
            if idx in output_ids:
                self.stages.append(nn.Sequential(*stage))
                stage_channels.append(out_c)
                stage = []

        self.out_convs = nn.ModuleList([
            ConvBNAct(c, out_channels, 1, 1) for c in stage_channels
        ])

    def forward(self, x):
        x = self.conv_stem(x)
        outs = []

        for stage, conv in zip(self.stages, self.out_convs):
            x = stage(x)
            outs.append(conv(x))

        return outs  # [P3, P4, P5] 三个尺度，通道数统一为 out_channels

