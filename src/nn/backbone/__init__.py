"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from .common import (
    get_activation, 
    FrozenBatchNorm2d,
    freeze_batch_norm2d,
)
from .presnet import PResNet
from .test_resnet import MResNet

from .timm_model import TimmModel
from .torchvision_model import TorchVisionModel

from .csp_resnet import CSPResNet
from .csp_darknet import CSPDarkNet, CSPPAN

from .hgnetv2 import HGNetv2

# src/nn/backbone/__init__.py
from .ghostnet import GhostNetV2
from src.core import register  # 确保导入 register 装饰器
@register(name="GhostNetV2")
def create_ghostnetv2():
    return GhostNetV2(in_channels=3, out_channels=256)
