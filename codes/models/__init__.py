"""
	Neural Architectures for image classification
"""

from .resnet import ResNet, ResNetEmbedding
from .lenet import LeNet, LeNet2d

__all__ = ["ResNet", "LeNet", "LeNet2d", "ResNetEmbedding"]
