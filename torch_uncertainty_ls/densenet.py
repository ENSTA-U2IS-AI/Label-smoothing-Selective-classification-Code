"""Implementation based on the DenseNet-BC implementation in torchvision.

https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
"""

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        return conv(relu(norm(concated_features)))

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: int,
    ) -> None:
        super().__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))

        self.add_module(
            "conv1",
            nn.Conv2d(
                num_input_features,
                bn_size * growth_rate,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module(
            "conv2",
            nn.Conv2d(
                bn_size * growth_rate,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )
        self.drop_rate = drop_rate
        if self.drop_rate > 0:
            self.add_module("drop", nn.Dropout(self.drop_rate))

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))

        if self.drop_rate > 0:
            new_features = self.drop(new_features)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features) -> None:
        super().__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module(
            "conv",
            nn.Conv2d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Module):
    def __init__(
        self,
        num_layers,
        num_input_features,
        bn_size,
        growth_rate,
        drop_rate,
    ):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            if name != "concat":
                new_features = layer(*features)
                features.append(new_features)
        return torch.cat(features, dim=1)


class DenseNetBC(nn.Module):
    r"""Densenet-BC model class, based on Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>.

    Args:
    ----
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.

    """

    def __init__(
        self,
        growth_rate: int = 12,
        block_config: tuple[int, int, int] = (16, 16, 16),
        num_init_features: int = 24,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 10,
        small_inputs: bool = True,
        resolution: int = 32,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.avgpool_size = 8 if small_inputs else 7
        self.resolution = resolution

        if small_inputs:
            self.features = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv0",
                            nn.Conv2d(
                                3,
                                num_init_features,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False,
                            ),
                        ),
                    ]
                )
            )
        else:
            self.features = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv0",
                            nn.Conv2d(
                                3,
                                num_init_features,
                                kernel_size=7,
                                stride=2,
                                padding=3,
                                bias=False,
                            ),
                        ),
                    ]
                )
            )
            self.features.add_module("norm0", nn.BatchNorm2d(num_init_features))
            self.features.add_module("relu0", nn.ReLU(inplace=True))
            self.features.add_module(
                "pool0",
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),
            )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=int(num_features * 0.5),
                )
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = int(num_features * 0.5)

        self.features.add_module("norm_final", nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def feat_forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        features = F.relu(features, inplace=True)
        return F.avg_pool2d(features, kernel_size=self.avgpool_size).view(features.size(0), -1)

    def forward(self, x: Tensor, return_features: bool = False) -> Tensor:
        features = self.feat_forward(x)
        out = self.classifier(features)

        if return_features:
            return out, features
        return out
