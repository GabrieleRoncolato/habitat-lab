#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Type, Union, cast

from torch import Tensor
from torch import nn as nn
from torch.nn.modules.container import Sequential
from torch.nn.modules.conv import Conv2d

import torch.nn.functional as F

class BasicBlock(nn.Module):

    def __init__(
        self,
        in_planes,
        planes,
        shortcut,
        stride=1
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        if shortcut:
            self.shortcut = nn.Sequential(
              nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
              nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def _build_bottleneck_branch(
    inplanes: int,
    planes: int,
    ngroups: int,
    stride: int,
    expansion: int,
    groups: int = 1,
) -> Sequential:
    return nn.Sequential(
        conv1x1(inplanes, planes),
        nn.GroupNorm(ngroups, planes),
        nn.ReLU(True),
        conv3x3(planes, planes, stride, groups=groups),
        nn.GroupNorm(ngroups, planes),
        nn.ReLU(True),
        conv1x1(planes, planes * expansion),
        nn.GroupNorm(ngroups, planes * expansion),
    )


class SE(nn.Module):
    def __init__(self, planes, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(planes, int(planes / r)),
            nn.ReLU(True),
            nn.Linear(int(planes / r), planes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        x = self.squeeze(x)
        x = x.view(b, c)
        x = self.excite(x)

        return x.view(b, c, 1, 1)


def _build_se_branch(planes, r=16):
    return SE(planes, r)


class Bottleneck(nn.Module):
    expansion = 4
    resneXt = False

    def __init__(
        self,
        inplanes: int,
        planes: int,
        ngroups: int,
        stride: int = 1,
        downsample: Optional[Sequential] = None,
        cardinality: int = 1,
    ) -> None:
        super().__init__()
        self.convs = _build_bottleneck_branch(
            inplanes,
            planes,
            ngroups,
            stride,
            self.expansion,
            groups=cardinality,
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def _impl(self, x: Tensor) -> Tensor:
        identity = x

        out = self.convs(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)

    def forward(self, x: Tensor) -> Tensor:
        return self._impl(x)


class SEBottleneck(Bottleneck):
    def __init__(
        self,
        inplanes,
        planes,
        ngroups,
        stride=1,
        downsample=None,
        cardinality=1,
    ):
        super().__init__(
            inplanes, planes, ngroups, stride, downsample, cardinality
        )

        self.se = _build_se_branch(planes * self.expansion)

    def _impl(self, x):
        identity = x

        out = self.convs(x)
        out = self.se(out) * out

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)


class SEResNeXtBottleneck(SEBottleneck):
    expansion = 2
    resneXt = True


class ResNeXtBottleneck(Bottleneck):
    expansion = 2
    resneXt = True


Block = Union[Type[Bottleneck], Type[BasicBlock]]


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        in_channels,
        in_planes=32
    ) -> None:
        super(ResNet, self).__init__()

        self.in_planes = in_planes

        self.conv1 = nn.Conv2d(in_channels, in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)

        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, in_planes * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, in_planes * 8, num_blocks[3], stride=2)

        self.final_channels = self.in_planes
        self.final_spatial_compress = 1.0 / (2**5)

    def _make_layer(
        self,
        block,
        planes,
        num_blocks,
        stride,
    ) -> Sequential:
        shortcut = stride != 1 or self.in_planes != planes

        layers = []
        layers.append(
            block(
                self.in_planes,
                planes,
                shortcut,
                stride
            )
        )
        self.in_planes = planes
        for _i in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, False, 1))

        return nn.Sequential(*layers)

    def forward(self, x) -> Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.avg_pool2d(x, 3, 3)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet18(in_channels, base_planes, ngroups):
    model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels, base_planes)

    return model


def resnet50(in_channels: int, base_planes: int, ngroups: int) -> ResNet:
    model = ResNet(in_channels, base_planes, ngroups, Bottleneck, [3, 4, 6, 3])

    return model


def resneXt50(in_channels, base_planes, ngroups):
    model = ResNet(
        in_channels,
        base_planes,
        ngroups,
        ResNeXtBottleneck,
        [3, 4, 6, 3],
        cardinality=int(base_planes / 2),
    )

    return model


def se_resnet50(in_channels, base_planes, ngroups):
    model = ResNet(
        in_channels, base_planes, ngroups, SEBottleneck, [3, 4, 6, 3]
    )

    return model


def se_resneXt50(in_channels, base_planes, ngroups):
    model = ResNet(
        in_channels,
        base_planes,
        ngroups,
        SEResNeXtBottleneck,
        [3, 4, 6, 3],
        cardinality=int(base_planes / 2),
    )

    return model


def se_resneXt101(in_channels, base_planes, ngroups):
    model = ResNet(
        in_channels,
        base_planes,
        ngroups,
        SEResNeXtBottleneck,
        [3, 4, 23, 3],
        cardinality=int(base_planes / 2),
    )

    return model
