"""
The script is adapted from torchvision.models.ResNet
"""

import torch.nn as nn
from models.fca_layer import FcaLayer as FCA
__all__ = ['fca_resnet20', 'fca_resnet32', 'fca_resnet44', 'fca_resnet56', 'fca_resnet110', 'fca_resnet164']


BN_momentum = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class FCABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, reduction=16,resolution=32,new_resnet=False):
        super(FCABasicBlock, self).__init__()
        self.new_resnet = new_resnet
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(
            inplanes if new_resnet else planes, momentum=BN_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_momentum)
        self.se = FCA(planes, reduction,resolution,resolution)
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes, momentum=BN_momentum))
        else:
            self.downsample = lambda x: x
        self.stride = stride
        self.output = planes * self.expansion

    def _old_resnet(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    def _new_resnet(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

    def forward(self, x):
        if self.new_resnet:
            return self._new_resnet(x)
        else:
            return self._old_resnet(x)


class CifarNet(nn.Module):
    """
    This is specially designed for cifar10
    """

    def __init__(self, block, n_size, num_classes=10, reduction=16, new_resnet=False, dropout=0.):
        super(CifarNet, self).__init__()
        self.inplane = 16
        self.new_resnet = new_resnet
        self.dropout_prob = dropout
        self.conv1 = nn.Conv2d(
            3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane, momentum=BN_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            block, 16, blocks=n_size, stride=1, reduction=reduction,resolution=28)
        self.layer2 = self._make_layer(
            block, 32, blocks=n_size, stride=2, reduction=reduction,resolution=14)
        self.layer3 = self._make_layer(
            block, 64, blocks=n_size, stride=2, reduction=reduction,resolution=7)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if self.dropout_prob > 0:
            self.dropout_layer = nn.Dropout(p=self.dropout_prob, inplace=True)
        self.fc = nn.Linear(self.inplane, num_classes)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, reduction,resolution):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride,
                                reduction, resolution,new_resnet=self.new_resnet))
            self.inplane = layers[-1].output

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.dropout_prob > 0:
            x = self.dropout_layer(x)
        x = self.fc(x)

        return x


def fca_resnet20(**kwargs):
    """Constructs a ResNet-18 model.

    """
    model = CifarNet(FCABasicBlock, 3, **kwargs)
    return model


def fca_resnet32(**kwargs):
    """Constructs a ResNet-34 model.

    """
    model = CifarNet(FCABasicBlock, 5, **kwargs)
    return model


def fca_resnet44(**kwargs):
    """Constructs a ResNet-34 model.

    """
    model = CifarNet(FCABasicBlock, 7, **kwargs)
    return model


def fca_resnet56(**kwargs):
    """Constructs a ResNet-34 model.

    """
    model = CifarNet(FCABasicBlock, 9, **kwargs)
    return model


def fca_resnet110(**kwargs):
    """Constructs a ResNet-34 model.

    """
    model = CifarNet(FCABasicBlock, 18, **kwargs)
    return model


def fca_resnet164(**kwargs):
    """Constructs a ResNet-34 model.

    """
    model = CifarNet(FCABasicBlock, 27, **kwargs)
    return model

