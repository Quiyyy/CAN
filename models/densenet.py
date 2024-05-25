import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

# Depthwise Separable Convolution
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, use_dropout=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        out = F.relu(out, inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        return out

# Bottleneck with Depthwise Separable Convolution
class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate, use_dropout):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.conv1 = DepthwiseSeparableConv(nChannels, interChannels, kernel_size=1, padding=0, use_dropout=use_dropout)
        self.conv2 = DepthwiseSeparableConv(interChannels, growthRate, kernel_size=3, padding=1, use_dropout=use_dropout)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.cat((x, out), 1)
        return out

# SingleLayer with Depthwise Separable Convolution
class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate, use_dropout):
        super(SingleLayer, self).__init__()
        self.conv1 = DepthwiseSeparableConv(nChannels, growthRate, kernel_size=3, padding=1, use_dropout=use_dropout)

    def forward(self, x):
        out = self.conv1(x)
        out = torch.cat((x, out), 1)
        return out

# Transition Layer
class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels, use_dropout):
        super(Transition, self).__init__()
        self.conv1 = DepthwiseSeparableConv(nChannels, nOutChannels, kernel_size=1, padding=0, use_dropout=use_dropout)
        self.pool = nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)
        return out

# DenseNet with Depthwise Separable Convolution
class DenseNet(nn.Module):
    def __init__(self, params):
        super(DenseNet, self).__init__()
        growthRate = params['densenet']['growthRate']
        reduction = params['densenet']['reduction']
        bottleneck = params['densenet']['bottleneck']
        use_dropout = params['densenet']['use_dropout']

        nDenseBlocks = 16
        nChannels = 2 * growthRate
        self.conv1 = nn.Conv2d(params['encoder']['input_channel'], nChannels, kernel_size=7, padding=3, stride=2, bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels, use_dropout)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels, use_dropout)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout)

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate, use_dropout))
            else:
                layers.append(SingleLayer(nChannels, growthRate, use_dropout))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out, inplace=True)
        out = F.max_pool2d(out, 2, ceil_mode=True)
        out = self.dense1(out)
        out = self.trans1(out)
        out = self.dense2(out)
        out = self.trans2(out)
        out = self.dense3(out)
        return out
