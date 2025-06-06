# ✅ ResNet101 variant with BlurPool downsampling + SE attention
# هدف: بهبود تشخیص چهره low-resolution (مثل TinyFace)

import torch
import torch.nn as nn
from typing import Optional

# 🔹 BlurPool برای کاهش aliasing هنگام downsampling
class BlurPool2D(nn.Module):
    def __init__(self, channels, filt_size=3, stride=2):
        super().__init__()
        assert filt_size == 3, "Only filt_size=3 supported in this lightweight version."
        self.stride = stride
        a = torch.tensor([1., 2., 1.])
        filt = a[:, None] * a[None, :]
        filt = filt / filt.sum()
        self.register_buffer('filt', filt[None, None, :, :].repeat((channels, 1, 1, 1)))
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, x):
        x = self.pad(x)
        return nn.functional.conv2d(x, self.filt, stride=self.stride, groups=x.shape[1])

# 🔹 SE attention
class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.PReLU(channels // reduction),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w

# 🔹 Basic Residual Block + BlurPool + SE
class BlurBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-5)
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
        self.prelu = nn.PReLU(planes)

        if stride == 2:
            self.blur = BlurPool2D(planes)
        else:
            self.blur = None
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
        self.se = SqueezeExcite(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        if self.blur:
            out = self.blur(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

# 🔹 ResNet101 with BlurPool + SE
class IResNet_Blur(nn.Module):
    fc_scale = 7 * 7
    def __init__(self, block, layers, dropout: float = 0.0, num_features: int = 512):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._make_layer(block,  64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512, eps=1e-5)
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(512 * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-5)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride == 2 or self.inplanes != planes:
            downsample = nn.Sequential(
                BlurPool2D(self.inplanes),
                nn.Conv2d(self.inplanes, planes, 1, bias=False),
                nn.BatchNorm2d(planes, eps=1e-5)
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.features(x)
        return x

# 🔹 Builder

def iresnet101_blur(input_size=(112, 112), output_dim: int = 512, dropout: float = 0.0, **kwargs):
    layers = [3, 13, 30, 3]
    return IResNet_Blur(
        BlurBasicBlock,
        layers,
        dropout=dropout,
        num_features=output_dim,
        **kwargs,
    )
