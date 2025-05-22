# iresnet101_spd.py
# ✅ ResNet101 variant with SPD‑Conv downsampling **و** Squeeze‑Excitation attention
#    سازگار با AdaFace / CVLFace برای تشخیص چهرهٔ کم‌رزولوشن

import torch
import torch.nn as nn

# ---------------------------------------------------------------
# 🔹 Squeeze‑and‑Excitation (SE) Attention
# ---------------------------------------------------------------
class SqueezeExcite(nn.Module):
    """Lightweight channel‑wise attention (SE)"""
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

# ---------------------------------------------------------------
# 🔹 SPD Basic Residual Block
# ---------------------------------------------------------------
from typing import Optional

class SPDBasicBlock(nn.Module):
    """Residual block + optional SPD downsample + SE attention"""
    expansion = 1  # no channel expansion in basic block

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.stride = stride

        # Pre‑activation BN
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-5)
        # 1️⃣ First 3×3 conv (stride 1)
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
        self.prelu = nn.PReLU(planes)

        # 2️⃣ Second 3×3 conv; if downsample, input comes from PixelUnshuffle
        if stride == 2:
            self.spd = nn.PixelUnshuffle(2)            # H,W → H/2,W/2 ; C → 4C
            self.conv2 = nn.Conv2d(planes * 4, planes, 3, stride=1, padding=1, bias=False)
        else:
            self.spd = None
            self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
        self.se   = SqueezeExcite(planes)              # ✨ attention layer
        self.downsample = downsample                  # skip branch

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)

        # SPD downsampling if needed
        if self.stride == 2 and self.spd is not None:
            out = self.spd(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.se(out)            # 🏷️ apply SE

        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

# ---------------------------------------------------------------
# 🔹 ResNet‑101 backbone with SPD + SE
# ---------------------------------------------------------------
class IResNet_SPD(nn.Module):
    fc_scale = 7 * 7  # final spatial dim (for 112×112 input)

    def __init__(self, block, layers, dropout: float = 0.0, num_features: int = 512):
        super().__init__()
        self.inplanes = 64

        # Stem (no downsample)
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64, eps=1e-5)
        self.prelu = nn.PReLU(64)

        # Residual stages (downsample by SPD in first block of each layer)
        self.layer1 = self._make_layer(block,  64, layers[0], stride=2)  # 112→56
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 56→28
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 28→14
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 14→7

        # Final BN → dropout → FC → feature BN
        self.bn2     = nn.BatchNorm2d(512, eps=1e-5)
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout > 0 else nn.Identity()
        self.fc      = nn.Linear(512 * self.fc_scale, num_features)
        self.features= nn.BatchNorm1d(num_features, eps=1e-5)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # -- helper
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride == 2 or self.inplanes != planes:
            if stride == 2:
                downsample = nn.Sequential(
                    nn.PixelUnshuffle(2),
                    nn.Conv2d(self.inplanes * 4, planes, 1, bias=False),
                    nn.BatchNorm2d(planes, eps=1e-5)
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes, 1, bias=False),
                    nn.BatchNorm2d(planes, eps=1e-5)
                )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    # -- forward
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

# ---------------------------------------------------------------
# 🔹 Builder
# ---------------------------------------------------------------
def iresnet101_spd(input_size=(112, 112), output_dim: int = 512, dropout: float = 0.0, **kwargs):
    """Creates a ResNet‑101 with SPD downsampling + SE attention.
    Args:
        input_size: kept only for compatibility with loader signature.
        output_dim: final embedding dimension.
        dropout   : dropout after conv body.
        **kwargs  : forwarded to IResNet_SPD (e.g., device).
    """
    layers = [3, 13, 30, 3]  # ≈101 layers
    return IResNet_SPD(
        SPDBasicBlock,
        layers,
        dropout=dropout,
        num_features=output_dim,
        **kwargs,
    )
