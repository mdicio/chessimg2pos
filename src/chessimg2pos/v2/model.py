"""Model v2.

BoardNet: the board image (cropped to the 8x8 grid, resized to 256x256)
goes through a pretrained CNN whose stride-32 feature map is exactly 8x8, so
each feature cell sits on one square. A small transformer then lets every
square see the whole board (piece-set style, colours, coordinates, which
side is at the bottom) before classifying it. A CLS token predicts the
orientation (which side is drawn at the bottom).

LocatorNet: finds the board's grid box [x0, y0, x1, y1] in a screenshot
(image letterboxed to 320x320, coordinates normalised to that square).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .preprocess import BOARD_SIZE, LOC_SIZE  # noqa: F401


def _backbone(name, pretrained):
    if name == "convnext_tiny":
        w = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        m = torchvision.models.convnext_tiny(weights=w)
        return m.features, 768
    if name in ("resnet18", "resnet34"):
        w = {"resnet18": torchvision.models.ResNet18_Weights,
             "resnet34": torchvision.models.ResNet34_Weights}[name].DEFAULT if pretrained else None
        m = getattr(torchvision.models, name)(weights=w)
        return nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4), 512
    raise ValueError(name)


class BoardNet(nn.Module):
    def __init__(self, backbone="convnext_tiny", pretrained=True, d=256, layers=4, heads=8):
        super().__init__()
        self.config = dict(backbone=backbone, d=d, layers=layers, heads=heads)
        self.features, c = _backbone(backbone, pretrained)
        self.proj = nn.Sequential(nn.Conv2d(c, d, 1), nn.GELU())
        self.pos = nn.Parameter(torch.zeros(1, 65, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.square_head = nn.Linear(d, 13)
        self.orient_head = nn.Linear(d, 2)

    def forward(self, x):
        f = self.features(x)
        if f.shape[-1] != 8 or f.shape[-2] != 8:
            f = F.adaptive_avg_pool2d(f, 8)
        t = self.proj(f).flatten(2).transpose(1, 2)            # B, 64, d
        t = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], 1) + self.pos
        t = self.norm(self.encoder(t))
        return self.square_head(t[:, 1:]), self.orient_head(t[:, 0])


class LocatorNet(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True):
        super().__init__()
        self.config = dict(backbone=backbone)
        self.features, c = _backbone(backbone, pretrained)
        self.reduce = nn.Sequential(nn.Conv2d(c, 64, 1), nn.BatchNorm2d(64), nn.ReLU())
        n = (LOC_SIZE // 32) ** 2 * 64
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(n, 512), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(512, 4))

    def forward(self, x):
        return torch.sigmoid(self.head(self.reduce(self.features(x)))) * 1.2 - 0.1
