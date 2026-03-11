import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """
    DnCNN for disparity denoising.
    Input/output: (B, 1, H, W) normalized disparity in [0, 1].
    Predicts the clean disparity directly (not residual).
    """
    def __init__(self, depth=8, channels=64):
        super().__init__()
        layers = [nn.Conv2d(1, channels, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(channels, channels, 3, padding=1),
                       nn.BatchNorm2d(channels),
                       nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(channels, 1, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
