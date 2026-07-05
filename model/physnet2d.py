

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysNet2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.start = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(5, 1), stride=1, padding=(2, 0)),
            nn.BatchNorm2d(32),
            nn.ELU(),
        )
        self.loop1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
        )
        self.encoder1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
        )
        self.encoder2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
        )
        self.loop4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
        )
        self.end = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(64, 1, kernel_size=(1, 1), stride=1, padding=(0, 0)),
        )

    def forward(self, x):
        _, _, _, width = x.shape
        means = torch.mean(x, dim=(2, 3), keepdim=True)
        stds = torch.std(x, dim=(2, 3), keepdim=True) + 1e-6
        x = (x - means) / stds

        parity = []
        x = self.start(x)
        x = self.loop1(x)
        parity.append(x.shape[-1] % 2)
        x = self.encoder1(x)
        parity.append(x.shape[-1] % 2)
        x = self.encoder2(x)
        x = self.loop4(x)

        x = F.interpolate(x, scale_factor=(1, 2), mode="nearest")
        x = self.decoder1(x)
        x = F.pad(x, (0, parity[-1], 0, 0, 0, 0), mode="replicate")
        x = F.interpolate(x, scale_factor=(1, 2), mode="nearest")
        x = self.decoder2(x)
        x = F.pad(x, (0, parity[-2], 0, 0, 0, 0), mode="replicate")
        x = self.end(x)
        return x.view(-1, width)
