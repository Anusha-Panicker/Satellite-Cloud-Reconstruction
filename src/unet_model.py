"""Small U-Net for three-band satellite cloud reconstruction.

Inputs:  [N, 4, H, W] = normalized NIR, Red, Green, binary cloud mask.
Outputs: [N, 3, H, W] = reconstructed normalized NIR, Red, Green.

The output is constrained to [0, 1]. This architecture does not assume that
synthetically reconstructed regions will match real obscured ground truth.
"""

import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            raise ValueError(
                f"Skip mismatch: upsampled={tuple(x.shape)}, skip={tuple(skip.shape)}"
            )
        return self.conv(torch.cat((skip, x), dim=1))


class CloudUNet(nn.Module):
    def __init__(
        self, in_channels: int = 4, out_channels: int = 3, base_channels: int = 16
    ):
        super().__init__()
        if base_channels <= 0 or base_channels % 4 != 0:
            raise ValueError("base_channels must be a positive multiple of 4")

        b = base_channels
        self.enc1 = DoubleConv(in_channels, b)
        self.enc2 = DoubleConv(b, b * 2)
        self.enc3 = DoubleConv(b * 2, b * 4)
        self.enc4 = DoubleConv(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(b * 8, b * 16)

        self.dec4 = UpBlock(b * 16, b * 8, b * 8)
        self.dec3 = UpBlock(b * 8, b * 4, b * 4)
        self.dec2 = UpBlock(b * 4, b * 2, b * 2)
        self.dec1 = UpBlock(b * 2, b, b)
        self.output = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 4:
            raise ValueError(f"Expected [N,4,H,W], received {tuple(x.shape)}")
        if x.shape[-2] % 16 or x.shape[-1] % 16:
            raise ValueError("Input image height and width must be divisible by 16")

        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        z = self.bottleneck(self.pool(s4))

        z = self.dec4(z, s4)
        z = self.dec3(z, s3)
        z = self.dec2(z, s2)
        z = self.dec1(z, s1)
        return torch.sigmoid(self.output(z))
