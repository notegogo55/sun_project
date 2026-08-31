"""U-Net สำหรับแบ่งส่วน active region จากภาพ magnetogram เต็มดวง

โครงสร้างตามต้นฉบับ (Ronneberger et al. 2015) แต่ปรับให้พอดีกับ **VRAM 4 GB**:

* ``base_channels=32`` แทน 64 ในต้นฉบับ — ลดหน่วยความจำลงครึ่งหนึ่ง
* upsample ด้วย bilinear + conv แทน ``ConvTranspose2d`` — พารามิเตอร์น้อยกว่าและ
  ไม่เกิด checkerboard artifact
* ทำงานที่ 512x512 (ย่อจากภาพจริง 4096x4096)

input มี 1 channel คือ magnetogram ที่ normalise แล้วด้วย ``tanh(B/300G)`` ซึ่ง
คงเครื่องหมายของขั้วแม่เหล็กไว้ (ดู :func:`normalise_magnetogram`)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_layer(kind: str, channels: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if kind == "group":
        return nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
    raise ValueError(f"ไม่รู้จัก normalisation ชนิด {kind!r}")


class DoubleConv(nn.Module):
    """(conv 3x3 -> norm -> ReLU) x 2 — หน่วยพื้นฐานของทุกระดับใน U-Net"""

    def __init__(self, in_channels: int, out_channels: int, norm: str = "batch") -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm_layer(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm_layer(norm, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """maxpool ลดขนาดครึ่งหนึ่ง แล้วตามด้วย DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int, norm: str = "batch") -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, norm),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """ขยายขนาด แล้วต่อ (concatenate) กับ skip connection จาก encoder"""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        bilinear: bool = True,
        norm: str = "batch",
    ) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            up_out = in_channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            up_out = in_channels // 2
        self.conv = DoubleConv(up_out + skip_channels, out_channels, norm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # ถ้าขนาดไม่ลงตัวพอดี (ภาพที่ด้านไม่หารด้วย 2^depth) ให้ pad ให้เท่า skip
        diff_y = skip.size(-2) - x.size(-2)
        diff_x = skip.size(-1) - x.size(-1)
        if diff_y or diff_x:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """U-Net สำหรับ binary segmentation

    Parameters
    ----------
    base_channels
        จำนวน channel ที่ระดับบนสุด แต่ละระดับที่ลึกลงไปจะเพิ่มเป็นสองเท่า
        ลดเหลือ 16 ได้ถ้า VRAM ไม่พอ
    depth
        จำนวนครั้งที่ลดขนาด — 4 หมายถึงภาพ 512 จะเล็กสุดที่ 32x32
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        norm: str = "batch",
        bilinear_upsample: bool = True,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth ต้องอย่างน้อย 1 ได้รับ {depth}")

        self.depth = depth
        # ตัวอย่าง base=32, depth=4 -> [32, 64, 128, 256, 512]
        channels = [base_channels * (2**i) for i in range(depth + 1)]

        self.stem = DoubleConv(in_channels, channels[0], norm)
        self.downs = nn.ModuleList(
            [Down(channels[i], channels[i + 1], norm) for i in range(depth)]
        )
        self.ups = nn.ModuleList(
            [
                Up(channels[i + 1], channels[i], channels[i], bilinear_upsample, norm)
                for i in reversed(range(depth))
            ]
        )
        self.out_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # เริ่มต้น bias ของชั้นสุดท้ายให้ทายว่า "ไม่ใช่ AR" (prior ~2% ของพิกเซล)
        # ช่วยให้ loss ไม่ระเบิดในช่วง epoch แรกเมื่อข้อมูลไม่สมดุลรุนแรง
        nn.init.constant_(self.out_conv.bias, -4.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """คืน logits รูปทรงเดียวกับ input (ยังไม่ผ่าน sigmoid)"""
        if x.dim() != 4:
            raise ValueError(f"คาดหวัง input 4 มิติ (B, C, H, W) แต่ได้ {tuple(x.shape)}")

        skips: list[torch.Tensor] = []
        out = self.stem(x)
        for down in self.downs:
            skips.append(out)
            out = down(out)

        for up, skip in zip(self.ups, reversed(skips), strict=True):
            out = up(out, skip)

        return self.out_conv(out)

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """คืน mask แบบไบนารี — ใช้ตอน inference"""
        self.eval()
        return (torch.sigmoid(self(x)) >= threshold).to(torch.uint8)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def normalise_magnetogram(data, scale_gauss: float = 300.0):
    """บีบ magnetogram ให้อยู่ในช่วง (-1, 1) ด้วย ``tanh(B / scale)``

    สนามแม่เหล็กบนดวงอาทิตย์มีช่วงค่ากว้างมาก: quiet Sun อยู่ราว +-10 G ขณะที่ใจกลาง
    จุดดับแรงได้ถึง +-3000 G การหารด้วยค่าสูงสุดตรงๆ จะทำให้ quiet Sun กลายเป็น 0
    เกือบหมด ส่วน ``tanh`` ให้ความละเอียดสูงในช่วงสนามอ่อน-ปานกลาง (ซึ่งเป็นบริเวณ
    ที่ขอบเขตของ AR อยู่พอดี) แล้วค่อยๆ อิ่มตัวที่สนามแรง

    ที่ ``scale=300`` สนาม 100 G จะได้ ~0.32 และ 1000 G จะได้ ~0.995
    เครื่องหมายถูกรักษาไว้เสมอ เพราะขั้วแม่เหล็กเป็นข้อมูลทางฟิสิกส์ที่สำคัญ
    """
    if isinstance(data, torch.Tensor):
        return torch.tanh(data / scale_gauss)
    import numpy as np

    return np.tanh(np.asarray(data, dtype=np.float32) / scale_gauss)


def build_unet(cfg) -> UNet:
    """สร้างโมเดลจาก ``configs/unet.yaml``"""
    return UNet(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        base_channels=cfg.base_channels,
        depth=cfg.depth,
        norm=cfg.norm,
        bilinear_upsample=cfg.bilinear_upsample,
    )
