import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from mamba_ssm import Mamba  # type: ignore
    HAS_MAMBA = True
except Exception:
    try:
        from mamba_ssm.torch import Mamba  # type: ignore
        HAS_MAMBA = True
    except Exception:
        HAS_MAMBA = False


class Mamba2D(nn.Module):
    def __init__(self, channels: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.ln = nn.LayerNorm(channels)
        self.mamba = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x
        x = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)  # (B,L,C)
        x = self.ln(x)
        x = self.mamba(x)  # (B,L,C)
        x = x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # (B,C,H,W)
        x = self.proj(x)
        return x + residual


class _AxisGlobalMixer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw7 = nn.Conv2d(channels, channels, 7, padding=3, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.act1 = nn.GELU()

        self.dw_dil = nn.Conv2d(channels, channels, 5, padding=2*2, dilation=2, groups=channels, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.pw2 = nn.Conv2d(channels, channels, 1, bias=False)

        squeeze = max(8, channels // 16)
        self.se_avg = nn.AdaptiveAvgPool2d(1)
        self.se_fc1 = nn.Conv2d(channels, squeeze, 1)
        self.se_fc2 = nn.Conv2d(squeeze, channels, 1)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw7(x)
        x = self.bn1(x)
        x = self.pw1(x)
        x = self.act1(x)

        x = self.dw_dil(x)
        x = self.bn2(x)
        x = self.pw2(x)

        w = self.se_avg(x)
        w = self.se_fc1(w)
        w = self.act2(w)
        w = self.se_fc2(w).sigmoid()
        x = x * w
        return x + residual


class _DiagonalMixer(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        # 主对角（↘）
        self.dw_tlbr = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        # 副对角（↙）
        self.dw_trbl = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels * 2)
        self.act = nn.GELU()
        self.proj = nn.Conv2d(channels * 2, channels, 1, bias=False)

        # 初始化为对角模板（之后可学习微调）
        with torch.no_grad():
            k1 = torch.zeros(channels, 1, 3, 3)
            k1[:, 0, 0, 0] = 1/3
            k1[:, 0, 1, 1] = 1/3
            k1[:, 0, 2, 2] = 1/3
            self.dw_tlbr.weight.copy_(k1)

            k2 = torch.zeros(channels, 1, 3, 3)
            k2[:, 0, 0, 2] = 1/3
            k2[:, 0, 1, 1] = 1/3
            k2[:, 0, 2, 0] = 1/3
            self.dw_trbl.weight.copy_(k2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.dw_tlbr(x)
        b = self.dw_trbl(x)
        y = torch.cat([a, b], dim=1)
        y = self.bn(y)
        y = self.act(y)
        y = self.proj(y)
        return y


class SS22DBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        if HAS_MAMBA:
            self.axis_mixer = Mamba2D(channels)
        else:
            self.axis_mixer = _AxisGlobalMixer(channels)
        self.diag_mixer = _DiagonalMixer(channels)

        self.fuse = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.fuse_bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        xa = self.axis_mixer(x)
        xd = self.diag_mixer(x)
        y = torch.cat([xa, xd], dim=1)
        y = self.fuse(y)
        y = self.fuse_bn(y)
        y = self.act(y)
        return y + residual


