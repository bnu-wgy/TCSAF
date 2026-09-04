import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import mamba_ssm  # noqa: F401
    HAS_MAMBA = True
except Exception:
    HAS_MAMBA = False


if HAS_MAMBA:
    try:
        from mamba_ssm import Mamba
    except ImportError:
        from mamba_ssm.torch import Mamba

class Mamba2D(nn.Module):
    def __init__(self, channels: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.ln = nn.LayerNorm(channels)
        self.mamba = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x
        x = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c) 
        x = self.ln(x)
        x = self.mamba(x)
        x = x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous() 
        x = self.proj_out(x)
        return x + residual

class _ConvGlobalMixer(nn.Module):
    """无 Mamba 时的降级实现：
    - 深度可分离卷积 (k=7) 建模局部
    - 膨胀卷积 (d=2, k=5) 提升感受野
    - SE 通道注意力 做通道级重标定
    """
    def __init__(self, channels: int):
        super().__init__()
        self.dw7 = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.act1 = nn.GELU()

        self.dw_dil = nn.Conv2d(channels, channels, kernel_size=5, padding=2*2, dilation=2, groups=channels, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.pw2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        # Squeeze-Excite
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

        # SE
        w = self.se_avg(x)
        w = self.se_fc1(w)
        w = self.act2(w)
        w = self.se_fc2(w).sigmoid()
        x = x * w

        return x + residual


class SS2DBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        if HAS_MAMBA:
            # 这里作为占位：多数开源实现以自定义 2D Mamba/SS2D 模块提供
            # 为保证可运行性，仍采用卷积版作为安全默认；
            # 后续若接入具体实现，只需替换以下一行为真实 Mamba2D 模块即可。
            self.mixer = Mamba2D(channels)
        else:
            self.mixer = _ConvGlobalMixer(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.mixer(x)
        return x


