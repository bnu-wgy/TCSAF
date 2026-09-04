from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba  # type: ignore
    HAS_MAMBA = True
except Exception:
    try:
        from mamba_ssm.torch import Mamba  # type: ignore
        HAS_MAMBA = True
    except Exception:
        Mamba = None  # type: ignore
        HAS_MAMBA = False


DIRECTION_NAMES = (
    'horizontal_forward',
    'horizontal_backward',
    'vertical_forward',
    'vertical_backward',
    'main_diagonal_forward',
    'main_diagonal_backward',
    'anti_diagonal_forward',
    'anti_diagonal_backward',
)


def _diagonal_order(height: int, width: int, anti: bool) -> List[int]:
    """Return a permutation that follows complete image diagonals."""
    order: List[int] = []

    if anti:
        # Traverse each anti-diagonal from upper-right to lower-left.
        starts = [(0, col) for col in range(width)]
        starts.extend((row, width - 1) for row in range(1, height))
        for row, col in starts:
            while row < height and col >= 0:
                order.append(row * width + col)
                row += 1
                col -= 1
    else:
        # Traverse each main diagonal from upper-left to lower-right.
        starts = [(0, col) for col in range(width - 1, -1, -1)]
        starts.extend((row, 0) for row in range(1, height))
        for row, col in starts:
            while row < height and col < width:
                order.append(row * width + col)
                row += 1
                col += 1

    if len(order) != height * width or len(set(order)) != height * width:
        raise RuntimeError('Diagonal scan construction did not cover each pixel once.')
    return order


class EightDirectionalSelectiveScan(nn.Module):
    """Apply an SSM to eight spatial scan directions and remap to 2D."""

    def __init__(
        self,
        channels: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        ssm_factory: Optional[Callable[[], nn.Module]] = None,
    ):
        super().__init__()
        self.channels = channels

        if ssm_factory is None:
            if not HAS_MAMBA:
                raise ImportError(
                    'mamba_ssm is required for the eight-directional selective '
                    'scan described in Eq. (7). Install mamba-ssm before '
                    'constructing TCSAF.'
                )

            def ssm_factory() -> nn.Module:
                return Mamba(  # type: ignore[misc]
                    d_model=channels,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )

        # Eq. (7) uses a direction-specific SSM_r for each scan route.
        self.ssm_layers = nn.ModuleList(
            [ssm_factory() for _ in DIRECTION_NAMES]
        )
        self.output_norm = nn.LayerNorm(channels)
        self._scan_cache: Dict[
            Tuple[int, int, str, Optional[int]],
            List[Tuple[torch.Tensor, torch.Tensor]],
        ] = {}

    def _apply(self, fn):
        # Cached permutations are device-specific and are not state buffers.
        self._scan_cache.clear()
        return super()._apply(fn)

    def _scan_orders(
        self,
        height: int,
        width: int,
        device: torch.device,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        cache_key = (height, width, device.type, device.index)
        cached = self._scan_cache.get(cache_key)
        if cached is not None:
            return cached

        length = height * width
        horizontal = torch.arange(length, device=device, dtype=torch.long)
        vertical = horizontal.view(height, width).transpose(0, 1).reshape(-1)
        main_diagonal = torch.tensor(
            _diagonal_order(height, width, anti=False),
            device=device,
            dtype=torch.long,
        )
        anti_diagonal = torch.tensor(
            _diagonal_order(height, width, anti=True),
            device=device,
            dtype=torch.long,
        )

        base_orders = (horizontal, vertical, main_diagonal, anti_diagonal)
        orders: List[torch.Tensor] = []
        for order in base_orders:
            orders.append(order)
            orders.append(torch.flip(order, dims=(0,)))

        cached = [(order, torch.argsort(order)) for order in orders]
        self._scan_cache[cache_key] = cached
        return cached

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f'Expected a BCHW tensor, got shape {tuple(x.shape)}.')

        batch, channels, height, width = x.shape
        if channels != self.channels:
            raise ValueError(
                f'Expected {self.channels} channels, got {channels} channels.'
            )

        # Original 2D token order used as the common remapping target.
        flat = x.permute(0, 2, 3, 1).contiguous().view(
            batch, height * width, channels
        )

        merged = None
        for ssm, (order, inverse) in zip(
            self.ssm_layers,
            self._scan_orders(height, width, x.device),
        ):
            sequence = flat.index_select(1, order)
            sequence = ssm(sequence)
            if not isinstance(sequence, torch.Tensor) or sequence.shape != flat.shape:
                raise RuntimeError(
                    'Each directional SSM must return a tensor with shape '
                    f'{tuple(flat.shape)}.'
                )
            remapped = sequence.index_select(1, inverse)
            merged = remapped if merged is None else merged + remapped

        if merged is None:
            raise RuntimeError('No directional SSM outputs were produced.')

        # Eq. (7): LN(sum_r Remap_r(SSM_r(Scan_r(F)))).
        merged = self.output_norm(merged)
        return merged.view(batch, height, width, channels).permute(
            0, 3, 1, 2
        ).contiguous()


class SS22DBlock(nn.Module):
    """Eight-directional SSM followed by the gated residual in Eq. (8)."""

    def __init__(
        self,
        channels: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        ssm_factory: Optional[Callable[[], nn.Module]] = None,
    ):
        super().__init__()
        self.channels = channels
        self.directional_scan = EightDirectionalSelectiveScan(
            channels=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            ssm_factory=ssm_factory,
        )
        self.gate_norm = nn.LayerNorm(channels)
        self.gate_projection = nn.Linear(channels, channels)
        self.output_projection = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f'Expected a BCHW tensor, got shape {tuple(x.shape)}.')
        if x.shape[1] != self.channels:
            raise ValueError(
                f'Expected {self.channels} channels, got {x.shape[1]} channels.'
            )

        feature_8d = self.directional_scan(x)
        residual = x.permute(0, 2, 3, 1).contiguous()
        feature_8d = feature_8d.permute(0, 2, 3, 1).contiguous()

        # Eq. (8): Linear(F_8d * Linear(LN(F))) + F.
        gate = self.gate_projection(self.gate_norm(residual))
        output = self.output_projection(feature_8d * gate) + residual
        return output.permute(0, 3, 1, 2).contiguous()
