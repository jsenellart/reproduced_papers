from __future__ import annotations

import logging
import merlin as ml
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class PhotonicQRNNConfig:
    kd: int  # logical "data" photons
    kh: int  # logical "hidden" photons
    depth: int = 2  # number of QRB layers per time step
    shots: int = 0  # 0 => full statevector
    dtype: Optional[torch.dtype] = None


class PhotonicQRNNCell(nn.Module):
    """
    Placeholder MerLin-based QRNN cell.

    This sketches the intended structure:
    - D register: 2*kd modes populated with k_d photons (input encoding)
    - H register: 2*kh modes populated with k_h photons (hidden state)
    - U_in(x): phase column encoding of x(t) sandwiched by trainable interferometers W1/W2
    - V: trainable interferometer on concatenated D+H modes
    - partial measurement on D to enforce k_d photons
    """

    def __init__(self, input_size: int, config: PhotonicQRNNConfig) -> None:
        super().__init__()
        self.input_size = input_size
        self.config = config

        self._dtype = config.dtype if config.dtype is not None else torch.float64

        self.available = False
        logger.info(
            "PhotonicQRNNCell placeholder initialized (kd=%d, kh=%d, depth=%d, shots=%d). "
            "Full construction of U_in/W1/W2/V interferometers is not yet implemented.",
            config.kd,
            config.kh,
            config.depth,
            config.shots,
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "PhotonicQRNNCell is a placeholder. Implement MerLin interferometer stack and "
            "photon-preserving measurement pipeline before use."
        )


class PhotonicQRNNRegressor(nn.Module):
    """Unrolled photonic QRNN placeholder to match the baseline API."""

    def __init__(self, input_size: int, config: PhotonicQRNNConfig) -> None:
        super().__init__()
        self.cell = PhotonicQRNNCell(input_size, config)
        self.hidden_dim = config.kh

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - placeholder
        raise NotImplementedError(
            "PhotonicQRNNRegressor forward is not implemented; wiring depends on MerLin circuit."
        )


__all__ = ["PhotonicQRNNConfig", "PhotonicQRNNCell", "PhotonicQRNNRegressor"]
