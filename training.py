from dataclasses import dataclass

import torch

from losses import (
    BandwidthLoss,
    GTGuidedSparsityLoss,
    NegativePearsonLoss,
    SparsityLoss,
    VarianceLoss,
    calc_psd,
)


@dataclass(frozen=True)
class MHSDConfig:
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    fps: float = 30.0
    fft_size: int = 256


@dataclass(frozen=True)
class MHSDLossWeights:
    bandwidth: float = 1.0
    sparsity: float = 1.0
    variance: float = 1.0
    temporal_consistency: float = 1.0
    spectral_consistency: float = 1.0


class MHSDLoss:
    def __init__(self, config: MHSDConfig = MHSDConfig()):
        self.fps = config.fps
        self.fft_size = config.fft_size
        self.bandwidth = BandwidthLoss(low_f=0.66, high_f=3.0)
        self.sparsity = SparsityLoss(low_f=0.66, high_f=3.0, freq_delta=0.1)
        self.variance = VarianceLoss(low_f=0.66, high_f=3.0)
        self.temporal_consistency = NegativePearsonLoss()
        self.spectral_consistency = GTGuidedSparsityLoss(
            low_f=0.66,
            high_f=3.0,
            freq_delta=0.1,
        )

    def __call__(
        self,
        reference_signal: torch.Tensor,
        motion_signal: torch.Tensor,
        weights: MHSDLossWeights = MHSDLossWeights(),
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        reference_psd, frequencies = calc_psd(
            reference_signal,
            n=self.fft_size,
            fs=self.fps,
        )
        teacher = reference_signal.detach()

        terms = {
            "bandwidth": self.bandwidth(reference_psd, frequencies),
            "sparsity": self.sparsity(reference_psd, frequencies),
            "variance": self.variance(reference_psd, frequencies),
            "temporal_consistency": self.temporal_consistency(motion_signal, teacher),
            "spectral_consistency": self.spectral_consistency(
                motion_signal,
                teacher,
                n=self.fft_size,
                fs=self.fps,
            ),
        }
        total = (
            weights.bandwidth * terms["bandwidth"]
            + weights.sparsity * terms["sparsity"]
            + weights.variance * terms["variance"]
            + weights.temporal_consistency * terms["temporal_consistency"]
            + weights.spectral_consistency * terms["spectral_consistency"]
        )
        return total, terms


def compute_loss(
    model,
    original_stmap: torch.Tensor,
    motion_augmented_stmap: torch.Tensor,
    loss_fn: MHSDLoss,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if hasattr(model, "forward_homologous_pair"):
        reference_signal, motion_signal = model.forward_homologous_pair(
            original_stmap,
            motion_augmented_stmap,
        )
    else:
        reference_signal = model(original_stmap)
        motion_signal = model(motion_augmented_stmap)
    return loss_fn(reference_signal, motion_signal)


def build_optimizer(model, config: MHSDConfig = MHSDConfig()):
    return torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=config.weight_decay,
    )
