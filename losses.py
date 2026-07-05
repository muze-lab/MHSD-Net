

import torch
import torch.nn as nn


class NegativePearsonLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, est_bvp, gt_bvp):
        if len(est_bvp.shape) == 1:
            est_bvp = est_bvp.unsqueeze(dim=0)

        batch_size, length = est_bvp.shape[0], est_bvp.shape[1]
        loss = 0.0
        for i in range(batch_size):
            sum_x = torch.sum(est_bvp[i])
            sum_y = torch.sum(gt_bvp[i])
            sum_xy = torch.sum(est_bvp[i] * gt_bvp[i])
            sum_x2 = torch.sum(torch.pow(est_bvp[i], 2))
            sum_y2 = torch.sum(torch.pow(gt_bvp[i], 2))
            num = length * sum_xy - sum_x * sum_y
            den = torch.sqrt(
                torch.clamp(
                    (length * sum_x2 - torch.pow(sum_x, 2))
                    * (length * sum_y2 - torch.pow(sum_y, 2)),
                    min=1e-12,
                )
            )
            pearson = num / (den + 1e-8)
            loss += 1 - pearson
        return loss / batch_size


def calc_psd(x, n=256, fs=30.0):
    x_fft = torch.fft.rfft(x.float(), n=n, norm="forward")
    psd = torch.abs(x_fft) ** 2
    psd = psd / (torch.sum(psd, dim=-1, keepdim=True) + 1e-10)
    freqs = torch.fft.rfftfreq(n, 1 / fs).to(x.device)
    return psd, freqs


def split_psd(psd, freqs, low_f=0.66, high_f=3.0):
    physio_mask = (freqs >= low_f) & (freqs <= high_f)
    physio_psd = psd[..., physio_mask]
    irrel_psd = psd[..., ~physio_mask]
    return physio_psd, irrel_psd


def filter_physio(psd, freqs, low_f=0.66, high_f=3.0):
    physio_mask = (freqs >= low_f) & (freqs <= high_f)
    physio_psd = psd[..., physio_mask]
    physio_freqs = freqs[physio_mask]
    return physio_psd, physio_freqs


class GTGuidedSparsityLoss(nn.Module):
    def __init__(self, low_f=0.66, high_f=3.0, freq_delta=0.1):
        super().__init__()
        self.low_f = low_f
        self.high_f = high_f
        self.freq_delta = freq_delta
        self.epsilon = 1e-10

    def forward(self, rppg_batch, ppg_batch, n=256, fs=30.0):
        rppg_psd, freqs = calc_psd(rppg_batch, n=n, fs=fs)
        ppg_psd, _ = calc_psd(ppg_batch, n=n, fs=fs)

        physio_mask = (freqs >= self.low_f) & (freqs <= self.high_f)
        rppg_physio_psd = rppg_psd[..., physio_mask]
        ppg_physio_psd = ppg_psd[..., physio_mask]
        physio_freqs = freqs[physio_mask]

        if physio_freqs.shape[-1] == 0:
            return torch.tensor(0.0, device=rppg_psd.device)

        peak_idx = torch.argmax(ppg_physio_psd, dim=-1)
        gt_peak_freq = physio_freqs[peak_idx]

        batch_size = rppg_physio_psd.shape[0]
        freq_grid = physio_freqs.unsqueeze(0).expand(batch_size, -1)
        gt_peak_freq_expand = gt_peak_freq.unsqueeze(1)
        low_cut = gt_peak_freq_expand - self.freq_delta
        high_cut = gt_peak_freq_expand + self.freq_delta

        signal_mask = (freq_grid >= low_cut) & (freq_grid <= high_cut)
        noise_mask = ~signal_mask
        signal_energy = torch.sum(rppg_physio_psd * signal_mask.float(), dim=-1)
        noise_energy = torch.sum(rppg_physio_psd * noise_mask.float(), dim=-1)
        physio_total_energy = signal_energy + noise_energy + self.epsilon

        noise_ratio = noise_energy / physio_total_energy
        return torch.mean(noise_ratio)


class BandwidthLoss(nn.Module):
    def __init__(self, low_f=0.66, high_f=3.0):
        super().__init__()
        self.low_f = low_f
        self.high_f = high_f
        self.epsilon = 1e-10

    def forward(self, psd, freqs):
        physio_psd, irrel_psd = split_psd(psd, freqs, self.low_f, self.high_f)

        physio_energy = torch.sum(physio_psd, dim=-1)
        irrel_energy = torch.sum(irrel_psd, dim=-1)
        total_energy = physio_energy + irrel_energy + self.epsilon

        irrel_ratio = irrel_energy / total_energy
        return torch.mean(irrel_ratio)


class SparsityLoss(nn.Module):
    def __init__(self, low_f=0.66, high_f=3.0, freq_delta=0.1):
        super().__init__()
        self.low_f = low_f
        self.high_f = high_f
        self.freq_delta = freq_delta
        self.epsilon = 1e-10

    def forward(self, psd, freqs):
        physio_psd, physio_freqs = filter_physio(
            psd, freqs, low_f=self.low_f, high_f=self.high_f
        )

        if physio_psd.shape[-1] == 0:
            return torch.tensor(0.0, device=psd.device)

        peak_idx = torch.argmax(physio_psd, dim=-1)
        peak_freq = physio_freqs[peak_idx]

        batch_size = physio_psd.shape[0]
        freq_grid = physio_freqs.unsqueeze(0).expand(batch_size, -1)
        peak_freq_expand = peak_freq.unsqueeze(1)

        low_cut = peak_freq_expand - self.freq_delta
        high_cut = peak_freq_expand + self.freq_delta

        signal_mask = (freq_grid >= low_cut) & (freq_grid <= high_cut)
        noise_mask = ~signal_mask

        signal_energy = torch.sum(physio_psd * signal_mask.float(), dim=-1)
        noise_energy = torch.sum(physio_psd * noise_mask.float(), dim=-1)
        physio_total_energy = signal_energy + noise_energy + self.epsilon

        noise_ratio = noise_energy / physio_total_energy
        return torch.mean(noise_ratio)


class VarianceLoss(nn.Module):
    def __init__(self, low_f=0.66, high_f=3.0):
        super().__init__()
        self.low_f = low_f
        self.high_f = high_f

    def forward(self, psd, freqs):
        physio_psd, _ = filter_physio(psd, freqs, low_f=self.low_f, high_f=self.high_f)

        if physio_psd.shape[-1] == 0:
            return torch.tensor(0.0, device=psd.device)

        physio_psd = physio_psd / (
            torch.sum(physio_psd, dim=-1, keepdim=True) + 1e-10
        )
        physio_psd_mean = torch.mean(physio_psd, dim=0)
        num_bins = physio_psd.shape[-1]
        expected = torch.full(
            (num_bins,),
            1.0 / num_bins,
            device=psd.device,
            dtype=physio_psd_mean.dtype,
        )

        cum_psd = torch.cumsum(physio_psd_mean, dim=0)
        cum_expected = torch.cumsum(expected, dim=0)
        return torch.mean(torch.square(cum_psd - cum_expected))
