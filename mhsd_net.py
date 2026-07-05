import torch.nn as nn

from model.physnet2d import PhysNet2D


class MHSDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.signal_recovery_network = PhysNet2D()

    def forward(self, stmap):
        return self.signal_recovery_network(stmap)

    def forward_homologous_pair(self, original_stmap, motion_augmented_stmap):
        reference_signal = self.forward(original_stmap)
        motion_signal = self.forward(motion_augmented_stmap)
        return reference_signal, motion_signal
