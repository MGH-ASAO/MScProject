import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalizingFlow(nn.Module):
    def __init__(self, dim, n_flows=4):
        super(NormalizingFlow, self).__init__()
        self.flows = nn.ModuleList([AffineCouplingLayer(dim) for _ in range(n_flows)])

    def forward(self, x):
        log_det = torch.zeros(x.shape[0], device=x.device)
        for flow in self.flows:
            x, ld = flow(x, log_det)
            log_det += ld
        return x, log_det

    def inverse(self, z):
        for flow in reversed(self.flows):
            z = flow.inverse(z)
        return z


class AffineCouplingLayer(nn.Module):
    def __init__(self, dim):
        super(AffineCouplingLayer, self).__init__()
        self.dim = dim
        self.half_dim = dim // 2
        self.net = nn.Sequential(
            nn.Linear(self.half_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, self.half_dim * 2)  # Output scale and shift
        )
        self.register_buffer('mask', self._create_mask())

    def _create_mask(self):
        mask = torch.zeros(self.dim)
        mask[:self.half_dim] = 1
        return mask

    def forward(self, x, log_det):
        x_masked = x * self.mask
        nn_out = self.net(x_masked[:, :self.half_dim])
        s, t = nn_out.chunk(2, dim=1)

        s = torch.tanh(s)  # Scale factor
        t = t  # Translation factor

        z = x.clone()
        z[:, self.half_dim:] = x[:, self.half_dim:] * torch.exp(s) + t
        log_det += torch.sum(s, dim=1)

        return z, log_det

    def inverse(self, z):
        z_masked = z * self.mask
        nn_out = self.net(z_masked[:, :self.half_dim])
        s, t = nn_out.chunk(2, dim=1)

        s = torch.tanh(s)
        t = t

        x = z.clone()
        x[:, self.half_dim:] = (z[:, self.half_dim:] - t) * torch.exp(-s)

        return x


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=0.01)
        nn.init.zeros_(m.bias)