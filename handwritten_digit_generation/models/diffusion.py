# handwritten_digit_generation/models/diffusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from handwritten_digit_generation.utils.training_utils import (
    betas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
    posterior_variance, sqrt_recip_alphas, alphas_cumprod_prev, alphas_cumprod
)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DiffusionModel(nn.Module):
    def __init__(self, input_channels=1, time_dim=256, hidden_dim=64, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.conv_in = ConvBlock(input_channels, hidden_dim)
        self.down1 = ConvBlock(hidden_dim, hidden_dim * 2)
        self.down2 = ConvBlock(hidden_dim * 2, hidden_dim * 4)

        self.bottleneck1 = ConvBlock(hidden_dim * 4 + time_dim, hidden_dim * 4)
        self.bottleneck2 = ConvBlock(hidden_dim * 4, hidden_dim * 4)

        self.up1 = ConvBlock(hidden_dim * 6, hidden_dim * 2)
        self.up2 = ConvBlock(hidden_dim * 3, hidden_dim)

        self.conv_out = nn.Conv2d(hidden_dim, input_channels, 3, padding=1)

        # Move precomputed tensors to the correct device
        self.sqrt_recip_alphas = sqrt_recip_alphas.view(-1).to(self.device)
        self.alphas_cumprod_prev = alphas_cumprod_prev.view(-1).to(self.device)
        self.alphas_cumprod = alphas_cumprod.view(-1).to(self.device)
        self.posterior_variance = posterior_variance.view(-1).to(self.device)
        self.betas = betas.view(-1).to(self.device)
        self.sqrt_alphas_cumprod = sqrt_alphas_cumprod.view(-1).to(self.device)

    def forward(self, x, t):
        # print("Input x shape:", x.shape)
        # print("Input t shape:", t.shape)

        # Check if the input is 2D, if so, reshape it to 4D
        if x.dim() == 2:
            x = x.view(x.size(0), 1, 28, 28)  # Assume the input is a MNIST image (28x28)

        # Save original shape
        original_shape = x.shape

        # Make sure t is two-dimensional
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        # print("Reshaped t shape:", t.shape)
        time_emb = self.time_mlp(t.float())
        # print("Time embedding shape:", time_emb.shape)

        x1 = self.conv_in(x)
        x2 = self.down1(F.max_pool2d(x1, 2))
        x3 = self.down2(F.max_pool2d(x2, 2))

        # Adjust the shape of time_emb to match x3
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x3.shape[2], x3.shape[3])

        x3 = torch.cat([x3, time_emb], dim=1)

        x3 = self.bottleneck1(x3)
        x3 = self.bottleneck2(x3)

        x = F.interpolate(x3, scale_factor=2, mode='nearest')
        x = self.up1(torch.cat([x, x2], dim=1))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.up2(torch.cat([x, x1], dim=1))

        x = self.conv_out(x)

        # If the original input is 2D, re-flatten the output
        if original_shape[1] == 784:
            x = x.view(x.size(0), -1)

        return x

    def sample(self, num_samples, device, T=300):
        # print("Starting sampling process")
        # Generate initial noise
        x = torch.randn((num_samples, 1, 28, 28)).to(device)

        for i in reversed(range(T)):
            # print(f"Sampling step {T - i}/{T}")
            t = torch.full((num_samples, 1), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t)

        return x

    def p_sample(self, x, t):
        # print("p_sample input x shape:", x.shape)
        # print("p_sample input t shape:", t.shape)

        t = t.to(self.device)
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        x_recon = self(x, t)

        # print("x_recon shape:", x_recon.shape)

        posterior_mean = (
                                 self.sqrt_recip_alphas[t].view(-1, 1, 1, 1) *
                                 (1 - self.alphas_cumprod_prev[t]).view(-1, 1, 1, 1) /
                                 (1 - self.alphas_cumprod[t]).view(-1, 1, 1, 1)
                         ) * x_recon + (
                                 self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) *
                                 self.betas[t].view(-1, 1, 1, 1) /
                                 (1 - self.alphas_cumprod[t]).view(-1, 1, 1, 1)
                         ) * x

        posterior_variance = self.posterior_variance[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(x)

        return posterior_mean + torch.sqrt(posterior_variance) * noise


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_0, t, device="cpu"):
    # If the input is 4D, flatten it
    original_shape = x_0.shape
    if len(original_shape) == 4:
        x_0 = x_0.view(x_0.shape[0], -1)

    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)

    x_noisy = sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
              + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)

    # If the original input was 4D, reshape the result back to the original shape
    if len(original_shape) == 4:
        x_noisy = x_noisy.view(original_shape)
        noise = noise.view(original_shape)

    return x_noisy, noise

# # Define beta schedule
# T = 300
# betas = linear_beta_schedule(timesteps=T)
#
# # Pre-calculate different terms for closed form
# alphas = 1. - betas
# alphas_cumprod = torch.cumprod(alphas, axis=0)
# alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
# sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
# sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
# sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
# posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
