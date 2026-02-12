#!/usr/bin/env python3
"""
Train an EDM-style diffusion model (Karras et al., 2022) on latent vectors saved as .npy.

Barebones on purpose:
- Latents: numpy array [N, D]
- Network: MLP denoiser with EDM preconditioning
- Training: sample sigma ~ log-normal, predict denoised latent
- Sampling: EDM sampler with Heun (2nd order), optional "churn"

This matches the "EDM-style diffusion" commonly used in latent diffusion pipelines.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils import set_seed


# ----------------------------
# Model
# ----------------------------
class FourierEmbed(nn.Module):
    """Small Fourier feature embedder for a scalar (here: log(sigma))."""
    def __init__(self, out_dim: int = 128, max_freq: float = 10.0):
        super().__init__()
        assert out_dim % 2 == 0
        half = out_dim // 2
        freqs = torch.exp(torch.linspace(0, math.log(max_freq), half))
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B] or [B,1]
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        x = x[:, None] * self.freqs[None, :]  # [B, half]
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # [B, out_dim]


class DenoiserMLP(nn.Module):
    """
    f_theta used by EDM preconditioning.

    Input: (c_in * x) concatenated with embedding of c_noise
    Output: residual used to build D(x, sigma)
    """
    def __init__(self, dim: int, hidden: int = 2048, layers: int = 3, noise_embed_dim: int = 128):
        super().__init__()
        self.noise_embed = FourierEmbed(out_dim=noise_embed_dim)
        in_dim = dim + noise_embed_dim
        blocks = []
        cur = in_dim
        for _ in range(layers):
            blocks += [nn.Linear(cur, hidden), nn.SiLU()]
            cur = hidden
        blocks += [nn.Linear(cur, dim)]
        self.net = nn.Sequential(*blocks)

    def forward(self, x_in: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        ne = self.noise_embed(c_noise)  # [B, noise_embed_dim]
        x = torch.cat([x_in, ne], dim=-1)
        return self.net(x)


class EDMPrecond(nn.Module):
    """
    EDM preconditioning wrapper to produce the denoised estimate D(x, sigma).

    D(x, sigma) = c_skip * x + c_out * f_theta(c_in * x, c_noise)
    """
    def __init__(self, denoiser: nn.Module, sigma_data: float = 0.5):
        super().__init__()
        self.denoiser = denoiser
        self.sigma_data = float(sigma_data)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # sigma: [B] or [B,1]
        if sigma.ndim == 2 and sigma.shape[1] == 1:
            sigma = sigma[:, 0]
        sigma = sigma.to(x.dtype)

        sd = self.sigma_data
        sigma2 = sigma * sigma
        sd2 = sd * sd

        c_skip = sd2 / (sigma2 + sd2)
        c_out  = sigma * sd / torch.sqrt(sigma2 + sd2)
        c_in   = 1.0 / torch.sqrt(sigma2 + sd2)
        c_noise = torch.log(sigma) / 4.0

        x_in = x * c_in[:, None]
        f = self.denoiser(x_in, c_noise)
        d = c_skip[:, None] * x + c_out[:, None] * f
        return d


# ----------------------------
# EDM training utils
# ----------------------------
def sample_sigmas(batch: int, sigma_min: float, sigma_max: float, device: torch.device) -> torch.Tensor:
    """
    EDM commonly samples log(sigma) ~ Uniform(log(sigma_min), log(sigma_max)).
    """
    u = torch.rand(batch, device=device)
    return torch.exp(torch.log(torch.tensor(sigma_min, device=device)) * (1 - u) +
                     torch.log(torch.tensor(sigma_max, device=device)) * u)


def loss_weight(sigma: torch.Tensor, sigma_data: float) -> torch.Tensor:
    """
    EDM loss weighting (Karras et al. 2022):
      lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
    """
    sd = sigma_data
    return (sigma * sigma + sd * sd) / ((sigma * sd) ** 2)


# ----------------------------
# EDM sampler (Heun + optional churn)
# ----------------------------
@torch.no_grad()
def edm_sample(
    model: EDMPrecond,
    dim: int,
    n: int,
    steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
    S_churn: float = 0.0,
    S_min: float = 0.0,
    S_max: float = float("inf"),
    S_noise: float = 1.0,
) -> torch.Tensor:
    """
    EDM sampler (Algorithm 2 style) using Heun's method.

    Returns: [n, dim] samples.
    """
    model.eval()

    # Discretize sigma schedule
    i = torch.arange(steps, device=device)
    sigmas = (sigma_max ** (1 / rho) + i / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])])  # append sigma=0

    x = torch.randn(n, dim, device=device) * sigmas[0]

    for j in range(steps):
        sigma_cur = sigmas[j]
        sigma_next = sigmas[j + 1]

        # Churn (optional)
        gamma = 0.0
        if S_churn > 0 and (sigma_cur >= S_min) and (sigma_cur <= S_max):
            gamma = min(S_churn / steps, math.sqrt(2) - 1)

        sigma_hat = sigma_cur * (1 + gamma)
        if gamma > 0:
            eps = torch.randn_like(x) * S_noise
            x = x + torch.sqrt(torch.clamp(sigma_hat**2 - sigma_cur**2, min=0.0)) * eps

        # Euler step
        sigma_batch = torch.full((n,), float(sigma_hat), device=device)
        den = model(x, sigma_batch)
        d = (x - den) / sigma_hat  # score-like derivative of x wrt sigma
        x_euler = x + (sigma_next - sigma_hat) * d

        # Heun correction (2nd order)
        if sigma_next != 0:
            sigma_batch_next = torch.full((n,), float(sigma_next), device=device)
            den_next = model(x_euler, sigma_batch_next)
            d_next = (x_euler - den_next) / sigma_next
            x = x + (sigma_next - sigma_hat) * (0.5 * d + 0.5 * d_next)
        else:
            x = x_euler

    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", type=str, required=True, help="Path to latents .npy, shape [N,D]")
    ap.add_argument("--out_dir", type=str, required=True)

    # training
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--sigma_data", type=float, default=0.5)
    ap.add_argument("--sigma_min", type=float, default=0.002)
    ap.add_argument("--sigma_max", type=float, default=80.0)

    # sampling
    ap.add_argument("--sample_every", type=int, default=10)
    ap.add_argument("--n_samples", type=int, default=16)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--rho", type=float, default=7.0)
    ap.add_argument("--S_churn", type=float, default=0.0)
    ap.add_argument("--S_min", type=float, default=0.0)
    ap.add_argument("--S_max", type=float, default=float("inf"))
    ap.add_argument("--S_noise", type=float, default=1.0)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    z = np.load(args.latents)
    if z.ndim != 2:
        raise ValueError(f"Expected latents [N,D], got {z.shape}")
    z = torch.from_numpy(z).float()
    dim = z.shape[1]

    dl = DataLoader(TensorDataset(z), batch_size=args.batch_size, shuffle=True, drop_last=True)

    den = DenoiserMLP(dim=dim, hidden=args.hidden, layers=args.layers).to(device)
    model = EDMPrecond(den, sigma_data=args.sigma_data).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(dl, desc=f"epoch {epoch}/{args.epochs}")
        for (z0,) in pbar:
            z0 = z0.to(device)
            b = z0.shape[0]

            sigma = sample_sigmas(b, args.sigma_min, args.sigma_max, device=device)
            noise = torch.randn_like(z0)
            zt = z0 + sigma[:, None] * noise  # forward diffusion with continuous sigma

            z_denoised = model(zt, sigma)
            w = loss_weight(sigma, args.sigma_data)[:, None]
            loss = torch.mean(w * (z_denoised - z0) ** 2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=float(loss.detach().cpu()))

        # save checkpoint
        ckpt = {
            "state_dict": model.state_dict(),
            "dim": dim,
            "sigma_data": args.sigma_data,
            "sigma_min": args.sigma_min,
            "sigma_max": args.sigma_max,
            "rho": args.rho,
            "seed": args.seed,
        }
        torch.save(ckpt, out_dir / "edm_ckpt.pt")

        # periodic samples
        if (epoch % args.sample_every) == 0 or epoch == args.epochs:
            zs = edm_sample(
                model=model,
                dim=dim,
                n=args.n_samples,
                steps=args.steps,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                rho=args.rho,
                device=device,
                S_churn=args.S_churn,
                S_min=args.S_min,
                S_max=args.S_max,
                S_noise=args.S_noise,
            ).detach().cpu().numpy()
            np.save(out_dir / f"samples_epoch{epoch:04d}.npy", zs)

    print(f"Saved EDM checkpoint + samples to: {out_dir}")


if __name__ == "__main__":
    main()
