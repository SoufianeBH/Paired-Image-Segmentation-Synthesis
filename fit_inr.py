#!/usr/bin/env python3
"""
Fit a simple joint INR (SIREN) to a single 3D LGE volume and (optionally) two masks.

This is intentionally barebones:
- load one case from disk (see data.py)
- create normalized coordinate grid in [-1, 1]^3
- train a SIREN to regress channels: [LGE, myo_mask, fibro_mask]
- save a checkpoint with model weights + metadata

Expected shapes:
- LGE:   [D, H, W] float32
- masks: [D, H, W] {0,1} float32 (optional)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model import OccupancySIREN
from utils import set_seed
from data import load_case_example


class PositionalEncoding:
    """Simple Fourier features positional encoding."""
    def __init__(self, num_frequencies: int = 10, include_input: bool = True):
        self.num_frequencies = int(num_frequencies)
        self.include_input = bool(include_input)

    def get_output_dim(self, input_dim: int) -> int:
        base = input_dim if self.include_input else 0
        return base + 2 * input_dim * self.num_frequencies

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., 3] in [-1,1]
        outs = [x] if self.include_input else []
        for i in range(self.num_frequencies):
            freq = 2.0 ** i
            outs.append(torch.sin(freq * x))
            outs.append(torch.cos(freq * x))
        return torch.cat(outs, dim=-1)


def make_coord_grid(shape_dhw: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    """Return coords with shape [N, 3] in [-1,1]^3 for a D×H×W grid."""
    D, H, W = shape_dhw
    zs = torch.linspace(-1, 1, D, device=device)
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
    coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)  # [N,3]
    return coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lge_path", type=str, required=True, help="Path to LGE volume (e.g., .mha/.nii.gz)")
    ap.add_argument("--myo_path", type=str, default="", help="Optional myocardium mask path")
    ap.add_argument("--fibro_path", type=str, default="", help="Optional fibrosis mask path")
    ap.add_argument("--out", type=str, required=True, help="Output checkpoint .pt path")
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--num_layers", type=int, default=5)
    ap.add_argument("--num_freq", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=65536)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # ---- load one case
    lge, myo, fibro, meta = load_case_example(
        lge_path=Path(args.lge_path),
        myo_path=Path(args.myo_path) if args.myo_path else None,
        fibro_path=Path(args.fibro_path) if args.fibro_path else None,
    )
    # lge/myo/fibro: torch [D,H,W]
    D, H, W = lge.shape
    coords = make_coord_grid((D, H, W), device=device)

    # targets: [N, C]
    chans = [lge]
    if myo is not None:
        chans.append(myo)
    if fibro is not None:
        chans.append(fibro)
    target = torch.stack(chans, dim=0).float()  # [C,D,H,W]
    target = target.reshape(target.shape[0], -1).T.contiguous()  # [N,C]

    # move
    target = target.to(device)

    # ---- model
    pe = PositionalEncoding(num_frequencies=args.num_freq, include_input=True)
    model = OccupancySIREN(input_dim=3, hidden_dim=args.hidden_dim, output_dim=target.shape[1],
                           num_layers=args.num_layers, pos_encoding=pe).to(device)

    # ---- data loader over coordinates (coords kept on device)
    ds = TensorDataset(coords, target)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # losses: L2 for LGE, BCE for masks (if present)
    l2 = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    pbar = tqdm(range(args.steps), desc="fit_inr")
    it = iter(dl)
    for step in pbar:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl)
            x, y = next(it)

        pred = model(x)

        # channel 0 = LGE regression
        loss = l2(pred[:, 0], y[:, 0])

        # channels 1.. = masks with BCE (logits)
        if y.shape[1] > 1:
            loss = loss + bce(pred[:, 1:], y[:, 1:])

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 50 == 0:
            pbar.set_postfix(loss=float(loss.detach().cpu()))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "meta": {
            "shape": [int(D), int(H), int(W)],
            "channels": int(target.shape[1]),
            "posenc_num_freq": int(args.num_freq),
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "seed": int(args.seed),
            **meta,
        }
    }, out_path)
    print(f"Saved INR checkpoint to: {out_path}")


if __name__ == "__main__":
    main()
