# Barebones INR + Latent Diffusion (for LGE / 3D volumes)

This is a minimal repo containing:
- `fit_inr.py`: fit a SIREN INR to a **single** 3D volume (and optionally two masks)
- `data.py`: tiny loader utilities (`.nii/.nii.gz` via nibabel; `.mha/.mhd` via SimpleITK)
- `train_diffusion.py`: train an EDM-style diffusion model (Karras et al., 2022) on latent vectors (`.npy`)

This is intentionally kept small so you can build from here without carrying messy legacy code.

## Install

Option 1 (pip):
```bash
pip install torch numpy tqdm nibabel
# for .mha/.mhd support:
pip install SimpleITK
```

## 1) Fit an INR on one case

```bash
python fit_inr.py \
  --lge_path /path/to/lge.nii.gz \
  --myo_path /path/to/myo_mask.nii.gz \
  --fibro_path /path/to/fibro_mask.nii.gz \
  --out outputs/inr_case001.pt \
  --steps 3000
```

Notes:
- The model predicts channels `[LGE, myo, fibro]` (if masks are provided).
- LGE uses MSE loss; masks use BCEWithLogits.

## 2) Train diffusion on latents

Assuming you have an `.npy` file containing latents with shape `[N, D]`:

```bash
python train_diffusion.py \
  --latents /path/to/z_train.npy \
  --out_dir outputs/diffusion/ \
  --epochs 200 \
  --steps 32 \
  --rho 7.0
```

This saves:
- `outputs/diffusion/diffusion_ckpt.pt`
- `outputs/diffusion/samples_epoch.npy` (16 sampled latents)

## Using INR2VEC

If you encoded INR parameters to latents using INR2VEC, keep that as an external dependency and only store:
- the **latent `.npy`** file you produced
- the exact INR2VEC commit hash you used (for reproducibility)

## Privacy

Remove or anonymize any patient/site identifiers in:
- file/folder names
- hardcoded paths
- notebooks

This repo does not include any patient data.
