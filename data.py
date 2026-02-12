"""
Example data utilities.

This is intentionally minimal and does NOT assume a specific dataset structure.
You can adapt `load_volume()` for your file type.

Supported by default:
- .nii / .nii.gz via nibabel
- .mha / .mhd via SimpleITK (if installed)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch

def _load_nib(path: Path) -> np.ndarray:
    import nibabel as nib
    img = nib.load(str(path))
    arr = img.get_fdata(dtype=np.float32)
    return arr

def _load_sitk(path: Path) -> np.ndarray:
    import SimpleITK as sitk
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # [D,H,W]
    return arr

def load_volume(path: Path) -> np.ndarray:
    suf = "".join(path.suffixes).lower()
    if suf.endswith(".nii") or suf.endswith(".nii.gz"):
        return _load_nib(path)
    if suf.endswith(".mha") or suf.endswith(".mhd"):
        return _load_sitk(path)
    raise ValueError(f"Unsupported file type: {path}")

def _normalize_lge(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    # simple robust scaling to [0,1]
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    x = np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0)
    return x.astype(np.float32)

def _to_binary(x: np.ndarray, thr: float = 0.5) -> np.ndarray:
    return (x > thr).astype(np.float32)

def load_case_example(
    lge_path: Path,
    myo_path: Optional[Path] = None,
    fibro_path: Optional[Path] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any]]:
    lge = load_volume(lge_path)
    lge = _normalize_lge(lge)

    myo = None
    if myo_path is not None and str(myo_path) != "":
        myo = _to_binary(load_volume(myo_path))

    fibro = None
    if fibro_path is not None and str(fibro_path) != "":
        fibro = _to_binary(load_volume(fibro_path))

    meta = {
        "lge_path": str(lge_path),
        "myo_path": str(myo_path) if myo_path else "",
        "fibro_path": str(fibro_path) if fibro_path else "",
    }

    # ensure [D,H,W]
    if lge.ndim == 4 and lge.shape[0] == 1:
        lge = lge[0]
    if lge.ndim != 3:
        raise ValueError(f"Expected 3D volume [D,H,W], got {lge.shape}")

    lge_t = torch.from_numpy(lge).float()
    myo_t = torch.from_numpy(myo).float() if myo is not None else None
    fibro_t = torch.from_numpy(fibro).float() if fibro is not None else None
    return lge_t, myo_t, fibro_t, meta
