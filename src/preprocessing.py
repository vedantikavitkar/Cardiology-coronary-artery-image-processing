"""
Preprocessing stage: normalization, fast denoising, contrast setup.

Operates on 16-bit grayscale frames and prepares them for the
background-suppression and vessel-enhancement stages while keeping the
per-frame cost small enough to fit inside the 36 ms real-time budget.
"""

from __future__ import annotations

import numpy as np
import cv2


def normalize_16bit(img: np.ndarray) -> np.ndarray:
    """Convert a 16-bit (0-65535) frame to float32 in [0, 1]."""
    if img.dtype != np.uint16:
        img = img.astype(np.uint16)
    return img.astype(np.float32) / 65535.0


def denoise(img_float: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Fast edge-preserving-ish denoise using a median blur.

    A median blur (O(n) per pixel via OpenCV's optimized implementation)
    removes shot/quantum noise spikes while a full bilateral filter is
    avoided here to stay within the latency budget on large frames.
    """
    # cv2.medianBlur requires uint8/uint16/float32 with odd ksize.
    return cv2.medianBlur(img_float, ksize)


def clahe_enhance(img_uint8: np.ndarray, clip_limit: float = 2.5,
                   tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (8-bit)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img_uint8)


def to_uint8(img_float: np.ndarray) -> np.ndarray:
    """Scale a float32 [0,1]-ish array to uint8 [0,255] for display/CLAHE."""
    img = img_float.copy()
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    img = (img - mn) / (mx - mn)
    return (img * 255.0).astype(np.uint8)


def preprocess(raw_16bit: np.ndarray) -> np.ndarray:
    """Full preprocessing chain: normalize + denoise. Returns float32 [0,1]."""
    normed = normalize_16bit(raw_16bit)
    denoised = denoise(normed, ksize=3)
    return denoised