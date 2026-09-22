"""
Coronary artery / vessel enhancement stage.

Uses fast morphological multi-scale top-hat filtering to boost thin,
elongated bright structures (contrast-filled coronary arteries) while
suppressing everything else. This is a well-established, computationally
cheap technique used in digital subtraction angiography (DSA) and X-ray
coronary angiography enhancement, and is far cheaper than Hessian/Frangi
vesselness filtering while still giving strong vessel contrast at
real-time frame rates.
"""

from __future__ import annotations

import numpy as np
import cv2

# Linear structuring elements at multiple orientations approximate vessel
# directionality without the cost of a full multi-scale Hessian analysis.
# Kept small (4 directions) so the pipeline stays comfortably inside the
# 36 ms real-time budget; see docs/PIPELINE.md for the speed/quality tradeoff.
_ANGLES_DEG = (0, 45, 90, 135)
_KERNEL_CACHE: dict[tuple[int, float], np.ndarray] = {}


def _linear_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Create (and cache) a thin linear structuring element rotated to angle_deg."""
    key = (length, angle_deg)
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    size = length if length % 2 == 1 else length + 1
    kernel = np.zeros((size, size), dtype=np.uint8)
    kernel[size // 2, :] = 1
    center = (size / 2, size / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(kernel, M, (size, size), flags=cv2.INTER_NEAREST)
    _KERNEL_CACHE[key] = rotated
    return rotated


def multiscale_vessel_tophat(img_float: np.ndarray, lengths=(13,)) -> np.ndarray:
    """
    Multi-directional (single-scale by default) white top-hat to enhance
    thin bright vessel-like structures (coronary arteries).

    Structuring-element kernels are cached (built once, reused every frame)
    so only the morphological opening itself is paid for per frame. Uses
    in-place cv2 ops (cv2.subtract / cv2.max) to avoid extra NumPy array
    allocations in the per-frame hot path, and defaults to a single scale
    (kept configurable) to stay comfortably inside the 36 ms real-time
    budget -- see docs/PIPELINE.md for the speed/quality tradeoff.
    """
    response = np.zeros_like(img_float, dtype=np.float32)
    scale_buf = np.empty_like(img_float, dtype=np.float32)
    tophat_buf = np.empty_like(img_float, dtype=np.float32)

    for length in lengths:
        scale_buf.fill(0.0)
        for angle in _ANGLES_DEG:
            kernel = _linear_kernel(length, angle)
            opened = cv2.morphologyEx(img_float, cv2.MORPH_OPEN, kernel)
            cv2.subtract(img_float, opened, dst=tophat_buf)
            cv2.max(scale_buf, tophat_buf, dst=scale_buf)
        cv2.add(response, scale_buf, dst=response)

    if len(lengths) > 1:
        response /= len(lengths)
    return np.clip(response, 0.0, 1.0, out=response)


def enhance_contrast(img_float: np.ndarray, gamma: float = 0.8) -> np.ndarray:
    """Gamma correction + min-max stretch to boost visibility of enhanced vessels."""
    img = np.clip(img_float, 0.0, 1.0)
    mn, mx = float(img.min()), float(img.max())
    if mx - mn > 1e-8:
        cv2.subtract(img, mn, dst=img)
        cv2.multiply(img, 1.0 / (mx - mn), dst=img)
    cv2.pow(img, gamma, dst=img)
    return img


def enhance_vessels(background_suppressed_float: np.ndarray) -> np.ndarray:
    """
    Full vessel-enhancement stage applied to the background-suppressed frame.
    Returns float32 [0,1] "Enhanced Coronary Artery Image".
    """
    tophat_response = multiscale_vessel_tophat(background_suppressed_float)
    enhanced = enhance_contrast(tophat_response, gamma=0.7)
    return enhanced