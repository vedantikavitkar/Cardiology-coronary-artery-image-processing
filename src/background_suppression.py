"""
Background structure suppression: ribs, spine, lung fields, background noise.

Approach
--------
Ribs, spine, and lung fields are large, smoothly-varying (low spatial
frequency) anatomical structures compared to the thin, high-frequency
coronary artery vessels. This module exploits that frequency separation
using a fast homomorphic/high-pass strategy that is safe for real-time
(<=36 ms) operation:

  1. Estimate the background (low-frequency) component with a large-kernel
     Gaussian blur (separable => O(n) and very fast in OpenCV).
  2. Subtract the estimated background from the original frame (unsharp
     masking / high-pass filtering) to suppress ribs/spine/lungs while
     retaining thin vessel edges.
  3. Apply a morphological opening (large elliptical structuring element)
     as a second, complementary background estimate and subtract it too
     (a fast approximation of grayscale top-hat filtering), which further
     removes rib/spine remnants that Gaussian blur alone may not fully
     remove.
"""

from __future__ import annotations

import numpy as np
import cv2


def estimate_background_gaussian(img_float: np.ndarray, sigma: float = 25.0,
                                  pyramid_levels: int = 2) -> np.ndarray:
    """
    Large-scale low-frequency background estimate (ribs/spine/lungs).

    A true full-resolution Gaussian blur with a large sigma requires a huge
    kernel (O(sigma) size, applied over the whole image) which is too slow
    for the real-time budget. Instead we approximate it cheaply using an
    image-pyramid trick: downsample the image (cv2.pyrDown, each halving
    width/height and cost by ~4x), blur at the much smaller scale with a
    proportionally smaller sigma, then upsample back (cv2.pyrUp). This
    preserves the same effective spatial low-pass behaviour at a fraction
    of the compute cost.
    """
    small = img_float
    for _ in range(pyramid_levels):
        small = cv2.pyrDown(small)

    small_sigma = max(1.0, sigma / (2 ** pyramid_levels))
    ksize = int(2 * round(3 * small_sigma) + 1)
    small_blurred = cv2.GaussianBlur(small, (ksize, ksize), small_sigma)

    large = small_blurred
    for _ in range(pyramid_levels):
        large = cv2.pyrUp(large)

    h, w = img_float.shape
    if large.shape != (h, w):
        large = cv2.resize(large, (w, h), interpolation=cv2.INTER_LINEAR)
    return large


def estimate_background_morphological(img_float: np.ndarray, radius: int = 15,
                                       pyramid_levels: int = 1) -> np.ndarray:
    """
    Morphological opening (large structuring element) as a second, cheaper
    background estimate. Performed at a downsampled scale (pyrDown) with a
    proportionally smaller structuring element for speed, then upsampled.
    """
    small = img_float
    for _ in range(pyramid_levels):
        small = cv2.pyrDown(small)

    small_radius = max(3, radius // (2 ** pyramid_levels))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (small_radius, small_radius))
    opened = cv2.morphologyEx(small, cv2.MORPH_OPEN, kernel)

    large = opened
    for _ in range(pyramid_levels):
        large = cv2.pyrUp(large)

    h, w = img_float.shape
    if large.shape != (h, w):
        large = cv2.resize(large, (w, h), interpolation=cv2.INTER_LINEAR)
    return large


def estimate_rib_structures(img_float: np.ndarray, length: int = 41,
                             angles_deg: tuple[float, ...] = (-10, -5, 0, 5, 10),
                             pyramid_levels: int = 1) -> np.ndarray:
    """
    Estimate rib-like structures for targeted rib suppression.

    Ribs are long, nearly-straight (gently curved), mostly horizontal
    structures spanning much of the image width. Coronary vessels are much
    shorter and far more tortuous over the same length scale. A grayscale
    morphological *opening* with a long, thin, near-horizontal linear
    structuring element only preserves structures that contain a straight
    run at least as long as the kernel -- ribs pass through (kept), while
    vessels are erased (removed) by the opening. Subtracting this estimate
    therefore removes ribs specifically while leaving vessels intact.

    Multiple slightly-tilted kernel angles approximate the gentle rib
    curvature; the elementwise max over angles gives the combined rib
    estimate. Performed at half resolution (pyrDown) for speed.
    """
    small = img_float
    for _ in range(pyramid_levels):
        small = cv2.pyrDown(small)

    small_length = max(9, length // (2 ** pyramid_levels))
    rib_estimate = np.zeros_like(small, dtype=np.float32)
    for angle in angles_deg:
        kernel = _rib_kernel(small_length, angle)
        opened = cv2.morphologyEx(small, cv2.MORPH_OPEN, kernel)
        cv2.max(rib_estimate, opened, dst=rib_estimate)

    large = rib_estimate
    for _ in range(pyramid_levels):
        large = cv2.pyrUp(large)

    h, w = img_float.shape
    if large.shape != (h, w):
        large = cv2.resize(large, (w, h), interpolation=cv2.INTER_LINEAR)
    return large


_RIB_KERNEL_CACHE: dict[tuple[int, float], np.ndarray] = {}


def _rib_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Create (and cache) a long thin near-horizontal linear structuring element."""
    key = (length, angle_deg)
    cached = _RIB_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    size = length if length % 2 == 1 else length + 1
    kernel = np.zeros((size, size), dtype=np.uint8)
    kernel[size // 2, :] = 1
    center = (size / 2, size / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(kernel, M, (size, size), flags=cv2.INTER_NEAREST)
    _RIB_KERNEL_CACHE[key] = rotated
    return rotated


def suppress_background(img_float: np.ndarray, gaussian_sigma: float = 25.0,
                         morph_radius: int = 15, blend: float = 0.5,
                         rib_length: int = 41) -> np.ndarray:
    """
    Suppress ribs/spine/lung/background structures via triple background
    estimation + subtraction (homomorphic-style high-pass filtering plus a
    targeted rib-line estimate).

    Returns a float32 image, still roughly in [0,1]-ish range but centered
    around 0.5 after suppression (values are re-normalized by caller/display).
    """
    bg_gauss = estimate_background_gaussian(img_float, sigma=gaussian_sigma)
    bg_morph = estimate_background_morphological(img_float, radius=morph_radius)
    bg_smooth = blend * bg_gauss + (1.0 - blend) * bg_morph

    rib_estimate = estimate_rib_structures(img_float, length=rib_length)
    # Only remove the rib contribution that exceeds the smooth low-frequency
    # background already captured above, avoiding double-subtraction.
    rib_excess = np.clip(rib_estimate - bg_smooth, 0.0, None)

    background = bg_smooth + rib_excess
    suppressed = img_float - background + 0.5  # recenter around mid-gray
    return np.clip(suppressed, 0.0, 1.0)