"""
Synthetic Cardiology X-ray / Cine Dataset Generator
=====================================================
Generates legally-usable synthetic 16-bit grayscale cardiology-style
X-ray/cine frames containing simulated anatomical structures:
    - Ribs        (bright/dark curved bands)
    - Spine       (central high-density vertical column)
    - Lung fields (large smooth low-frequency regions)
    - Coronary arteries (thin branching contrast-filled vessels)
    - Quantum/electronic noise (Poisson + Gaussian)
    - Motion blur (simulating cardiac/respiratory motion)

This avoids any copyrighted/real patient data while still producing
images with the statistical/structural properties needed to develop
and benchmark the background-suppression + vessel-enhancement pipeline.

Images are saved as 16-bit PNG (0-65535) to `data/synthetic/`.
"""

from __future__ import annotations

import os
import numpy as np
import cv2


RNG = np.random.default_rng(42)


def _draw_ribs(img: np.ndarray, n_ribs: int = 8) -> np.ndarray:
    """Draw curved rib-like bands (elliptical arcs) across the thorax."""
    h, w = img.shape
    rib_layer = np.zeros_like(img, dtype=np.float32)
    cx = w * 0.5
    for i in range(n_ribs):
        cy = h * (0.08 + i * 0.10)
        axis_x = w * 0.62
        axis_y = h * 0.10
        thickness = 5 + (i % 2)
        color = 2200 + 250 * (i % 3)
        cv2.ellipse(
            rib_layer,
            (int(cx), int(cy)),
            (int(axis_x), int(axis_y)),
            0, 15, 165, color, thickness, lineType=cv2.LINE_AA,
        )
    rib_layer = cv2.GaussianBlur(rib_layer, (11, 11), 4)
    return img + rib_layer


def _draw_spine(img: np.ndarray) -> np.ndarray:
    """Draw a bright, smooth (low-frequency) vertical spine column.

    Deliberately kept smooth/solid (no thin periodic vertebral lines):
    thin high-contrast periodic structures would sit at the same spatial
    scale as coronary vessels and be indistinguishable from them for a
    vessel-enhancement filter. Real vertebral cortical edges are much
    lower-contrast than contrast-filled vessels, so a smooth heavy column
    is the more faithful and more useful stand-in for benchmarking
    background suppression.
    """
    h, w = img.shape
    spine_layer = np.zeros_like(img, dtype=np.float32)
    cx = int(w * 0.5)
    col_width = int(w * 0.09)
    cv2.rectangle(spine_layer, (cx - col_width, 0), (cx + col_width, h), 5500, -1)
    # very subtle, wide low-frequency density undulation (not thin lines)
    for y in range(0, h, int(h / 6)):
        cv2.ellipse(spine_layer, (cx, y), (col_width, int(h / 10)), 0, 0, 360, 400, -1)
    spine_layer = cv2.GaussianBlur(spine_layer, (31, 31), 12)
    return img + spine_layer


def _draw_lung_fields(img: np.ndarray) -> np.ndarray:
    """Draw large smooth low-frequency lung-field regions (low density)."""
    h, w = img.shape
    lung_layer = np.zeros_like(img, dtype=np.float32)
    for sign in (-1, 1):
        cx = int(w * (0.5 + sign * 0.28))
        cy = int(h * 0.5)
        axes = (int(w * 0.22), int(h * 0.38))
        cv2.ellipse(lung_layer, (cx, cy), axes, 0, 0, 360, -2500, -1)
    lung_layer = cv2.GaussianBlur(lung_layer, (61, 61), 25)
    return img + lung_layer


def _bezier_point(p0, p1, p2, t):
    return (1 - t) ** 2 * np.array(p0) + 2 * (1 - t) * t * np.array(p1) + t ** 2 * np.array(p2)


def _draw_vessel_branch(layer, start, end, ctrl, width, intensity, n_pts=100):
    for t in np.linspace(0, 1, n_pts):
        pt = _bezier_point(start, ctrl, end, t)
        w_t = max(1, int(width * (1 - 0.6 * t)))
        cv2.circle(layer, (int(pt[0]), int(pt[1])), w_t, intensity, -1, lineType=cv2.LINE_AA)


def _draw_coronary_arteries(img: np.ndarray, n_branches: int = 6) -> np.ndarray:
    """Draw thin branching contrast-filled vessel structures (bright, high freq)."""
    h, w = img.shape
    vessel_layer = np.zeros_like(img, dtype=np.float32)
    origin = (int(w * 0.52), int(h * 0.35))

    main_end = (int(w * 0.35), int(h * 0.85))
    main_ctrl = (int(w * 0.30), int(h * 0.55))
    _draw_vessel_branch(vessel_layer, origin, main_end, main_ctrl, width=6, intensity=9000)

    for i in range(n_branches):
        t_branch = (i + 1) / (n_branches + 1)
        branch_start = _bezier_point(origin, main_ctrl, main_end, t_branch)
        angle = RNG.uniform(-1.2, 1.2)
        length = RNG.uniform(40, 120)
        end_pt = branch_start + length * np.array([np.cos(angle), np.sin(angle)])
        ctrl_pt = branch_start + (length * 0.5) * np.array([np.cos(angle + 0.3), np.sin(angle + 0.3)])
        _draw_vessel_branch(
            vessel_layer, branch_start, end_pt, ctrl_pt,
            width=max(1, 4 - i // 2), intensity=6000 - i * 300,
        )

    vessel_layer = cv2.GaussianBlur(vessel_layer, (3, 3), 0.8)
    return img + vessel_layer


def _apply_noise_and_motion(img: np.ndarray, motion_blur: bool = True) -> np.ndarray:
    """Add Poisson (quantum) + Gaussian (electronic) noise and optional motion blur."""
    img = np.clip(img, 0, 65535)
    # Poisson noise scaled to intensity (quantum noise simulation)
    scaled = img / 65535.0 * 4000
    poisson_noisy = RNG.poisson(scaled).astype(np.float32) / 4000 * 65535
    gaussian_noise = RNG.normal(0, 250, img.shape).astype(np.float32)
    noisy = poisson_noisy + gaussian_noise

    if motion_blur:
        ksize = RNG.integers(3, 8)
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        kernel[ksize // 2, :] = 1.0 / ksize
        angle = RNG.uniform(0, 180)
        M = cv2.getRotationMatrix2D((ksize / 2, ksize / 2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
        kernel /= kernel.sum() if kernel.sum() != 0 else 1
        noisy = cv2.filter2D(noisy, -1, kernel)

    return np.clip(noisy, 0, 65535)


def generate_frame(width: int = 512, height: int = 512, seed: int | None = None) -> np.ndarray:
    """Generate a single synthetic 16-bit cardiology X-ray/cine frame."""
    global RNG
    if seed is not None:
        RNG = np.random.default_rng(seed)

    base = np.full((height, width), 20000, dtype=np.float32)
    base = _draw_lung_fields(base)
    base = _draw_ribs(base)
    base = _draw_spine(base)
    base = _draw_coronary_arteries(base)
    base = _apply_noise_and_motion(base)
    return base.astype(np.uint16)


def generate_dataset(out_dir: str = "data/synthetic", n_frames: int = 20,
                      width: int = 512, height: int = 512) -> list[str]:
    """Generate and save a synthetic dataset of n_frames 16-bit PNG images."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i in range(n_frames):
        frame = generate_frame(width, height, seed=1000 + i)
        path = os.path.join(out_dir, f"frame_{i:04d}.png")
        cv2.imwrite(path, frame)
        paths.append(path)
    return paths


if __name__ == "__main__":
    saved = generate_dataset()
    print(f"Generated {len(saved)} synthetic 16-bit frames in data/synthetic/")