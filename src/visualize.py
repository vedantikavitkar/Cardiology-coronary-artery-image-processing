"""
Visualization helpers: side-by-side comparison of Raw / Processed / Enhanced.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt

from . import preprocessing as pre


def save_comparison(raw_16bit: np.ndarray, processed_u8: np.ndarray,
                     enhanced_u8: np.ndarray, out_path: str,
                     title: str = "Coronary Artery Image Processing") -> None:
    """Save a 3-panel comparison figure: Raw | Processed | Enhanced."""
    raw_u8 = pre.to_uint8(raw_16bit.astype(np.float32))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        (raw_u8, "Original / Raw Image (16-bit)"),
        (processed_u8, "Processed Image\n(Background/Rib/Spine/Lung Suppressed)"),
        (enhanced_u8, "Enhanced Coronary Artery Image"),
    ]
    for ax, (img, subtitle) in zip(axes, panels):
        ax.imshow(img, cmap="gray")
        ax.set_title(subtitle, fontsize=10)
        ax.axis("off")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)