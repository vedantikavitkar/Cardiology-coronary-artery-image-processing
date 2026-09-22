"""
Shared image-loading utilities for real (non-synthetic) cardiology images.

Supports:
    - Standard image formats (PNG/JPG/TIFF/BMP), 8-bit or 16-bit
    - DICOM files (.dcm/.dicom) via pydicom

All loaders normalize output to a single-channel uint16 array so the
existing 16-bit pipeline (`src/pipeline.py`) can process real data exactly
the same way it processes the synthetic dataset.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

DICOM_EXTENSIONS = (".dcm", ".dicom")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def load_dicom_as_uint16(path: str) -> np.ndarray:
    """Load a DICOM file's pixel data as a normalized uint16 grayscale array."""
    import pydicom

    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)

    # Multi-frame cine DICOMs: use the first frame for a single-image test.
    if arr.ndim == 3:
        arr = arr[0]

    lo, hi = float(arr.min()), float(arr.max())
    arr = (arr - lo) / max(hi - lo, 1e-6) * 65535.0
    return arr.astype(np.uint16)


def load_image_as_uint16(path: str) -> np.ndarray:
    """Load a standard image file (any bit depth) as a uint16 grayscale array."""
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.dtype == np.uint8:
        # Scale 8-bit [0,255] -> 16-bit-range [0,65535] so preprocessing
        # (which assumes 16-bit dynamic range) behaves consistently.
        arr = arr.astype(np.uint16) * 257
    elif arr.dtype != np.uint16:
        arr = arr.astype(np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        arr = ((arr - lo) / max(hi - lo, 1e-6) * 65535.0).astype(np.uint16)
    return arr.astype(np.uint16)


def load_any(path: str) -> np.ndarray:
    """Load any supported real-image file (image or DICOM) as uint16 grayscale."""
    ext = os.path.splitext(path)[1].lower()
    if ext in DICOM_EXTENSIONS:
        return load_dicom_as_uint16(path)
    return load_image_as_uint16(path)


def find_real_images(folder: str) -> list[str]:
    """Recursively find all supported image/DICOM files under a folder."""
    paths = []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTENSIONS + DICOM_EXTENSIONS:
                paths.append(os.path.join(root, f))
    return sorted(paths)