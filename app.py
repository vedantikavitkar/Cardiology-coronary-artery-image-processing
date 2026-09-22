"""
Streamlit UI - Cardiology Coronary Artery Image Processing
============================================================

A clean, interactive front-end for the classical image-processing
pipeline in `src/`. Lets a user:

  - Upload a real 16-bit (or 8-bit) grayscale cardiology image, load a
    DICOM file, or generate a synthetic demo frame.
  - View Raw / Processed / Enhanced side-by-side.
  - Tune pipeline parameters live (Gaussian sigma, morph radius, blend,
    CLAHE clip limit) and see the result update instantly.
  - See the per-frame latency for that exact run and confirm it is
    within the mandatory 36 ms budget.
  - Run the full benchmark suite (avg/min/max/FPS + hardware info)
    from a button, without touching the terminal.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import time

import numpy as np
import cv2
import streamlit as st

from src.dataset import generate_frame
from src.pipeline import CoronaryPipeline
from src.benchmark import run_benchmark, LATENCY_BUDGET_MS, get_hardware_info

st.set_page_config(
    page_title="Coronary Artery Image Processing",
    page_icon="🫀",
    layout="wide",
)

# ----------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }
    .metric-pass { color: #22c55e; font-weight: 700; }
    .metric-fail { color: #ef4444; font-weight: 700; }
    div[data-testid="stImage"] img { border-radius: 8px; }
    h1, h2, h3 { letter-spacing: -0.02em; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _load_uploaded_image(uploaded_file) -> np.ndarray:
    """Load an uploaded image as a 16-bit single-channel array."""
    name = uploaded_file.name.lower()
    data = uploaded_file.read()

    if name.endswith((".dcm", ".dicom")):
        import pydicom
        ds = pydicom.dcmread(io.BytesIO(data))
        arr = ds.pixel_array.astype(np.float32)
        arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-6) * 65535.0
        return arr.astype(np.uint16)

    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError("Could not decode image file.")
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.dtype == np.uint8:
        arr = (arr.astype(np.uint16) * 257)  # scale 8-bit -> 16-bit range
    return arr.astype(np.uint16)


@st.cache_resource
def _get_pipeline(gaussian_sigma, morph_radius, blend, clahe_clip) -> CoronaryPipeline:
    return CoronaryPipeline(
        gaussian_sigma=gaussian_sigma,
        morph_radius=morph_radius,
        blend=blend,
        clahe_clip=clahe_clip,
    )


def _to_display(img: np.ndarray) -> np.ndarray:
    """Normalize any dtype to a displayable 8-bit image."""
    if img.dtype == np.uint16:
        img = (img.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
    return img


# ----------------------------------------------------------------------
# Sidebar - controls
# ----------------------------------------------------------------------
st.sidebar.title("🫀 Controls")

st.sidebar.subheader("1. Input image")
source = st.sidebar.radio(
    "Source", ["Synthetic demo frame", "Upload image / DICOM"], index=0
)

if source == "Synthetic demo frame":
    seed = st.sidebar.slider("Random seed", 0, 9999, 7)
    size = st.sidebar.select_slider("Frame size", options=[256, 512, 768], value=512)
    raw = generate_frame(size, size, seed=seed)
else:
    uploaded = st.sidebar.file_uploader(
        "Upload a grayscale image or DICOM file",
        type=["png", "jpg", "jpeg", "tif", "tiff", "dcm", "dicom"],
    )
    raw = None
    if uploaded is not None:
        try:
            raw = _load_uploaded_image(uploaded)
        except Exception as e:
            st.sidebar.error(f"Failed to load image: {e}")

st.sidebar.subheader("2. Pipeline parameters")
gaussian_sigma = st.sidebar.slider("Background blur sigma", 5.0, 60.0, 25.0, 1.0)
morph_radius = st.sidebar.slider("Morphological radius", 5, 35, 15, 1)
blend = st.sidebar.slider("Background subtraction blend", 0.0, 1.0, 0.5, 0.05)
clahe_clip = st.sidebar.slider("CLAHE clip limit", 0.5, 6.0, 2.5, 0.1)

st.sidebar.subheader("3. Performance check")
run_bench = st.sidebar.button("▶ Run latency benchmark (300 frames)", use_container_width=True)

# ----------------------------------------------------------------------
# Main content
# ----------------------------------------------------------------------
st.title("Cardiology Coronary Artery Image Processing")
st.caption(
    "Classical OpenCV/NumPy pipeline — rib/spine/lung suppression + "
    "coronary vessel enhancement — engineered for ≤ 36 ms/frame latency."
)

if raw is None:
    st.info("👈 Choose a synthetic frame or upload an image/DICOM to begin.")
    st.stop()

pipeline = _get_pipeline(gaussian_sigma, morph_radius, blend, clahe_clip)

# Time this exact run
t0 = time.perf_counter()
result = pipeline.process(raw)
t1 = time.perf_counter()
latency_ms = (t1 - t0) * 1000.0

# ---- Metrics row ----
m1, m2, m3, m4 = st.columns(4)
m1.metric("This-frame latency", f"{latency_ms:.2f} ms")
m2.metric("Budget", f"{LATENCY_BUDGET_MS:.0f} ms")
status_ok = latency_ms <= LATENCY_BUDGET_MS
m3.metric("Status", "✅ PASS" if status_ok else "❌ FAIL")
m4.metric("Input dtype", str(raw.dtype))

st.divider()

# ---- Image comparison ----
c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Original / Raw")
    st.image(_to_display(result["raw"]), use_container_width=True, clamp=True)
    st.caption(f"shape={result['raw'].shape}, dtype={result['raw'].dtype}")
with c2:
    st.subheader("Processed")
    st.image(result["processed"], use_container_width=True, clamp=True)
    st.caption("Background/rib/spine/lung suppressed + contrast enhanced (CLAHE)")
with c3:
    st.subheader("Enhanced Coronary Arteries")
    st.image(result["enhanced"], use_container_width=True, clamp=True)
    st.caption("Multi-scale vessel top-hat enhancement")

st.divider()

# ---- Benchmark section ----
if run_bench:
    with st.spinner("Running 300-frame latency benchmark..."):
        bench = run_benchmark(n_frames=300, width=512, height=512)

    st.subheader("📊 Latency Benchmark Report")
    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("Average", f"{bench['avg_latency_ms']:.2f} ms")
    b2.metric("Median", f"{bench['median_latency_ms']:.2f} ms")
    b3.metric("Minimum", f"{bench['min_latency_ms']:.2f} ms")
    b4.metric("Maximum", f"{bench['max_latency_ms']:.2f} ms")
    b5.metric("FPS", f"{bench['fps']:.1f}")

    req_ok = bench["meets_requirement"]
    st.markdown(
        f"**Requirement (max latency ≤ {LATENCY_BUDGET_MS:.0f} ms): "
        f"{'<span class=\"metric-pass\">PASS ✅</span>' if req_ok else '<span class=\"metric-fail\">FAIL ❌</span>'}** "
        f"&nbsp;|&nbsp; Frames within budget: **{bench['pass_rate_pct']:.1f}%**",
        unsafe_allow_html=True,
    )

    with st.expander("Hardware configuration used for this benchmark"):
        st.json(bench["hardware"])
else:
    st.caption("Click **Run latency benchmark** in the sidebar for a full 300-frame report.")

st.divider()
with st.expander("ℹ️ About this pipeline"):
    st.markdown(
        """
        This is a **classical (non-deep-learning) image-processing pipeline**,
        chosen specifically for deterministic, hardware-predictable latency
        well under the mandatory 36 ms/frame budget.

        **Stages:** 16-bit normalization → denoise → pyramid-accelerated
        background estimation (Gaussian + morphological) → shape-based rib
        suppression (long, near-horizontal structuring elements) →
        background subtraction → CLAHE contrast enhancement → multi-angle
        morphological top-hat vessel enhancement.

        See `README.md` and `docs/PIPELINE.md` for full technical details.
        """
    )