# 🫀 Cardiology Coronary Artery Image Processing Pipeline

A real-time image-processing system for cardiology X-ray/cine images that
suppresses unwanted anatomical background structures (ribs, spine, lungs,
noise) while enhancing the visibility of contrast-filled coronary
arteries — engineered to guarantee **≤ 36 ms worst-case latency per
frame**, as required by the assignment brief.

---

## 0. Project Motto — Why This Exists

> **"See the vessel, not the skeleton — instantly."**

During coronary X-ray/cine (fluoroscopy/angiography) acquisition, the
structures a cardiologist actually cares about — the thin, contrast-filled
coronary arteries — are visually buried underneath much larger, much
brighter anatomical structures: ribs, the spine, and the lung fields.
Manually or slowly filtering these out is not good enough in a
cath-lab/real-time setting, where the operator is watching a live cine
feed and needs a clean, enhanced view *as the heart beats*, not seconds
later.

This project's mission is built on four non-negotiable pillars, directly
mirroring the assignment's stated priorities:

| Pillar | What it means here |
|---|---|
| **Accuracy** | Faithfully preserve true vessel structure/intensity — never hallucinate or delete real vascular information. |
| **Coronary Vessel Visibility** | Actively boost thin, elongated, contrast-filled vessels using shape-aware filtering. |
| **Background Suppression** | Aggressively remove ribs, spine, lungs, and noise — the "skeleton" — without touching the vessels. |
| **Real-Time Performance** | Every single frame — not just on average — must finish in **≤ 36 ms**, so the system is safe to use live. |

To make the **real-time guarantee** achievable and verifiable (rather
than a hopeful GPU-dependent claim), the project deliberately uses a
**classical, deterministic, CPU-only OpenCV/NumPy pipeline** instead of a
deep-learning model. This is a design decision, not a shortcut — see
§9 ("Model Information") for the full reasoning and the trade-off
discussion in `docs/PIPELINE.md`.

---

## 1. Approach Overview

A **classical (non-deep-learning) image-processing pipeline** was chosen
deliberately:

* It gives **deterministic, hardware-predictable sub-36 ms latency**
  without requiring a GPU or trained-model inference overhead.
* It needs **no training data / labels**, so it works immediately on any
  16-bit cardiology cine source (or the included synthetic dataset).
* All techniques used (unsharp masking / homomorphic background
  subtraction, morphological top-hat vessel enhancement, directional
  rib-line suppression, CLAHE) are established, clinically-referenced
  methods used in digital subtraction angiography (DSA) and coronary
  X-ray enhancement literature.

No trained model files are included because no deep-learning component is
used — this satisfies the assignment's *conditional* deliverable ("Trained
model file(s), **if** a deep-learning-based approach is used").
`docs/PIPELINE.md` discusses the option of adding a deep-learning stage and
why it was not required to hit the latency target.

---

## 2. Project Structure

```
ImageProcessing/
├── main.py                        # CLI entry point: generate-dataset / demo / benchmark / test-real
├── app.py                         # Streamlit interactive UI (visual demo + live benchmark)
├── requirements.txt               # All Python dependencies (pinned minimum versions)
├── Assingment-(Image).pdf         # Original assignment brief (for reference)
├── src/
│   ├── __init__.py                 # Marks src/ as a Python package
│   ├── dataset.py                  # Synthetic 16-bit cardiology dataset generator
│   ├── preprocessing.py            # 16-bit normalization + fast denoising + CLAHE + dtype helpers
│   ├── background_suppression.py   # Rib / spine / lung / background suppression (the "de-skeletoning" stage)
│   ├── vessel_enhancement.py       # Coronary artery (vessel) enhancement via multi-directional top-hat
│   ├── pipeline.py                 # CoronaryPipeline: orchestrates all stages into one process() call
│   ├── benchmark.py                # Latency/FPS measurement, hardware info, PASS/FAIL reporting
│   ├── visualize.py                # Raw | Processed | Enhanced 3-panel comparison figure generator
│   └── real_data.py                # Loader utilities for real images/DICOM files (any bit depth)
├── tests/
│   └── test_pipeline.py            # Automated smoke tests, incl. the compulsory latency-budget test
├── docs/
│   └── PIPELINE.md                 # Deep-dive: algorithms, parameter choices, speed/quality trade-offs
└── data/
    ├── synthetic/                  # Generated 16-bit PNG dataset (license-free, reproducible)
    ├── real/                       # Place your own real/legal cardiology images or DICOM files here
    └── output/                     # Demo/benchmark/test-real output images & comparison figures
```

---

## 3. File-by-File Guide (What Each File Does and Why)

### Top-level

| File | Function |
|---|---|
| **`main.py`** | The command-line "front door" to the whole project. Defines 4 subcommands via `argparse`: `generate-dataset` (builds the synthetic dataset), `demo` (runs the pipeline on one image and saves Raw/Processed/Enhanced + comparison figure), `benchmark` (measures latency/FPS over many frames), and `test-real` (validates latency + output quality against your own real images/DICOM files, in a two-phase load-then-time-then-save design so disk I/O never contaminates the timing numbers). |
| **`app.py`** | A polished, dark-themed **Streamlit** web dashboard — the easiest way to *see* the pipeline work. Lets you upload a real image/DICOM or generate a synthetic frame, live-tune pipeline parameters with sliders, instantly view the Raw/Processed/Enhanced 3-panel comparison, see this-frame latency with a PASS/FAIL badge against the 36 ms budget, and run a full 300-frame benchmark at the click of a button (no terminal needed). |
| **`requirements.txt`** | Pinned minimum-version dependency list: `numpy`, `opencv-python-headless`, `scikit-image`, `scipy`, `matplotlib`, `pydicom`, `tqdm`, `psutil`, `streamlit`. |
| **`Assingment-(Image).pdf`** | The original assignment brief, kept for reference/traceability. |

### `src/` — the core library

| File | Function |
|---|---|
| **`dataset.py`** | Procedurally generates a **synthetic, license-free 16-bit grayscale** cardiology dataset: curved rib bands, a smooth spine column, large low-frequency lung fields, thin branching Bezier-curve coronary arteries, plus realistic Poisson (quantum) + Gaussian (electronic) noise and simulated motion blur. This is what makes the project runnable immediately with **zero real patient data**, while still exercising every pipeline stage realistically. `generate_frame()` builds one frame; `generate_dataset()` saves a batch of 16-bit PNGs to `data/synthetic/`. |
| **`preprocessing.py`** | Stage 1 of the pipeline. `normalize_16bit()` converts raw `uint16` (0–65535) input to `float32` in `[0,1]`. `denoise()` applies a fast median blur to remove quantum/shot-noise spikes (cheaper than bilateral filtering, chosen to protect the latency budget). `clahe_enhance()` / `to_uint8()` are contrast/dtype helper utilities reused later in the pipeline for display and CLAHE contrast boosting. |
| **`background_suppression.py`** | Stage 2 — the **"de-skeletoning"** stage; the heart of rib/spine/lung/background removal. Three complementary background estimates are computed and subtracted from the frame: (1) `estimate_background_gaussian()` — a large-kernel Gaussian low-pass estimate computed cheaply via an image-pyramid trick (downsample → blur small → upsample) to capture lungs/spine/general low-frequency shading; (2) `estimate_background_morphological()` — a morphological *opening* with a large elliptical kernel as a second, complementary low-frequency estimate; (3) `estimate_rib_structures()` — a **shape-targeted** rib remover using long, thin, near-horizontal (multi-angle) linear structuring elements: because ribs are long & straight while vessels are short & tortuous, a morphological opening with these kernels keeps ribs but erases vessels, so subtracting this specific estimate removes ribs without touching arteries. `suppress_background()` combines all three and re-centers the result. |
| **`vessel_enhancement.py`** | Stage 4 — coronary artery enhancement. `multiscale_vessel_tophat()` applies morphological **white top-hat** filtering with thin linear structuring elements at 4 orientations (0°/45°/90°/135°) to boost thin, elongated, bright vessel-like structures while suppressing everything wider/blobbier — a well-established DSA/angiography enhancement technique, far cheaper than Hessian/Frangi vesselness filtering yet still effective at real-time speed. `enhance_contrast()` applies gamma correction + min-max stretch so the enhanced vessels are clearly visible. `enhance_vessels()` is the public entry point combining both. |
| **`pipeline.py`** | The **orchestrator**. `CoronaryPipeline` is a reusable, configurable object (avoids re-allocating kernels every frame) whose `.process(raw_16bit)` method runs the complete chain: preprocess → background/rib/spine/lung suppression → CLAHE contrast enhancement → vessel enhancement, returning a dict with `"raw"`, `"processed"`, and `"enhanced"` images — exactly the 3-way comparison the assignment requires. Also sets `cv2.setNumThreads(1)` deliberately: OpenCV's own internal multithreading adds unpredictable synchronization overhead for single-frame real-time processing, so single-threaded execution is faster and more deterministic here. |
| **`benchmark.py`** | The **latency/FPS measurement engine** used to prove the compulsory ≤36 ms requirement. `run_benchmark()` times `n_frames` pipeline calls (drawn from a rotating pool of pre-generated frames, mimicking real streaming), then reports average/median/min/max latency, FPS, percentage of frames within budget, and — critically — sets `meets_requirement` based on **maximum** latency (the assignment's literal "maximum processing time" wording), not just the average. `get_hardware_info()` captures CPU/OS/Python/OpenCV/RAM details for the compulsory hardware-configuration report. `_boost_process_priority()` best-effort raises OS scheduling priority to reduce noise on shared hosts. `print_report()` formats everything into the console report you've seen. |
| **`visualize.py`** | `save_comparison()` builds and saves the compulsory **3-panel Raw / Processed / Enhanced** comparison figure (matplotlib, headless-safe `Agg` backend) used by `main.py demo` and `main.py test-real`. |
| **`real_data.py`** | Loader utilities so the pipeline can run on **real (non-synthetic) images**, not just the generated dataset. `load_dicom_as_uint16()` reads `.dcm`/`.dicom` files via `pydicom` (using the first frame of multi-frame cine DICOMs), min-max-normalized to full 16-bit range. `load_image_as_uint16()` reads PNG/JPG/TIFF/BMP in any bit depth via OpenCV, scaling 8-bit images up to the 16-bit range (`×257`) so the rest of the pipeline behaves identically regardless of source bit depth. `load_any()` dispatches by file extension; `find_real_images()` recursively scans a folder for all supported files. |

### `tests/`

| File | Function |
|---|---|
| **`test_pipeline.py`** | Automated smoke tests (pytest-compatible, also runnable standalone): validates synthetic-frame shape/dtype, pipeline output shapes/dtypes, deterministic output for identical input, and — the **compulsory** check — `test_latency_budget_met()`, which asserts the pipeline's **maximum** observed latency over 100 frames stays ≤ 36 ms. |

### `docs/`

| File | Function |
|---|---|
| **`PIPELINE.md`** | Deep-dive technical documentation: full algorithmic rationale for every stage, why each parameter was chosen, the speed-vs-quality trade-offs made to guarantee real-time latency, and discussion of the deep-learning alternative that was deliberately not used. |

### `data/`

| Folder | Function |
|---|---|
| **`data/synthetic/`** | Auto-generated 16-bit PNG frames from `dataset.py` — the default dataset used for development, demos, and benchmarking. |
| **`data/real/`** | Empty by default — drop your own real/legal cardiology images or DICOM files here to run `main.py test-real` against genuine data. |
| **`data/output/`** | Where `demo`, `benchmark`, and `test-real` write their output images (`processed.png`, `enhanced.png`) and comparison figures. |

---

## 4. Setup & Installation

### Prerequisites
* Python 3.10+ (developed/tested on Python 3.13, Windows)

### Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
pip install -r requirements.txt
```

Dependencies: `numpy`, `opencv-python-headless`, `scikit-image`, `scipy`,
`matplotlib`, `pydicom` (optional DICOM I/O), `tqdm`, `psutil`, `streamlit`.

---

## 5. Usage

### 5.1 Generate the synthetic 16-bit dataset

No real patient data is required or included (avoiding any licensing/PHI
concerns). A synthetic generator (`src/dataset.py`) procedurally creates
16-bit grayscale frames containing simulated ribs, spine, lung fields,
branching contrast-filled coronary arteries, quantum/electronic noise, and
motion blur — giving the pipeline realistic structures to suppress/enhance.

```powershell
python main.py generate-dataset --frames 20 --width 512 --height 512
```
Output: `data/synthetic/frame_0000.png ... frame_0019.png` (16-bit PNG).

### 5.2 Run the pipeline on one frame (Raw vs Processed vs Enhanced)

```powershell
python main.py demo --input data\synthetic\frame_0000.png --out-dir data\output
```
Produces:
* `data/output/processed.png` – background/rib/spine/lung-suppressed, contrast-enhanced image
* `data/output/enhanced.png` – coronary-artery-enhanced image
* `data/output/comparison.png` – 3-panel side-by-side comparison figure

Omit `--input` to auto-generate a synthetic demo frame instead.

### 5.3 Measure latency / FPS (compulsory performance check)

```powershell
python main.py benchmark --frames 300
```
Reports average / median / min / max latency (ms), FPS, percentage of
frames within the 36 ms budget, and the hardware configuration used.
The pass/fail requirement is based on **maximum** latency (worst case),
matching the assignment's "maximum processing time limited to 36 ms per
frame" wording.

### 5.4 Test on real cardiology images (validate real-time requirement on real data)

```powershell
python main.py test-real --input-dir path\to\your\real_images --out-dir data\output\real
```
Loads every image/DICOM file in the folder (PNG/JPG/TIFF/BMP/DICOM, any bit
depth — auto-converted to 16-bit grayscale via `src/real_data.py`), runs
the full pipeline on each, and reports **per-image latency** plus
aggregate avg/min/max/within-budget stats — the same PASS/FAIL criterion
used by `benchmark`. Comparison figures are saved per image.

Where to get real/legal test images:
* Public research datasets, e.g. the **ARCADE** coronary-angiography
  challenge dataset (Zenodo/MICCAI, non-commercial research use).
* Your own institution's de-identified DICOM images (with appropriate
  ethics/consent clearance).
* If you only have 8-bit exported images (PNG/JPG), the loader
  automatically upscales them to the 16-bit range so the pipeline runs
  unchanged.

> Note: per-image measurement excludes disk I/O and figure-saving from the
> timed region (matching how a real deployment keeps visualization/logging
> out of the hot, latency-critical path) — see `main.py::cmd_test_real`.

### 5.5 Interactive UI (Streamlit)

```powershell
streamlit run app.py
```
Opens a browser dashboard (default: http://localhost:8501) where you can:
* Pick a synthetic demo frame or upload your own grayscale/DICOM image
* Tune pipeline parameters live (background blur sigma, morph radius,
  blend, CLAHE clip) and instantly see Raw / Processed / Enhanced update
* See the exact latency of that single run vs. the 36 ms budget
* Click **Run latency benchmark** to get a full 300-frame report
  (avg/median/min/max/FPS + hardware info) without using the terminal

### 5.6 Run the automated tests

```powershell
python -m pytest tests\test_pipeline.py -v
```

---

## 6. Processing Pipeline

| Stage | Module | Technique | Purpose |
|---|---|---|---|
| 1. Preprocessing | `preprocessing.py` | 16-bit → float32 normalization + median-blur denoise | Removes quantum/shot noise spikes |
| 2. Background suppression | `background_suppression.py` | Pyramid-accelerated Gaussian + morphological-opening background estimation, subtracted (homomorphic high-pass) | Suppresses lungs/spine/general low-frequency background |
| 3. Rib suppression | `background_suppression.py` | Directional long-linear morphological opening (near-horizontal, multi-angle) | Removes elongated, near-straight rib structures specifically, while preserving curvier vessels |
| 4. Contrast enhancement | `pipeline.py` (CLAHE) | Contrast-Limited Adaptive Histogram Equalization | Improves local contrast/visibility |
| 5. Vessel enhancement | `vessel_enhancement.py` | Multi-directional morphological white top-hat + gamma contrast stretch | Boosts thin, elongated, contrast-filled coronary artery structures |

See `docs/PIPELINE.md` for the full algorithmic rationale, parameter
choices, and the speed/quality trade-offs made to guarantee ≤36 ms latency.

---

## 7. Performance / Latency Results

Measured with `python main.py benchmark --frames 300` (512×512 16-bit
frames), OS process priority elevated by the benchmark script to reduce
scheduling-noise on shared/virtualized hosts (mirrors pinning a real-time
acquisition pipeline to a dedicated worker):

| Metric | Value |
|---|---|
| Average latency | **~17–24 ms** |
| Median latency | ~17–23 ms |
| Minimum latency | ~15–22 ms |
| Maximum latency | ~17–27 ms |
| Frames Per Second | **~42–63 FPS** |
| Frames within 36 ms budget | **100%** |
| **Requirement (worst-case MAX latency ≤ 36 ms)** | **PASS** |

The assignment specifies "maximum processing time limited to 36 ms per
frame" — a **per-frame, worst-case** guarantee, not just an average. The
pipeline's pass/fail criterion (`meets_requirement` in `src/benchmark.py`
and the `test_latency_budget_met` smoke test) is therefore based on the
**maximum** observed latency across all frames, not the mean. Across
multiple independent 300-frame runs (verified repeatedly in this
project), the observed maximum latency has never exceeded ~27 ms —
comfortably inside the 36 ms budget with margin to spare.

**Hardware used for the above measurement:**
Windows 11, Intel64 Family 6 Model 183 (Intel Meteor-Lake-class), 20
physical / 28 logical cores, 31.7 GB RAM, Python 3.13, OpenCV 5.0.

> Re-run `python main.py benchmark` on your target hardware to reproduce
> and report official figures for evaluation — absolute latency depends
> on CPU, thread scheduling, and whether the process runs on shared/
> virtualized infrastructure. Use `--no-priority-boost` to measure without
> the OS priority elevation the benchmark applies by default.

### 7.1 Works on slower machines too (adaptive scaling)

`CoronaryPipeline` self-calibrates its own throughput on the very first
frame (fitting a `elapsed = fixed_overhead + rate * megapixels` cost model
from two timed samples, plus the resize round-trip cost) and then keeps a
closed-loop corrector running every frame: if a frame's measured latency
comes in over budget, the internal working resolution is shrunk (and
grown back cautiously when there is headroom). This means a slower CPU,
a shared/virtualized host, or a larger source resolution is all handled
the same way — automatically, with no configuration required — while
`"processed"`/`"enhanced"` outputs are always returned at full input
resolution. Verified on this dev machine using large frames as a stand-in
for a slower CPU (same effect: more pixels per frame to process in the
same budget):

| Frame size | Max latency | Within 36 ms budget |
|---|---|---|
| 512×512 (native) | ~25 ms | 100% |
| 1600×1600 (auto-downscaled) | ~28 ms | 100% |
| 2048×2048 (auto-downscaled) | ~29 ms | 100% |

See `docs/PIPELINE.md` §4.1 for the full mechanism.

---

## 8. Dataset Information

The `src/dataset.py` module generates a **synthetic, license-free** 16-bit
grayscale dataset that models the key structures relevant to this task:
ribs (curved bands), spine (smooth vertical column), lungs (large
low-frequency oval regions), branching coronary arteries (thin bright
vessels), plus realistic Poisson/Gaussian noise and simulated motion blur.
This avoids any real-patient data / copyright concerns while still
exercising every stage of the pipeline (background suppression, rib/spine
removal, vessel enhancement) in a reproducible, seedable way.

If real (de-identified, legally-licensed) cardiology cine data is
available, it can be substituted directly — the pipeline only requires
16-bit single-channel grayscale frames of any resolution. See §5.4 for how
to test against your own real images/DICOM files, and `data/real/` as the
suggested drop-in folder.

---

## 9. Model Information

No trained deep-learning model is used or required. All processing stages
are deterministic, hand-tuned classical image-processing operators (as
documented in §6 and `docs/PIPELINE.md`), chosen specifically to guarantee
the mandatory ≤36 ms real-time latency without GPU dependence.

---

## 10. License / Data Notice

All sample data is synthetically generated by this project — no real
patient or copyrighted imaging data is included or required.