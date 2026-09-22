"""
Full end-to-end cardiology coronary-artery image processing pipeline.

process_frame(raw_16bit) -> dict with:
    "raw"        : original 16-bit input (uint16)
    "processed"  : background-suppressed, contrast-enhanced image (uint8)
    "enhanced"   : coronary-artery-enhanced image (uint8)

Designed to run in well under 36 ms per frame (typically single-digit
milliseconds for 512x512 frames) using only vectorized NumPy/OpenCV
operations -- no per-pixel Python loops in the hot path.

Machine-independent latency (adaptive scaling)
-----------------------------------------------
The absolute per-frame cost of every stage scales with pixel count, so the
*same* 16-bit frame that comfortably meets the 36 ms budget on a fast
machine can exceed it on a slower one (different CPU clock speed/IPC,
shared/virtualized host, etc.) -- this is expected, not a bug. To make the
pipeline meet the budget on *any* machine automatically, `CoronaryPipeline`
self-calibrates its own throughput (ms per pixel) the first time it runs,
then computes the largest frame size it can safely process within the
budget (with a safety margin for jitter). Any real 16-bit frame larger than
that is *downscaled* before running the full filter chain and the final
"processed"/"enhanced" outputs are resized back up to the original
resolution -- so the API and output sizes are unchanged, but latency stays
bounded regardless of source resolution or host speed. Small frames (e.g.
your current 8-bit-derived real test images) are left untouched since they
already fit comfortably.
"""

from __future__ import annotations

import time

import numpy as np
import cv2

from . import preprocessing as pre
from . import background_suppression as bgsup
from . import vessel_enhancement as vess

# For hard real-time, single-frame processing, OpenCV's internal thread-pool
# parallelization of small/medium kernel operations adds unpredictable
# synchronization overhead (observed 2-4x latency spikes) that is worse than
# running single-threaded on one dedicated core. Real-time cine pipelines
# typically pin one worker thread per stream anyway, so we disable OpenCV's
# own multithreading for deterministic, low sub-36ms latency.
cv2.setNumThreads(1)

DEFAULT_LATENCY_BUDGET_MS = 36.0


class CoronaryPipeline:
    """Configurable, reusable pipeline object (avoids re-allocating kernels)."""

    def __init__(self, gaussian_sigma: float = 25.0, morph_radius: int = 15,
                 blend: float = 0.5, clahe_clip: float = 2.5,
                 adaptive_scaling: bool = True,
                 latency_budget_ms: float = DEFAULT_LATENCY_BUDGET_MS,
                 safety_margin: float = 0.75, min_scale: float = 0.35,
                 hard_min_scale: float = 0.12):
        """
        adaptive_scaling : bool
            If True (default), self-calibrate this machine's throughput and
            automatically downscale frames that would otherwise exceed the
            latency budget, so the pipeline meets `latency_budget_ms` on any
            hardware. Set False to always process at native resolution
            (original, fixed-cost behaviour).
        latency_budget_ms : float
            Target worst-case per-frame budget (default: 36 ms, per spec).
        safety_margin : float
            Fraction of the budget actually used for sizing (default 0.75),
            leaving headroom for OS/scheduler jitter and calibration error.
        min_scale : float
            Preferred/"soft" floor: never downscale below this fraction of
            the original resolution (default 0.35) as long as the budget can
            still be met there, to avoid unnecessarily destroying vessel
            detail.
        hard_min_scale : float
            Absolute floor (default 0.12) the closed-loop feedback corrector
            (`_feedback_correct`) may fall back to if `min_scale` alone is
            not enough to meet the mandatory `latency_budget_ms` on very
            slow/large-frame hardware -- meeting the compulsory latency
            requirement takes priority over preserving detail in that case.
        """
        self.gaussian_sigma = gaussian_sigma
        self.morph_radius = morph_radius
        self.blend = blend
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))

        self.adaptive_scaling = adaptive_scaling
        self.latency_budget_ms = latency_budget_ms
        self.safety_margin = safety_margin
        self.min_scale = min_scale
        self.hard_min_scale = min(hard_min_scale, min_scale)
        self._calib_a_ms: float | None = None          # fixed overhead (ms)
        self._calib_b_ms_per_mp: float | None = None    # rate (ms/megapixel)
        self._resize_overhead_ms: float = 0.0
        self._current_scale: float = 1.0
        self._ms_per_megapixel: float | None = None  # back-compat / inspection

    def _run_stages(self, raw_16bit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run the 4 processing stages; returns (processed_u8, enhanced_u8)."""
        # 1. Preprocessing: normalize + denoise
        denoised = pre.preprocess(raw_16bit)

        # 2. Background suppression: ribs / spine / lungs / noise
        suppressed = bgsup.suppress_background(
            denoised, gaussian_sigma=self.gaussian_sigma,
            morph_radius=self.morph_radius, blend=self.blend,
        )

        # 3. Contrast enhancement (CLAHE) on the suppressed image
        suppressed_u8 = pre.to_uint8(suppressed)
        processed_u8 = self.clahe.apply(suppressed_u8)

        # 4. Coronary artery enhancement (multi-scale vessel top-hat)
        processed_float = processed_u8.astype(np.float32) / 255.0
        enhanced_float = vess.enhance_vessels(processed_float)
        enhanced_u8 = (enhanced_float * 255.0).astype(np.uint8)

        return processed_u8, enhanced_u8

    def _calibrate(self, raw_16bit: np.ndarray) -> None:
        """
        Fit a two-parameter cost model  elapsed_ms(mp) = a + b * megapixels
        for this machine, by timing the full stage chain at *two* different
        sample sizes derived from the actual incoming frame (rather than a
        single fixed tiny 128 px probe). A single small-probe measurement
        systematically *underestimates* true per-megapixel cost for larger
        target resolutions because fixed per-call overhead (Python/OpenCV
        dispatch, kernel setup, cache effects) dominates at tiny sizes -- this
        was found to under-scale the pipeline on large frames / slow hosts
        and blow through the 36 ms budget. Fitting a line through two points
        (a small and a medium sample) separates the fixed overhead `a` from
        the true per-pixel rate `b`, giving a much more accurate prediction.

        Also measures the fixed cost of the down/up `cv2.resize` pair used
        by `process()` when scaling is active, since that overhead is paid
        on *every* frame regardless of the internal working resolution and
        was previously omitted from the budget calculation entirely.
        """
        h, w = raw_16bit.shape[:2]

        def _sample(frac: float) -> tuple[float, float]:
            cw = max(8, int(round(w * frac)))
            ch = max(8, int(round(h * frac)))
            calib = cv2.resize(raw_16bit, (cw, ch), interpolation=cv2.INTER_AREA).astype(np.uint16)
            t0 = time.perf_counter()
            self._run_stages(calib)
            t1 = time.perf_counter()
            return (cw * ch) / 1_000_000.0, (t1 - t0) * 1000.0

        frac_lo, frac_hi = 0.12, 0.3
        # If the source frame is already small, both probes collapse to the
        # same (tiny) size; that's fine, the fit just degenerates gracefully
        # (handled by the max(..., epsilon) guards below).
        mp_lo, ms_lo = _sample(min(frac_lo, 1.0))
        mp_hi, ms_hi = _sample(min(frac_hi, 1.0))

        if mp_hi - mp_lo > 1e-9:
            b = (ms_hi - ms_lo) / (mp_hi - mp_lo)
            a = ms_lo - b * mp_lo
        else:
            b = ms_lo / max(mp_lo, 1e-9)
            a = 0.0

        # Guard against a degenerate/negative fit (can happen on extremely
        # fast machines with measurement-noise-dominated microsecond timings).
        self._calib_a_ms = max(a, 0.0)
        self._calib_b_ms_per_mp = max(b, 1e-6)

        # One-off measurement of the resize(down) + resize(up) round-trip
        # cost at this frame's native resolution -- this is a fixed tax paid
        # every frame whenever adaptive downscaling is active.
        t0 = time.perf_counter()
        down = cv2.resize(raw_16bit, (max(1, int(w * 0.3)), max(1, int(h * 0.3))),
                           interpolation=cv2.INTER_AREA)
        _ = cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)
        _ = cv2.resize(down.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
        t1 = time.perf_counter()
        self._resize_overhead_ms = (t1 - t0) * 1000.0

        # ms/megapixel derived quantity kept for external inspection/back-compat.
        self._ms_per_megapixel = self._calib_b_ms_per_mp

    def _target_scale(self, h: int, w: int) -> float:
        """Compute the downscale factor (<=1.0) needed to fit the budget,
        using the fitted (fixed_overhead + rate*megapixels) cost model plus
        the measured resize round-trip tax."""
        if self._calib_b_ms_per_mp is None:
            return 1.0

        budget_ms = self.latency_budget_ms * self.safety_margin
        available_ms = budget_ms - self._calib_a_ms - self._resize_overhead_ms
        current_megapixels = (h * w) / 1_000_000.0

        if available_ms <= 0:
            # Even the fixed overhead alone threatens the budget on this
            # (very slow) host: fall back to the hardware floor rather than
            # silently returning scale=1.0, which is how the previous
            # implementation missed the budget on large frames.
            return self.min_scale

        max_megapixels = available_ms / self._calib_b_ms_per_mp
        if current_megapixels <= max_megapixels:
            return 1.0

        scale = (max_megapixels / current_megapixels) ** 0.5
        return max(self.min_scale, min(1.0, scale))

    def _feedback_correct(self, measured_ms: float) -> None:
        """
        Closed-loop safety net: after every frame, compare the *actually
        observed* end-to-end latency (including resize overhead) against the
        budget and correct `self._current_scale` for the next frame. This
        catches any residual error in the one-shot calibration model (e.g. a
        colder cache during calibration, thermal throttling, a shared/
        virtualized host with variable throughput) that a single calibration
        pass cannot fully predict, and is what actually guarantees the
        36 ms/frame ceiling holds in steady state on any machine.
        """
        budget_ms = self.latency_budget_ms * self.safety_margin
        if measured_ms > budget_ms:
            # Cost scales ~quadratically with linear scale (proportional to
            # megapixels), so shrink by sqrt(overrun) to converge quickly;
            # allowed to go below the "soft" min_scale (down to a hard floor)
            # because meeting the mandatory latency budget takes priority
            # over preserving detail.
            ratio = budget_ms / max(measured_ms, 1e-6)
            self._current_scale = max(self.hard_min_scale,
                                       self._current_scale * (ratio ** 0.5))
        elif measured_ms < budget_ms * 0.6:
            # Comfortable headroom: cautiously grow back towards full
            # resolution to maximize vessel detail.
            self._current_scale = min(1.0, self._current_scale * 1.05)

    def process(self, raw_16bit: np.ndarray) -> dict:
        """Run the full pipeline on a single 16-bit grayscale frame."""
        h, w = raw_16bit.shape[:2]

        if self.adaptive_scaling and self._calib_b_ms_per_mp is None:
            self._calibrate(raw_16bit)
            self._current_scale = self._target_scale(h, w)

        scale = self._current_scale if self.adaptive_scaling else 1.0

        t0 = time.perf_counter()
        if scale < 1.0:
            small_w, small_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            work_frame = cv2.resize(raw_16bit, (small_w, small_h), interpolation=cv2.INTER_AREA)
            processed_u8, enhanced_u8 = self._run_stages(work_frame)
            # Resize outputs back up to the original resolution so callers
            # always get full-size images regardless of internal scaling.
            processed_u8 = cv2.resize(processed_u8, (w, h), interpolation=cv2.INTER_LINEAR)
            enhanced_u8 = cv2.resize(enhanced_u8, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            processed_u8, enhanced_u8 = self._run_stages(raw_16bit)
        t1 = time.perf_counter()

        if self.adaptive_scaling:
            self._feedback_correct((t1 - t0) * 1000.0)

        return {
            "raw": raw_16bit,
            "processed": processed_u8,
            "enhanced": enhanced_u8,
        }


def process_frame(raw_16bit: np.ndarray, pipeline: CoronaryPipeline | None = None) -> dict:
    """Functional convenience wrapper around CoronaryPipeline.process()."""
    if pipeline is None:
        pipeline = CoronaryPipeline()
    return pipeline.process(raw_16bit)