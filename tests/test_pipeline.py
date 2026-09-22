"""
Basic smoke tests for the coronary artery processing pipeline.

Run with:
    python -m pytest tests/ -v
or simply:
    python tests/test_pipeline.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import generate_frame
from src.pipeline import CoronaryPipeline
from src.benchmark import run_benchmark, LATENCY_BUDGET_MS


def test_generate_frame_shape_and_dtype():
    frame = generate_frame(256, 256, seed=1)
    assert frame.shape == (256, 256)
    assert frame.dtype == np.uint16


def test_pipeline_output_shapes_and_dtypes():
    frame = generate_frame(256, 256, seed=2)
    pipeline = CoronaryPipeline()
    result = pipeline.process(frame)

    assert result["raw"].shape == (256, 256)
    assert result["processed"].shape == (256, 256)
    assert result["enhanced"].shape == (256, 256)
    assert result["processed"].dtype == np.uint8
    assert result["enhanced"].dtype == np.uint8


def test_pipeline_deterministic_for_same_input():
    frame = generate_frame(256, 256, seed=3)
    pipeline = CoronaryPipeline()
    r1 = pipeline.process(frame)
    r2 = pipeline.process(frame)
    assert np.array_equal(r1["processed"], r2["processed"])
    assert np.array_equal(r1["enhanced"], r2["enhanced"])


def test_latency_budget_met():
    results = run_benchmark(n_frames=100, width=512, height=512, warmup=20)
    # The assignment specifies a MAXIMUM per-frame latency of 36 ms, i.e. a
    # worst-case guarantee -- not just an average. Assert on max latency.
    assert results["max_latency_ms"] <= LATENCY_BUDGET_MS, (
        f"Max latency {results['max_latency_ms']:.2f} ms exceeds "
        f"the {LATENCY_BUDGET_MS} ms real-time budget"
    )


if __name__ == "__main__":
    test_generate_frame_shape_and_dtype()
    print("test_generate_frame_shape_and_dtype: PASSED")
    test_pipeline_output_shapes_and_dtypes()
    print("test_pipeline_output_shapes_and_dtypes: PASSED")
    test_pipeline_deterministic_for_same_input()
    print("test_pipeline_deterministic_for_same_input: PASSED")
    test_latency_budget_met()
    print("test_latency_budget_met: PASSED")
