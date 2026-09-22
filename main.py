"""
Cardiology Coronary Artery Image Processing - CLI entry point.

Commands:
    python main.py generate-dataset [--frames N] [--width W] [--height H]
    python main.py demo [--input PATH] [--out-dir data/output]
    python main.py benchmark [--frames N] [--width W] [--height H]
    python main.py test-real --input-dir DIR [--out-dir data/output/real]

If --input is omitted for `demo`, a synthetic frame is generated on the fly.
`test-real` runs the full pipeline on every real image/DICOM found in
--input-dir, measuring per-image latency and saving comparison figures --
use this to validate accuracy AND the 36 ms real-time requirement on your
own real cardiology data (not just the synthetic dataset).
"""

from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np

from src.dataset import generate_dataset, generate_frame
from src.pipeline import CoronaryPipeline
from src.visualize import save_comparison
from src.benchmark import run_benchmark, print_report, LATENCY_BUDGET_MS, _boost_process_priority
from src.real_data import load_any, find_real_images


def cmd_generate_dataset(args):
    paths = generate_dataset(out_dir=args.out_dir, n_frames=args.frames,
                              width=args.width, height=args.height)
    print(f"Generated {len(paths)} synthetic 16-bit frames -> {args.out_dir}")


def cmd_demo(args):
    os.makedirs(args.out_dir, exist_ok=True)

    if args.input:
        raw = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Could not read input image: {args.input}")
        if raw.dtype != "uint16":
            raw = raw.astype("uint16")
    else:
        raw = generate_frame(args.width, args.height, seed=7)
        raw_path = os.path.join(args.out_dir, "raw_input.png")
        cv2.imwrite(raw_path, raw)
        print(f"No --input provided; generated synthetic frame -> {raw_path}")

    pipeline = CoronaryPipeline()
    result = pipeline.process(raw)

    processed_path = os.path.join(args.out_dir, "processed.png")
    enhanced_path = os.path.join(args.out_dir, "enhanced.png")
    comparison_path = os.path.join(args.out_dir, "comparison.png")

    cv2.imwrite(processed_path, result["processed"])
    cv2.imwrite(enhanced_path, result["enhanced"])
    save_comparison(result["raw"], result["processed"], result["enhanced"], comparison_path)

    print(f"Processed image  -> {processed_path}")
    print(f"Enhanced image   -> {enhanced_path}")
    print(f"Comparison figure-> {comparison_path}")


def cmd_benchmark(args):
    results = run_benchmark(n_frames=args.frames, width=args.width,
                             height=args.height, warmup=args.warmup,
                             pool_size=args.pool_size,
                             boost_priority=not args.no_priority_boost)
    print_report(results)


def cmd_test_real(args):
    """Run the pipeline on real (non-synthetic) images and report per-image latency."""
    paths = find_real_images(args.input_dir)
    if not paths:
        print(f"No supported image/DICOM files found under: {args.input_dir}")
        return

    if not args.no_priority_boost:
        _boost_process_priority()

    os.makedirs(args.out_dir, exist_ok=True)
    pipeline = CoronaryPipeline()

    # Phase 1: load every image up-front and measure PURE pipeline latency
    # with nothing else happening in between calls (no disk writes/plotting
    # interleaved). This mirrors run_benchmark()'s methodology and avoids
    # OS-level disk-flush/heap-fragmentation noise from imwrite/matplotlib
    # contaminating the next frame's timing -- exactly the kind of I/O a
    # real deployment would keep out of the hot 36ms-budget path anyway.
    loaded = []
    for path in paths:
        try:
            loaded.append((path, load_any(path)))
        except Exception as e:
            print(f"SKIPPED {os.path.basename(path)} ({e})")

    if not loaded:
        print("No images could be loaded.")
        return

    # Warm-up (JIT/cache warm-up), matching run_benchmark()'s methodology.
    for _path, raw in loaded[: min(5, len(loaded))]:
        pipeline.process(raw)

    results = []
    latencies_ms = []
    print(f"Found {len(loaded)} real image(s) in {args.input_dir}\n")
    print(f"{'File':40s} {'Size':>12s} {'Latency (ms)':>14s}  Status")
    print("-" * 80)

    for path, raw in loaded:
        t0 = time.perf_counter()
        result = pipeline.process(raw)
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        latencies_ms.append(latency_ms)
        results.append((path, result))

        status = "PASS" if latency_ms <= LATENCY_BUDGET_MS else "FAIL"
        size_str = f"{raw.shape[1]}x{raw.shape[0]}"
        print(f"{os.path.basename(path):40s} {size_str:>12s} {latency_ms:14.3f}  {status}")

    # Phase 2: save outputs (comparison figures / PNGs) -- out-of-band,
    # does not affect the latency numbers reported above.
    for path, result in results:
        stem = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(args.out_dir, f"{stem}_processed.png"), result["processed"])
        cv2.imwrite(os.path.join(args.out_dir, f"{stem}_enhanced.png"), result["enhanced"])
        save_comparison(result["raw"], result["processed"], result["enhanced"],
                         os.path.join(args.out_dir, f"{stem}_comparison.png"))

    if not latencies_ms:
        print("\nNo images were successfully processed.")
        return

    lat = np.array(latencies_ms)
    print("-" * 80)
    print(f"Images processed     : {len(lat)}")
    print(f"Average latency      : {lat.mean():.3f} ms")
    print(f"Minimum latency      : {lat.min():.3f} ms")
    print(f"Maximum latency      : {lat.max():.3f} ms")
    print(f"Latency budget       : {LATENCY_BUDGET_MS} ms/frame")
    pass_rate = float((lat <= LATENCY_BUDGET_MS).mean() * 100.0)
    print(f"Within budget        : {pass_rate:.1f}%")
    print(f"Requirement (MAX)    : {'PASS' if lat.max() <= LATENCY_BUDGET_MS else 'FAIL'}")
    print(f"\nOutputs saved to -> {args.out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Cardiology Coronary Artery Image Processing Pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate-dataset", help="Generate synthetic 16-bit dataset")
    p_gen.add_argument("--out-dir", default="data/synthetic")
    p_gen.add_argument("--frames", type=int, default=20)
    p_gen.add_argument("--width", type=int, default=512)
    p_gen.add_argument("--height", type=int, default=512)
    p_gen.set_defaults(func=cmd_generate_dataset)

    p_demo = sub.add_parser("demo", help="Run pipeline on one frame and save comparison")
    p_demo.add_argument("--input", default=None, help="Path to 16-bit grayscale image")
    p_demo.add_argument("--out-dir", default="data/output")
    p_demo.add_argument("--width", type=int, default=512)
    p_demo.add_argument("--height", type=int, default=512)
    p_demo.set_defaults(func=cmd_demo)

    p_bench = sub.add_parser("benchmark", help="Measure per-frame latency / FPS")
    p_bench.add_argument("--frames", type=int, default=200)
    p_bench.add_argument("--width", type=int, default=512)
    p_bench.add_argument("--height", type=int, default=512)
    p_bench.add_argument("--warmup", type=int, default=20)
    p_bench.add_argument("--pool-size", type=int, default=30)
    p_bench.add_argument("--no-priority-boost", action="store_true")
    p_bench.set_defaults(func=cmd_benchmark)

    p_real = sub.add_parser("test-real", help="Run pipeline on real images/DICOM and measure latency")
    p_real.add_argument("--input-dir", required=True, help="Folder containing real images/DICOM files")
    p_real.add_argument("--out-dir", default="data/output/real")
    p_real.add_argument("--no-priority-boost", action="store_true")
    p_real.set_defaults(func=cmd_test_real)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()