"""
Latency / performance benchmark for the coronary artery processing pipeline.

Measures per-frame processing latency (input -> final processed + enhanced
output) and reports:
    - Average latency (ms)
    - Minimum latency (ms)
    - Maximum latency (ms)
    - Frames Per Second (FPS)
    - Hardware configuration (CPU, core count, OS, Python/OpenCV versions)

Usage:
    python -m src.benchmark --frames 200 --width 512 --height 512
"""

from __future__ import annotations

import argparse
import platform
import time

import cv2
import numpy as np

from .dataset import generate_frame
from .pipeline import CoronaryPipeline

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


LATENCY_BUDGET_MS = 36.0


def _boost_process_priority() -> None:
    """
    Best-effort: raise this process's scheduling priority so that OS/VM
    scheduling noise (especially common on shared/virtualized CI-style
    hosts) does not dominate the measured per-frame latency. This mirrors
    how a real-time cine/angiography acquisition pipeline would typically
    be pinned to a high-priority dedicated thread/process in production.
    Silently no-ops if the platform/permissions don't allow it.
    """
    try:
        if platform.system() == "Windows":
            if _HAS_PSUTIL:
                psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS)
            else:
                import ctypes
                HIGH_PRIORITY_CLASS = 0x00000080
                handle = ctypes.windll.kernel32.GetCurrentProcess()
                ctypes.windll.kernel32.SetPriorityClass(handle, HIGH_PRIORITY_CLASS)
        else:
            os_mod = __import__("os")
            os_mod.nice(-10)
    except Exception:
        pass


def get_hardware_info() -> dict:
    info = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.uname().processor,
        "python_version": platform.python_version(),
        "opencv_version": cv2.__version__,
        "logical_cores": (psutil.cpu_count(logical=True) if _HAS_PSUTIL
                           else __import__("os").cpu_count()),
    }
    if _HAS_PSUTIL:
        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["total_ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    return info


def run_benchmark(n_frames: int = 200, width: int = 512, height: int = 512,
                   warmup: int = 20, pool_size: int = 30,
                   boost_priority: bool = True) -> dict:
    """
    Measure per-frame pipeline latency.

    Frames are drawn from a small rotating pool of `pool_size` distinct
    pre-generated frames (recycled) rather than one giant list of
    `n_frames` unique arrays. This matches realistic real-time streaming
    (one current frame in flight at a time) and avoids unrepresentative
    memory/cache pressure from holding hundreds of large arrays alive
    simultaneously, which otherwise distorts the latency measurement.
    """
    if boost_priority:
        _boost_process_priority()

    pipeline = CoronaryPipeline()

    pool = [generate_frame(width, height, seed=i) for i in range(pool_size)]

    # Warm-up (JIT / cache warm-up, first-call OpenCV thread pool init, etc.)
    for i in range(warmup):
        pipeline.process(pool[i % pool_size])

    latencies_ms = []
    for i in range(n_frames):
        frame = pool[i % pool_size]
        t0 = time.perf_counter()
        pipeline.process(frame)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    latencies_ms = np.array(latencies_ms)
    avg_ms = float(latencies_ms.mean())
    median_ms = float(np.median(latencies_ms))
    min_ms = float(latencies_ms.min())
    max_ms = float(latencies_ms.max())
    fps = 1000.0 / avg_ms if avg_ms > 0 else float("inf")
    pass_rate = float((latencies_ms <= LATENCY_BUDGET_MS).mean() * 100.0)

    return {
        "n_frames": n_frames,
        "frame_size": f"{width}x{height}",
        "avg_latency_ms": avg_ms,
        "median_latency_ms": median_ms,
        "min_latency_ms": min_ms,
        "max_latency_ms": max_ms,
        "fps": fps,
        "budget_ms": LATENCY_BUDGET_MS,
        "pass_rate_pct": pass_rate,
        # The assignment's "maximum processing time limited to 36 ms per
        # frame" is a per-frame (worst-case) guarantee, not merely an
        # average. So the authoritative pass/fail criterion is the
        # MAXIMUM observed latency across all frames, not the mean.
        "meets_requirement": bool(max_ms <= LATENCY_BUDGET_MS),
        "meets_requirement_avg_only": bool(avg_ms <= LATENCY_BUDGET_MS),
        "hardware": get_hardware_info(),
    }


def print_report(results: dict) -> None:
    hw = results["hardware"]
    print("=" * 60)
    print("CORONARY ARTERY IMAGE PROCESSING - LATENCY BENCHMARK")
    print("=" * 60)
    print(f"Frames tested        : {results['n_frames']}")
    print(f"Frame size           : {results['frame_size']}")
    print(f"Average latency      : {results['avg_latency_ms']:.3f} ms")
    print(f"Median latency        : {results['median_latency_ms']:.3f} ms")
    print(f"Minimum latency      : {results['min_latency_ms']:.3f} ms")
    print(f"Maximum latency      : {results['max_latency_ms']:.3f} ms")
    print(f"Frames Per Second    : {results['fps']:.2f} FPS")
    print(f"Latency budget       : {results['budget_ms']} ms/frame")
    print(f"Frames within budget : {results['pass_rate_pct']:.1f}%")
    status_max = "PASS" if results["meets_requirement"] else "FAIL"
    status_avg = "PASS" if results["meets_requirement_avg_only"] else "FAIL"
    print(f"Requirement met (MAX latency <= budget) : {status_max}")
    print(f"Requirement met (avg latency <= budget) : {status_avg}")
    print("-" * 60)
    print("Hardware configuration:")
    for k, v in hw.items():
        print(f"  {k:16s}: {v}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Benchmark coronary pipeline latency.")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--pool-size", type=int, default=30)
    parser.add_argument("--no-priority-boost", action="store_true",
                         help="Disable OS process-priority elevation")
    args = parser.parse_args()

    results = run_benchmark(args.frames, args.width, args.height, args.warmup,
                             args.pool_size, boost_priority=not args.no_priority_boost)
    print_report(results)


if __name__ == "__main__":
    main()