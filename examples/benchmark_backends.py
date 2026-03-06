"""
================================
Benchmark backends
================================

Compare motion energy computation speed across available backends
(numpy, torch, torch_cuda, torch_mps).

Usage::

    python examples/benchmark_backends.py
    python examples/benchmark_backends.py --nimages 200 --vdim 128 --hdim 256
    python examples/benchmark_backends.py --backends numpy torch_cuda
"""
import argparse
import sys
import time

import numpy as np

from moten.backend import set_backend, get_backend, ALL_BACKENDS
from moten import pyramids


def detect_available_backends():
    """Return list of backends that can be loaded."""
    available = []
    for name in ALL_BACKENDS:
        try:
            set_backend(name)
            available.append(name)
        except BaseException:
            pass
    set_backend("numpy")
    return available


def run_benchmark(backend_name, stimulus_np, vdim, hdim, stimulus_fps,
                  n_repeats=3, use_batched=False, batch_size=128):
    """Benchmark a single backend. Returns dict with timing results."""
    backend_mod = set_backend(backend_name)

    stimulus = backend_mod.asarray(stimulus_np)
    pyramid = pyramids.MotionEnergyPyramid(
        stimulus_vhsize=(vdim, hdim),
        stimulus_fps=stimulus_fps,
    )

    if use_batched:
        project_fn = lambda s: pyramid.project_stimulus_batched(
            s, dtype='float32', batch_size=batch_size)
    else:
        project_fn = lambda s: pyramid.project_stimulus(s, dtype='float32')

    # Warm-up (important for GPU backends to trigger JIT/kernel compilation)
    project_fn(stimulus)

    # Timed runs
    durations = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        project_fn(stimulus)
        elapsed = time.perf_counter() - start
        durations.append(elapsed)

    return {
        "backend": backend_name,
        "nfilters": pyramid.nfilters,
        "nimages": stimulus_np.shape[0],
        "vhsize": (vdim, hdim),
        "best_seconds": min(durations),
        "mean_seconds": sum(durations) / len(durations),
        "all_seconds": durations,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark pymoten across backends")
    parser.add_argument("--nimages", type=int, default=100,
                        help="Number of video frames (default: 100)")
    parser.add_argument("--vdim", type=int, default=96,
                        help="Vertical pixels (default: 96)")
    parser.add_argument("--hdim", type=int, default=128,
                        help="Horizontal pixels (default: 128)")
    parser.add_argument("--fps", type=int, default=24,
                        help="Stimulus frame rate (default: 24)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Number of timed repeats (default: 3)")
    parser.add_argument("--backends", nargs="*", default=None,
                        help="Backends to test (default: all available)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for batched projection (default: 128)")
    args = parser.parse_args()

    # Detect backends
    available = detect_available_backends()
    if args.backends:
        backends = []
        for b in args.backends:
            if b not in ALL_BACKENDS:
                print(f"Unknown backend: {b}")
                sys.exit(1)
            if b not in available:
                print(f"Backend '{b}' not available in this environment, "
                      f"skipping.")
            else:
                backends.append(b)
    else:
        backends = available

    if not backends:
        print("No backends available.")
        sys.exit(1)

    # Generate stimulus
    rng = np.random.RandomState(args.seed)
    stimulus_np = rng.randn(args.nimages, args.vdim, args.hdim).astype(
        np.float64)

    # Header
    print("=" * 65)
    print("pymoten backend benchmark")
    print("=" * 65)
    print(f"  Stimulus   : {args.nimages} frames x ({args.vdim}, {args.hdim}) "
          f"@ {args.fps} fps")
    print(f"  Repeats    : {args.repeats}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Backends   : {', '.join(backends)}")
    print("-" * 65)

    # Run benchmarks (unbatched and batched for each backend)
    results = []
    for backend_name in backends:
        for use_batched in [False, True]:
            label = f"{backend_name}" + (" (batched)" if use_batched else "")
            print(f"  {label:25s} ... ", end="", flush=True)
            try:
                res = run_benchmark(
                    backend_name, stimulus_np,
                    args.vdim, args.hdim, args.fps,
                    n_repeats=args.repeats,
                    use_batched=use_batched,
                    batch_size=args.batch_size,
                )
                res["label"] = label
                results.append(res)
                print(f"{res['best_seconds']:8.3f}s (best of {args.repeats}), "
                      f"{res['nfilters']} filters")
            except Exception as exc:
                print(f"FAILED: {exc}")

    # Reset backend
    set_backend("numpy")

    # Summary table
    if len(results) > 1:
        print("-" * 65)
        baseline = results[0]["best_seconds"]
        print(f"\n{'Backend':>25s}  {'Best (s)':>10s}  {'Mean (s)':>10s}  "
              f"{'Speedup':>8s}")
        print(f"{'':>25s}  {'':>10s}  {'':>10s}  "
              f"{'(vs %s)' % results[0]['label']:>8s}")
        print("-" * 60)
        for res in results:
            speedup = baseline / res["best_seconds"]
            print(f"{res['label']:>25s}  "
                  f"{res['best_seconds']:10.3f}  "
                  f"{res['mean_seconds']:10.3f}  "
                  f"{speedup:7.2f}x")
    print()


if __name__ == "__main__":
    main()
