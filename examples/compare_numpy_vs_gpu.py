"""
Compare numpy vs GPU backend on a real video stimulus.

Downloads a short demo video, extracts motion energy features using both
numpy and a GPU backend (torch_mps or torch_cuda), and saves the results.

Usage::

    python examples/demo_numpy_vs_gpu.py
    python examples/demo_numpy_vs_gpu.py --backend torch_cuda
    python examples/demo_numpy_vs_gpu.py --nimages 100 --output my_features.npz
"""
import argparse
import os
import sys
import time
import warnings

os.environ["TQDM_DISABLE"] = "1"

import numpy as np

import moten.io
import moten.pyramids
from moten.backend import set_backend, get_backend


def detect_gpu_backend():
    """Try to find an available GPU backend."""
    for name in ["torch_mps", "torch_cuda"]:
        try:
            set_backend(name)
            set_backend("numpy")
            return name
        except BaseException:
            pass
    return None


def run_projection(pyramid, stimulus, n_repeats, batch_size, batched=True):
    """Run projection and return (features, best_time, all_times)."""
    if batched:
        project = lambda: pyramid.project_stimulus_batched(
            stimulus, dtype='float32', batch_size=batch_size)
    else:
        project = lambda: pyramid.project_stimulus(stimulus, dtype='float32')

    # Warm-up
    project()

    durations = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        features = project()
        elapsed = time.perf_counter() - start
        durations.append(elapsed)

    return features, min(durations), durations


def main():
    parser = argparse.ArgumentParser(
        description="Compare numpy vs GPU motion energy extraction on video")
    parser.add_argument(
        "--backend", default=None,
        help="GPU backend to compare against numpy "
             "(default: auto-detect torch_mps or torch_cuda)")
    parser.add_argument(
        "--video-url", default="http://anwarnunez.github.io/downloads/avsnr150s24fps_tiny.mp4",
        help="URL of video file")
    parser.add_argument("--nimages", type=int, default=None,
                        help="Number of frames to extract (default: all)")
    parser.add_argument("--size", type=int, nargs=2, default=[72, 128],
                        metavar=("VDIM", "HDIM"),
                        help="Spatial size for downsampling (default: 72 128)")
    parser.add_argument("--output", default="motion_energy_features.npz",
                        help="Output file (default: motion_energy_features.npz)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Number of timed repeats (default: 3)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for batched projection (default: 128)")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", module="moten")

    # Detect GPU backend
    gpu_backend = args.backend
    if gpu_backend is None:
        gpu_backend = detect_gpu_backend()
        if gpu_backend is None:
            print("No GPU backend available (need torch_mps or torch_cuda).")
            sys.exit(1)
        print(f"Auto-detected GPU backend: {gpu_backend}")
    else:
        try:
            set_backend(gpu_backend)
            set_backend("numpy")
        except BaseException as e:
            print(f"Backend '{gpu_backend}' not available: {e}")
            sys.exit(1)

    vdim, hdim = args.size
    stimulus_fps = 24

    # Load video
    nimages = args.nimages if args.nimages is not None else np.inf
    print(f"\nLoading video: {args.video_url}")
    print(f"  Frames: {'all' if args.nimages is None else args.nimages}, "
          f"Size: ({vdim}, {hdim})")
    luminance_images = moten.io.video2luminance(
        args.video_url, size=(vdim, hdim), nimages=nimages)
    print(f"  Loaded: {luminance_images.shape}")

    # Create pyramid
    pyramid = moten.pyramids.MotionEnergyPyramid(
        stimulus_vhsize=(vdim, hdim),
        stimulus_fps=stimulus_fps,
    )
    print(f"  Filters: {pyramid.nfilters}")

    # --- numpy ---
    print(f"\nRunning numpy...")
    set_backend("numpy")
    features_np, time_np, times_np = run_projection(
        pyramid, luminance_images, args.repeats, args.batch_size, batched=False)
    features_np = np.asarray(features_np)
    print(f"  Best: {time_np:.3f}s (of {args.repeats} runs)")

    # --- GPU backend ---
    print(f"\nRunning {gpu_backend} (batched, batch_size={args.batch_size})...")
    gpu_mod = set_backend(gpu_backend)
    stimulus_gpu = gpu_mod.asarray(luminance_images)
    features_gpu, time_gpu, times_gpu = run_projection(
        pyramid, stimulus_gpu, args.repeats, args.batch_size)
    features_gpu_np = gpu_mod.to_numpy(features_gpu)
    print(f"  Best: {time_gpu:.3f}s (of {args.repeats} runs)")

    # Reset
    set_backend("numpy")

    # Summary
    speedup = time_np / time_gpu
    max_diff = np.max(np.abs(features_np.astype(np.float32)
                             - features_gpu_np.astype(np.float32)))

    # Timepoint-by-timepoint correlation
    corrs = np.array([np.corrcoef(features_np[t], features_gpu_np[t])[0, 1]
                      for t in range(features_np.shape[0])])

    print(f"\n{'='*50}")
    print(f"  numpy:       {time_np:.3f}s")
    print(f"  {gpu_backend:12s}: {time_gpu:.3f}s")
    print(f"  Speedup:     {speedup:.2f}x")
    print(f"  Max |diff|:  {max_diff:.2e}")
    print(f"  Correlation: min={np.nanmin(corrs):.6f} "
          f"mean={np.nanmean(corrs):.6f}")
    print(f"  Output:      {features_np.shape}")
    print(f"{'='*50}")

    # Save
    timings = dict(
        numpy_best=time_np,
        numpy_all=times_np,
        gpu_best=time_gpu,
        gpu_all=times_gpu,
        speedup=speedup,
        backend=gpu_backend,
    )
    np.savez(args.output,
             numpy=features_np,
             **{gpu_backend: features_gpu_np},
             timings=timings)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
