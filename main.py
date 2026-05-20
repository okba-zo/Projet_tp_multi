import argparse
import glob
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

import video_core as vc


def _load_frames(folder):
    paths = sorted(glob.glob(os.path.join(folder, "*.png"))
                   + glob.glob(os.path.join(folder, "*.jpg")))
    if not paths:
        sys.exit(f"no frames found in {folder}")
    return paths, [cv2.imread(p) for p in paths]


def cmd_encode(args):
    frame_paths, frame_list = _load_frames(args.frames)
    config = vc.EncoderConfig(
        gop=args.gop, quality=args.q, block=8,
        macroblock=16, search=args.search,
        subsample=not args.no_chroma_subsample,
    )
    total_raw = sum(f.nbytes for f in frame_list)
    compressed_blob = vc.encode(frame_list, config)
    with open(args.out, "wb") as fh:
        fh.write(compressed_blob)
    print(f"frames        : {len(frame_list)}")
    print(f"raw bytes     : {total_raw:,}")
    print(f"compressed    : {len(compressed_blob):,} bytes")
    print(f"ratio         : {total_raw / len(compressed_blob):.2f}x")
    print(f"output        : {args.out}")


def cmd_decode(args):
    compressed_blob = open(args.bin_path, "rb").read()
    ref_shape = None
    if args.ref:
        _, ref_frames = _load_frames(args.ref)
        ref_shape = ref_frames[0].shape[:2]
    reconstructed, config, frame_records = vc.decode(compressed_blob, output_shape=ref_shape)
    os.makedirs(args.out, exist_ok=True)
    for i, frame in enumerate(reconstructed):
        cv2.imwrite(os.path.join(args.out, f"rec_{i:04d}.png"), frame)
    num_i, num_p = vc.frame_breakdown(frame_records)
    print(f"decoded {len(reconstructed)} frames -> {args.out}")
    print(f"  I-frames: {num_i}, P-frames: {num_p}")
    if args.ref:
        psnr_values = [vc.psnr(r, c) for r, c in zip(ref_frames, reconstructed)]
        print(f"  mean PSNR: {np.mean(psnr_values):.2f} dB")
        for i, val in enumerate(psnr_values):
            print(f"    frame {i:02d}: {val:.2f} dB")


def cmd_viz(args):
    import pipeline_visualizer as pv
    pv.make_figure(args.frames, args.bin_path, args.out)


def cmd_sweep(args):
    _, frame_list = _load_frames(args.frames)
    total_raw = sum(f.nbytes for f in frame_list)

    quality_levels = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    quality_ratios = []
    print("sweep quality...")
    for q in quality_levels:
        blob = vc.encode(frame_list, vc.EncoderConfig(
            gop=args.gop, quality=q, block=8, macroblock=16, search=8, subsample=True))
        quality_ratios.append(total_raw / len(blob))
        print(f"  Q={q:>3}  ratio={quality_ratios[-1]:.2f}x  ({len(blob):,} bytes)")

    gop_levels = [1, 2, 4, 8, 16]
    gop_ratios = []
    print("sweep GOP...")
    for g in gop_levels:
        blob = vc.encode(frame_list, vc.EncoderConfig(
            gop=g, quality=args.q, block=8, macroblock=16, search=8, subsample=True))
        gop_ratios.append(total_raw / len(blob))
        print(f"  GOP={g:>3}  ratio={gop_ratios[-1]:.2f}x  ({len(blob):,} bytes)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(quality_levels, quality_ratios, "o-", color="#1f77b4")
    axes[0].set_xlabel("quality"); axes[0].set_ylabel("compression ratio (x)")
    axes[0].set_title(f"ratio vs quality (GOP={args.gop})"); axes[0].grid(alpha=0.3)
    axes[1].plot(gop_levels, gop_ratios, "s-", color="#d62728")
    axes[1].set_xlabel("GOP size"); axes[1].set_ylabel("compression ratio (x)")
    axes[1].set_title(f"ratio vs GOP (Q={args.q})"); axes[1].grid(alpha=0.3)
    fig.suptitle("Experimental sweeps")
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"saved {args.out}")


def main():
    root_parser = argparse.ArgumentParser()
    subparsers = root_parser.add_subparsers(dest="cmd", required=True)

    enc = subparsers.add_parser("encode")
    enc.add_argument("frames")
    enc.add_argument("-o", "--out", default="video.bin")
    enc.add_argument("--gop", type=int, default=8)
    enc.add_argument("--q", type=int, default=50)
    enc.add_argument("--search", type=int, default=8)
    enc.add_argument("--no-chroma-subsample", action="store_true")
    enc.set_defaults(func=cmd_encode)

    dec = subparsers.add_parser("decode")
    dec.add_argument("bin_path")
    dec.add_argument("-o", "--out", default="decoded")
    dec.add_argument("--ref", default=None)
    dec.set_defaults(func=cmd_decode)

    viz = subparsers.add_parser("viz")
    viz.add_argument("frames")
    viz.add_argument("bin_path")
    viz.add_argument("-o", "--out", default="pipeline.png")
    viz.set_defaults(func=cmd_viz)

    swp = subparsers.add_parser("sweep")
    swp.add_argument("frames")
    swp.add_argument("-o", "--out", default="experiments.png")
    swp.add_argument("--gop", type=int, default=8)
    swp.add_argument("--q", type=int, default=50)
    swp.set_defaults(func=cmd_sweep)

    args = root_parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
