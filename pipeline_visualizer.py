import argparse
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

import video_core as vc


def _load_frames(folder):
    paths = sorted(glob.glob(os.path.join(folder, "*.png"))
                   + glob.glob(os.path.join(folder, "*.jpg")))
    return [cv2.imread(p) for p in paths]


def make_figure(frames_dir, bin_path, output_png):
    source_frames = _load_frames(frames_dir)
    blob = open(bin_path, "rb").read()
    reconstructed, config, frame_records = vc.decode(blob, output_shape=source_frames[0].shape[:2])
    qt_luma, _ = vc.make_qtables(config.quality)

    fig, axes = plt.subplots(5, 6, figsize=(16, 12), constrained_layout=True)
    fig.suptitle("MPEG-4-like pipeline — stage visualisation", fontsize=13)

    selected_indices = np.linspace(0, len(source_frames) - 1, 6, dtype=int)
    for col, idx in enumerate(selected_indices):
        axes[0, col].imshow(cv2.cvtColor(source_frames[idx], cv2.COLOR_BGR2RGB))
        axes[0, col].set_title(f"orig {idx}", fontsize=8)
        axes[0, col].axis("off")
    fig.text(0.005, 0.93, "1) originals", weight="bold", rotation=90)

    ycbcr_frame = vc.bgr_to_ycbcr(source_frames[0])
    channel_titles = ["Y", "Cb", "Cr"]
    channel_cmaps = ["gray", "coolwarm", "coolwarm"]
    for i in range(3):
        ax = axes[1, i * 2]
        ax.imshow(ycbcr_frame[..., i], cmap=channel_cmaps[i])
        ax.set_title(channel_titles[i], fontsize=9)
        ax.axis("off")
        axes[1, i * 2 + 1].axis("off")
    fig.text(0.005, 0.73, "2) Y / Cb / Cr", weight="bold", rotation=90)

    luma_plane = ycbcr_frame[..., 0]
    center_row, center_col = luma_plane.shape[0] // 2, luma_plane.shape[1] // 2
    raw_block = luma_plane[center_row:center_row + 8, center_col:center_col + 8].astype(np.float32) - 128.0
    dct_coeffs = cv2.dct(raw_block)
    quantized_block = np.round(dct_coeffs / qt_luma).astype(np.int16)
    reconstructed_block = cv2.idct(quantized_block.astype(np.float32) * qt_luma) + 128.0
    dct_panels = [
        ("raw pixels", raw_block + 128.0, "gray"),
        ("DCT", np.log1p(np.abs(dct_coeffs)), "viridis"),
        ("quantised", quantized_block, "viridis"),
        ("reconstructed", reconstructed_block, "gray"),
        (f"Q-table (Q={config.quality})", qt_luma, "magma"),
    ]
    for i, (title, data, cmap) in enumerate(dct_panels):
        axes[2, i].imshow(data, cmap=cmap)
        axes[2, i].set_title(title, fontsize=9)
        axes[2, i].axis("off")
    axes[2, 5].axis("off")
    fig.text(0.005, 0.53, "3) DCT & quant", weight="bold", rotation=90)

    p_frame_idx = next((i for i, r in enumerate(frame_records) if r["type"] == "P"), None)
    if p_frame_idx is not None:
        motion_vecs = frame_records[p_frame_idx]["mv"]
        grid_spec = axes[3, 0].get_gridspec()
        for c in range(3):
            axes[3, c].axis("off")
        ax_motion = fig.add_subplot(grid_spec[3, 0:3])
        ax_motion.imshow(cv2.cvtColor(source_frames[p_frame_idx], cv2.COLOR_BGR2RGB))
        mv_rows, mv_cols, _ = motion_vecs.shape
        grid_ys = (np.arange(mv_rows) + 0.5) * config.macroblock
        grid_xs = (np.arange(mv_cols) + 0.5) * config.macroblock
        XS, YS = np.meshgrid(grid_xs, grid_ys)
        ax_motion.quiver(XS, YS, motion_vecs[..., 1], motion_vecs[..., 0],
                         color="yellow", angles="xy",
                         scale_units="xy", scale=1, width=0.003)
        ax_motion.set_title(f"motion vectors on P-frame {p_frame_idx}", fontsize=9)
        ax_motion.axis("off")

        for c in range(3, 6):
            axes[3, c].axis("off")
        ax_residual = fig.add_subplot(grid_spec[3, 3:6])
        prev_gray = cv2.cvtColor(reconstructed[p_frame_idx - 1], cv2.COLOR_BGR2GRAY).astype(np.int16)
        curr_gray = cv2.cvtColor(reconstructed[p_frame_idx], cv2.COLOR_BGR2GRAY).astype(np.int16)
        residual_img = ax_residual.imshow(curr_gray - prev_gray, cmap="seismic", vmin=-40, vmax=40)
        ax_residual.set_title(f"residual (P-frame {p_frame_idx} vs prev)", fontsize=9)
        ax_residual.axis("off")
        fig.colorbar(residual_img, ax=ax_residual, fraction=0.04)
    fig.text(0.005, 0.33, "4 & 5) motion + residual", weight="bold", rotation=90)

    recon_indices = np.linspace(0, len(reconstructed) - 1, 6, dtype=int)
    for col, idx in enumerate(recon_indices):
        axes[4, col].imshow(cv2.cvtColor(reconstructed[idx], cv2.COLOR_BGR2RGB))
        axes[4, col].set_title(f"rec {idx} [{frame_records[idx]['type']}]", fontsize=8)
        axes[4, col].axis("off")
    fig.text(0.005, 0.13, "5) reconstructions", weight="bold", rotation=90)

    fig.savefig(output_png, dpi=130, bbox_inches="tight")
    print(f"saved {output_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("frames")
    parser.add_argument("bin_path")
    parser.add_argument("-o", "--out", default="pipeline.png")
    args = parser.parse_args()
    make_figure(args.frames, args.bin_path, args.out)
