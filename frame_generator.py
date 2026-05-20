import argparse
import os
import sys

import cv2
import numpy as np


def generate_clip(output_dir, num_frames=12, width=128, height=96, seed=1):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    base_intensity = 120.0 + 50.0 * np.sin(xs / 18.0) + 30.0 * np.cos(ys / 15.0)
    base_intensity = np.clip(base_intensity, 0, 255).astype(np.uint8)
    background = cv2.cvtColor(base_intensity, cv2.COLOR_GRAY2BGR)

    shape_list = []
    for _ in range(5):
        shape_list.append(dict(
            pos=np.array([rng.integers(15, height - 25),
                          rng.integers(15, width - 25)], dtype=np.float32),
            vel=rng.uniform(-3.5, 3.5, 2).astype(np.float32),
            radius=int(rng.integers(7, 18)),
            color=tuple(int(x) for x in rng.integers(60, 240, 3)),
        ))

    for i in range(num_frames):
        frame = background.copy()
        for shape in shape_list:
            shape["pos"] += shape["vel"]
            if shape["pos"][0] < shape["radius"] or shape["pos"][0] > height - shape["radius"]:
                shape["vel"][0] *= -1
            if shape["pos"][1] < shape["radius"] or shape["pos"][1] > width - shape["radius"]:
                shape["vel"][1] *= -1
            cv2.circle(frame,
                       (int(shape["pos"][1]), int(shape["pos"][0])),
                       shape["radius"], shape["color"], -1)
        cv2.imwrite(os.path.join(output_dir, f"f_{i:04d}.png"), frame)
    print(f"wrote {num_frames} frames -> {output_dir}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out", default="sample_frames")
    parser.add_argument("-n", "--num", type=int, default=12)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=96)
    args = parser.parse_args()
    generate_clip(args.out, args.num, args.width, args.height)
