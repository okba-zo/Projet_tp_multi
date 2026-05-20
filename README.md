# MPEG-4-like Video Codec

A compact video codec built from scratch in Python, implementing the core ideas behind MPEG-4: color space conversion, DCT-based quantization, and motion-compensated P-frames.

---

## Project Structure

```
video_core.py          # Core codec library (encode / decode / metrics)
frame_generator.py     # Synthetic test clip generator
pipeline_visualizer.py # Stage-by-stage pipeline visualization
main.py                # CLI entry point
```

---

## Dependencies

```
pip install numpy opencv-python matplotlib
```

---

## Usage

### 1. Generate test frames
```bash
python frame_generator.py -o sample_frames -n 12 --width 128 --height 96
```

### 2. Encode
```bash
python main.py encode sample_frames -o video.bin --gop 8 --q 50
```

### 3. Decode
```bash
python main.py decode video.bin -o decoded --ref sample_frames
```
`--ref` is optional; when provided it prints per-frame PSNR values.

### 4. Visualize the pipeline
```bash
python main.py viz sample_frames video.bin -o pipeline.png
```
Produces a figure showing originals, YCbCr channels, DCT block internals, motion vectors, and reconstructed frames.

### 5. Sweep quality and GOP
```bash
python main.py sweep sample_frames -o experiments.png --gop 8 --q 50
```
Plots compression ratio across a range of quality levels and GOP sizes.

---

## Codec Pipeline

```
BGR frames
    │
    ▼
YCbCr conversion + 4:2:0 chroma subsampling
    │
    ├─ I-frame: DCT → quantize (luma + chroma tables)
    │
    └─ P-frame: Three-Step Search motion estimation
                → motion compensation → DCT on residual
    │
    ▼
Struct-packed + bz2 compressed bitstream (.bin)
```

---

## Key Parameters

| Parameter | Flag | Default | Description |
|---|---|---|---|
| Quality | `--q` | 50 | 1–100, higher = better quality, larger file |
| GOP size | `--gop` | 8 | Interval between I-frames |
| Search range | `--search` | 8 | Motion estimation search radius in pixels |
| Chroma subsample | `--no-chroma-subsample` | on | Disable 4:2:0 subsampling |
