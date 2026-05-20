import bz2
import struct
from collections import namedtuple

import cv2
import numpy as np


EncoderConfig = namedtuple(
    "EncoderConfig",
    "gop quality block macroblock search subsample",
)

DEFAULT_CONFIG = EncoderConfig(
    gop=8,
    quality=50,
    block=8,
    macroblock=16,
    search=8,
    subsample=True,
)

_LUMA_BASE = np.array([
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99],
], dtype=np.float32)

_CHROMA_BASE = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float32)


def make_qtables(quality):
    q = max(1, min(100, int(quality)))
    scale = (5000.0 / q) if q < 50 else (200.0 - 2.0 * q)
    def _apply(base):
        return np.clip(np.floor((base * scale + 50.0) / 100.0), 1, 255).astype(np.int32)
    return _apply(_LUMA_BASE), _apply(_CHROMA_BASE)


def bgr_to_ycbcr(bgr):
    converted = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    return converted[..., [0, 2, 1]]


def ycbcr_to_bgr(ycbcr):
    swapped = ycbcr[..., [0, 2, 1]].astype(np.float32)
    swapped = np.clip(swapped, 0, 255).astype(np.uint8)
    return cv2.cvtColor(swapped, cv2.COLOR_YCrCb2BGR)


def downsample_chroma(plane):
    h, w = plane.shape
    h2, w2 = (h // 2) * 2, (w // 2) * 2
    cropped = plane[:h2, :w2]
    return cropped.reshape(h2 // 2, 2, w2 // 2, 2).mean(axis=(1, 3))


def upsample_chroma(plane, target_hw):
    th, tw = target_hw
    enlarged = np.repeat(np.repeat(plane, 2, axis=0), 2, axis=1)
    return enlarged[:th, :tw]


def _pad_plane(plane, multiple):
    h, w = plane.shape
    ph = (-h) % multiple
    pw = (-w) % multiple
    return np.pad(plane, ((0, ph), (0, pw)), mode="edge")


def encode_plane(plane, qtable, block_size):
    shifted = plane.astype(np.float32) - 128.0
    return _run_dct_quant(shifted, qtable, block_size)


def encode_residual(residual, qtable, block_size):
    return _run_dct_quant(residual.astype(np.float32), qtable, block_size)


def _run_dct_quant(plane, qtable, block_size):
    h, w = plane.shape
    output = np.empty((h // block_size, w // block_size, block_size, block_size), dtype=np.int16)
    for row in range(h // block_size):
        for col in range(w // block_size):
            tile = plane[row * block_size:(row + 1) * block_size,
                         col * block_size:(col + 1) * block_size]
            coeffs = cv2.dct(tile)
            output[row, col] = np.round(coeffs / qtable).astype(np.int16)
    return output


def decode_plane(quant_blocks, qtable, block_size, shift=True):
    rows, cols = quant_blocks.shape[:2]
    h, w = rows * block_size, cols * block_size
    result = np.empty((h, w), dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            dequant = quant_blocks[row, col].astype(np.float32) * qtable
            tile = cv2.idct(dequant)
            result[row * block_size:(row + 1) * block_size,
                   col * block_size:(col + 1) * block_size] = tile
    if shift:
        result += 128.0
    return result


def _block_sad(a, b):
    return int(np.abs(a.astype(np.int32) - b.astype(np.int32)).sum())


def _get_ref_block(padded_ref, row_origin, col_origin, block_size, padding):
    return padded_ref[row_origin + padding:row_origin + padding + block_size,
                      col_origin + padding:col_origin + padding + block_size]


def three_step_search(current, reference, block_size, max_search):
    h, w = current.shape
    num_rows, num_cols = h // block_size, w // block_size
    padding = max_search
    padded_ref = np.pad(reference, padding, mode="edge")
    motion_vectors = np.zeros((num_rows, num_cols, 2), dtype=np.int16)

    for row in range(num_rows):
        for col in range(num_cols):
            r0, c0 = row * block_size, col * block_size
            curr_block = current[r0:r0 + block_size, c0:c0 + block_size]

            best_dr, best_dc = 0, 0
            best_cost = _block_sad(curr_block, _get_ref_block(padded_ref, r0, c0, block_size, padding))
            step = max(1, max_search // 2)

            while step >= 1:
                cr, cc = best_dr, best_dc
                for dr in (cr - step, cr, cr + step):
                    for dc in (cc - step, cc, cc + step):
                        if abs(dr) > max_search or abs(dc) > max_search:
                            continue
                        candidate = _get_ref_block(padded_ref, r0 + dr, c0 + dc, block_size, padding)
                        cost = _block_sad(curr_block, candidate)
                        if cost < best_cost:
                            best_cost, best_dr, best_dc = cost, dr, dc
                step //= 2

            motion_vectors[row, col] = (best_dr, best_dc)
    return motion_vectors


def apply_motion_compensation(reference, motion_vectors, block_size):
    h, w = reference.shape
    num_rows, num_cols, _ = motion_vectors.shape
    padding = int(np.max(np.abs(motion_vectors))) if motion_vectors.size else 0
    padded_ref = np.pad(reference, padding, mode="edge")
    prediction = np.zeros_like(reference)
    for row in range(num_rows):
        for col in range(num_cols):
            dr, dc = motion_vectors[row, col]
            r0, c0 = row * block_size, col * block_size
            prediction[r0:r0 + block_size, c0:c0 + block_size] = \
                padded_ref[r0 + dr + padding:r0 + dr + padding + block_size,
                           c0 + dc + padding:c0 + dc + padding + block_size]
    return prediction


_STREAM_MAGIC = b"MV2\x00"
_DTYPE_TO_TAG = {np.int8: 0, np.int16: 1, np.int32: 2}
_TAG_TO_DTYPE = {v: np.dtype(k) for k, v in _DTYPE_TO_TAG.items()}


def _pack_array(arr):
    tag = _DTYPE_TO_TAG[arr.dtype.type]
    header = bytes([arr.ndim]) + struct.pack(f"<{arr.ndim}I", *arr.shape) + bytes([tag])
    return header + arr.tobytes()


def _unpack_array(buf, offset):
    ndim = buf[offset]; offset += 1
    shape = struct.unpack_from(f"<{ndim}I", buf, offset)
    offset += 4 * ndim
    tag = buf[offset]; offset += 1
    dtype = _TAG_TO_DTYPE[tag]
    count = int(np.prod(shape)) if shape else 0
    arr = np.frombuffer(buf, dtype=dtype, count=count, offset=offset).reshape(shape)
    offset += count * dtype.itemsize
    return arr.copy(), offset


def pack_bitstream(config, luma_shape, chroma_shape, frame_records):
    parts = [_STREAM_MAGIC, b"\x01"]
    parts.append(struct.pack(
        "<H HH HH B B B B B",
        len(frame_records),
        luma_shape[0], luma_shape[1],
        chroma_shape[0], chroma_shape[1],
        config.gop, config.quality, config.macroblock,
        config.search, 1 if config.subsample else 0,
    ))
    for rec in frame_records:
        if rec["type"] == "I":
            parts.append(b"I")
            parts.append(_pack_array(rec["y"]))
            parts.append(_pack_array(rec["cb"]))
            parts.append(_pack_array(rec["cr"]))
        else:
            parts.append(b"P")
            parts.append(_pack_array(rec["mv"]))
            parts.append(_pack_array(rec["y"]))
            parts.append(_pack_array(rec["cb"]))
            parts.append(_pack_array(rec["cr"]))
    raw = b"".join(parts)
    return bz2.compress(raw, compresslevel=9)


def unpack_bitstream(blob):
    raw = bz2.decompress(blob)
    if raw[:4] != _STREAM_MAGIC:
        raise ValueError("Not a MV2 bitstream")
    version = raw[4]
    if version != 1:
        raise ValueError(f"Unsupported bitstream version: {version}")
    offset = 5
    (num_frames, ly, lx, cy, cx, gop, q, mb, search, sub) = struct.unpack_from(
        "<H HH HH B B B B B", raw, offset)
    offset += struct.calcsize("<H HH HH B B B B B")
    config = EncoderConfig(gop=gop, quality=q, block=8,
                           macroblock=mb, search=search, subsample=bool(sub))
    luma_shape = (ly, lx)
    chroma_shape = (cy, cx)

    frame_records = []
    for _ in range(num_frames):
        tag = chr(raw[offset]); offset += 1
        if tag == "I":
            y, offset = _unpack_array(raw, offset)
            cb, offset = _unpack_array(raw, offset)
            cr, offset = _unpack_array(raw, offset)
            frame_records.append({"type": "I", "y": y, "cb": cb, "cr": cr})
        elif tag == "P":
            mv, offset = _unpack_array(raw, offset)
            y, offset = _unpack_array(raw, offset)
            cb, offset = _unpack_array(raw, offset)
            cr, offset = _unpack_array(raw, offset)
            frame_records.append({"type": "P", "mv": mv, "y": y, "cb": cb, "cr": cr})
        else:
            raise ValueError(f"Unknown frame tag: {tag!r}")
    return config, luma_shape, chroma_shape, frame_records


def encode(frames_bgr, config=DEFAULT_CONFIG):
    qt_luma, qt_chroma = make_qtables(config.quality)
    mb = config.macroblock
    block_size = config.block

    luma_shape = None
    chroma_shape = None
    frame_records = []
    prev_planes = None

    for frame_idx, bgr in enumerate(frames_bgr):
        ycbcr = bgr_to_ycbcr(bgr)
        luma = ycbcr[..., 0]
        cb = ycbcr[..., 1]
        cr = ycbcr[..., 2]
        if config.subsample:
            cb = downsample_chroma(cb)
            cr = downsample_chroma(cr)

        luma = _pad_plane(luma, mb)
        cb = _pad_plane(cb, block_size)
        cr = _pad_plane(cr, block_size)

        if luma_shape is None:
            luma_shape = luma.shape
            chroma_shape = cb.shape

        is_intra = (frame_idx % config.gop) == 0 or prev_planes is None
        if is_intra:
            q_luma = encode_plane(luma, qt_luma, block_size)
            q_cb = encode_plane(cb, qt_chroma, block_size)
            q_cr = encode_plane(cr, qt_chroma, block_size)
            frame_records.append({"type": "I", "y": q_luma, "cb": q_cb, "cr": q_cr})
            rec_luma = np.clip(decode_plane(q_luma, qt_luma, block_size), 0, 255)
            rec_cb = np.clip(decode_plane(q_cb, qt_chroma, block_size), 0, 255)
            rec_cr = np.clip(decode_plane(q_cr, qt_chroma, block_size), 0, 255)
        else:
            prev_luma, prev_cb, prev_cr = prev_planes
            mv = three_step_search(luma.astype(np.uint8), prev_luma.astype(np.uint8),
                                   mb, config.search)
            pred_luma = apply_motion_compensation(prev_luma, mv, mb)
            res_luma = luma.astype(np.float32) - pred_luma.astype(np.float32)
            q_luma = encode_residual(res_luma, qt_luma, block_size)
            dec_res_luma = decode_plane(q_luma, qt_luma, block_size, shift=False)
            rec_luma = np.clip(pred_luma + dec_res_luma, 0, 255)

            if config.subsample:
                mv_chroma = (mv // 2).astype(np.int16)
                mb_chroma = mb // 2
            else:
                mv_chroma = mv
                mb_chroma = mb
            pred_cb = apply_motion_compensation(prev_cb, mv_chroma, mb_chroma)
            pred_cr = apply_motion_compensation(prev_cr, mv_chroma, mb_chroma)
            res_cb = cb.astype(np.float32) - pred_cb.astype(np.float32)
            res_cr = cr.astype(np.float32) - pred_cr.astype(np.float32)
            q_cb = encode_residual(res_cb, qt_chroma, block_size)
            q_cr = encode_residual(res_cr, qt_chroma, block_size)
            dec_res_cb = decode_plane(q_cb, qt_chroma, block_size, shift=False)
            dec_res_cr = decode_plane(q_cr, qt_chroma, block_size, shift=False)
            rec_cb = np.clip(pred_cb + dec_res_cb, 0, 255)
            rec_cr = np.clip(pred_cr + dec_res_cr, 0, 255)
            frame_records.append({"type": "P", "mv": mv,
                                  "y": q_luma, "cb": q_cb, "cr": q_cr})

        prev_planes = (rec_luma, rec_cb, rec_cr)

    return pack_bitstream(config, luma_shape, chroma_shape, frame_records)


def decode(blob, output_shape=None):
    config, luma_shape, chroma_shape, frame_records = unpack_bitstream(blob)
    qt_luma, qt_chroma = make_qtables(config.quality)
    mb = config.macroblock
    block_size = config.block

    decoded_frames = []
    prev_planes = None

    for rec in frame_records:
        if rec["type"] == "I":
            luma = np.clip(decode_plane(rec["y"], qt_luma, block_size), 0, 255)
            cb = np.clip(decode_plane(rec["cb"], qt_chroma, block_size), 0, 255)
            cr = np.clip(decode_plane(rec["cr"], qt_chroma, block_size), 0, 255)
        else:
            if prev_planes is None:
                raise RuntimeError("P-frame before any I-frame")
            prev_luma, prev_cb, prev_cr = prev_planes
            mv = rec["mv"]
            pred_luma = apply_motion_compensation(prev_luma, mv, mb)
            res_luma = decode_plane(rec["y"], qt_luma, block_size, shift=False)
            luma = np.clip(pred_luma + res_luma, 0, 255)

            if config.subsample:
                mv_chroma = (mv // 2).astype(np.int16)
                mb_chroma = mb // 2
            else:
                mv_chroma = mv
                mb_chroma = mb
            pred_cb = apply_motion_compensation(prev_cb, mv_chroma, mb_chroma)
            pred_cr = apply_motion_compensation(prev_cr, mv_chroma, mb_chroma)
            res_cb = decode_plane(rec["cb"], qt_chroma, block_size, shift=False)
            res_cr = decode_plane(rec["cr"], qt_chroma, block_size, shift=False)
            cb = np.clip(pred_cb + res_cb, 0, 255)
            cr = np.clip(pred_cr + res_cr, 0, 255)

        prev_planes = (luma, cb, cr)

        if config.subsample:
            cb_full = upsample_chroma(cb, luma.shape)
            cr_full = upsample_chroma(cr, luma.shape)
        else:
            cb_full = cb
            cr_full = cr

        ycbcr = np.stack([luma, cb_full, cr_full], axis=-1)
        bgr = ycbcr_to_bgr(ycbcr)

        if output_shape is not None:
            h, w = output_shape
            bgr = bgr[:h, :w]
        decoded_frames.append(bgr)

    return decoded_frames, config, frame_records


def psnr(original, reconstructed):
    a = original.astype(np.float64)
    b = reconstructed.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0) - 10.0 * np.log10(mse)


def frame_breakdown(frame_records):
    num_i = sum(1 for r in frame_records if r["type"] == "I")
    num_p = sum(1 for r in frame_records if r["type"] == "P")
    return num_i, num_p
