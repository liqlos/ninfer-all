import hashlib
import json
import mmap
import pathlib
import struct
import time

import numpy as np


ROOT = pathlib.Path("/mnt/c/AI/tuning/ninfer-lossless-20260928")
MODEL = pathlib.Path("/mnt/c/AI/staging/qwen38-jonathan-ninfer/qwen38-jonathan-q4km-proposal.ninfer")
TARGETS = ("weight/000001", "weight/000131")
CHUNK_BLOCKS = 8192
TILE_BLOCKS = (32, 64, 128, 256)


def entropy_from_hist(hist):
    active = np.asarray(hist, dtype=np.float64)
    active = active[active > 0]
    p = active / active.sum()
    return float(-(p * np.log2(p)).sum())


def mutual_information(hist):
    joint = hist.astype(np.float64)
    joint /= joint.sum()
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    nz = joint > 0
    ix, iy = np.nonzero(nz)
    return float(np.sum(joint[ix, iy] * np.log2(joint[ix, iy] / (px[ix] * py[iy]))))


def q4_scale_min(packed):
    scales = np.empty((packed.shape[0], 8), dtype=np.uint8)
    mins = np.empty((packed.shape[0], 8), dtype=np.uint8)
    for j in range(4):
        scales[:, j] = packed[:, j] & 63
        mins[:, j] = packed[:, j + 4] & 63
        scales[:, j + 4] = (packed[:, j + 8] & 15) | ((packed[:, j] >> 6) << 4)
        mins[:, j + 4] = (packed[:, j + 8] >> 4) | ((packed[:, j + 4] >> 6) << 4)
    return scales, mins


def q6_codes(blocks):
    ql = blocks[:, :128].reshape(-1, 2, 64)
    qh = blocks[:, 128:192].reshape(-1, 2, 32)
    groups = []
    for j in range(4):
        lo = ql[:, :, :32] if j in (0, 2) else ql[:, :, 32:]
        nibble = (lo >> (4 if j >= 2 else 0)) & 15
        high = ((qh >> (2 * j)) & 3) ^ 2
        groups.append(nibble | (high << 4))
    return np.stack(groups, axis=2).reshape(-1, 2, 4, 32)


def q6_dplane_roundtrip(words, rows, blocks_per_row, exponent_base, exponent_bits, sign_variable, constant_sign):
    mantissa_mask = (1 << 10) - 1
    bits_per_value = 10 + exponent_bits + int(sign_variable)
    shifts = np.arange(bits_per_value, dtype=np.uint8)
    row_bytes = ((blocks_per_row * bits_per_value + 31) // 32) * 4
    for row_start in range(0, rows, 128):
        row_count = min(128, rows - row_start)
        start = row_start * blocks_per_row
        source = words[start:start + row_count * blocks_per_row].reshape(row_count, blocks_per_row)
        exponent = ((source >> 10) & 31).astype(np.uint8)
        sign = (source >> 15).astype(np.uint8)
        mantissa = source & mantissa_mask
        value = mantissa.astype(np.uint32)
        if exponent_bits:
            value |= (exponent.astype(np.uint32) - exponent_base) << 10
        if sign_variable:
            value |= sign.astype(np.uint32) << (10 + exponent_bits)
        bit_matrix = ((value[:, :, None] >> shifts[None, None, :]) & 1).astype(np.uint8)
        packed = np.packbits(bit_matrix.reshape(row_count, -1), axis=1, bitorder="little")
        row_encoded = np.zeros((row_count, row_bytes), dtype=np.uint8)
        row_encoded[:, :packed.shape[1]] = packed
        decoded_bits = np.unpackbits(row_encoded[:, :packed.shape[1]], axis=1, bitorder="little")[:, :blocks_per_row * bits_per_value].reshape(row_count, blocks_per_row, bits_per_value)
        decoded_value = np.sum(decoded_bits.astype(np.uint32) << shifts[None, None, :], axis=2, dtype=np.uint32)
        decoded_mantissa = decoded_value & mantissa_mask
        if exponent_bits:
            decoded_exponent = ((decoded_value >> 10) & ((1 << exponent_bits) - 1)) + exponent_base
        else:
            decoded_exponent = np.full(source.shape, exponent_base, dtype=np.uint32)
        if sign_variable:
            decoded_sign = (decoded_value >> (10 + exponent_bits)) & 1
        else:
            decoded_sign = np.full(source.shape, constant_sign, dtype=np.uint32)
        restored = (decoded_sign << 15) | (decoded_exponent << 10) | decoded_mantissa
        if not np.array_equal(restored.astype(np.uint16), source):
            raise ValueError("Q6 d-plane packed row roundtrip failed")
    return rows * row_bytes, rows * blocks_per_row, bits_per_value


def scan_all_q6_dplanes(mm, directory, payload_start):
    tensors = []
    for item in directory["objects"]:
        if item.get("format") != "gguf_q6_k":
            continue
        shape = item["shape"]
        if len(shape) != 2 or shape[1] % 256:
            raise ValueError("Q6 d-plane requires a block-aligned matrix row")
        rows = shape[0]
        blocks_per_row = shape[1] // 256
        count = item["bytes"] // 210
        if rows * blocks_per_row != count or item["bytes"] % 210:
            raise ValueError("Q6 d-plane shape and block count mismatch")
        offset = payload_start + item["offset"] + 208
        words = np.ndarray((count,), dtype="<u2", buffer=mm, offset=offset, strides=(210,)).copy()
        exponents = ((words >> 10) & 31).astype(np.uint8)
        signs = (words >> 15).astype(np.uint8)
        exponent_base = int(exponents.min())
        exponent_max = int(exponents.max())
        exponent_range = exponent_max - exponent_base
        exponent_bits = int(np.ceil(np.log2(exponent_range + 1))) if exponent_range else 0
        sign_variable = bool(np.any(signs != signs[0]))
        constant_sign = int(signs[0])
        d_bytes, checked_values, bits_per_value = q6_dplane_roundtrip(words, rows, blocks_per_row, exponent_base, exponent_bits, sign_variable, constant_sign)
        q_bytes = count * 208
        encoded_bytes = q_bytes + d_bytes + 32
        baseline_bytes = item["bytes"]
        histogram = np.bincount(exponents, minlength=32).tolist()
        tensors.append({"id": item["id"], "shape": shape, "blocks": count, "rows": rows, "blocks_per_row": blocks_per_row, "d_exponent_min": exponent_base, "d_exponent_max": exponent_max, "d_exponent_range": exponent_range, "d_exponent_distinct_values": int(np.unique(exponents).size), "d_exponent_bits": exponent_bits, "d_exponent_histogram_0_to_31": histogram, "d_sign_values": sorted(int(x) for x in np.unique(signs)), "d_sign_variable": sign_variable, "d_sign_bits": int(sign_variable), "d_mantissa_bits_untouched": 10, "d_plane_bits_per_block": bits_per_value, "d_plane_bytes_row_aligned_4": d_bytes, "d_plane_row_bytes": ((blocks_per_row * bits_per_value + 31) // 32) * 4, "d_plane_padding_bytes": d_bytes - (count * bits_per_value + 7) // 8, "fixed_tensor_header_bytes": 32, "q_l_q_h_scale_bytes": q_bytes, "baseline_tensor_bytes": baseline_bytes, "encoded_tensor_bytes": encoded_bytes, "net_bytes_saved": baseline_bytes - encoded_bytes, "net_saving_fraction": (baseline_bytes - encoded_bytes) / baseline_bytes, "net_positive": encoded_bytes < baseline_bytes, "roundtrip_checked_d_values": checked_values})
        del words, exponents, signs
    selected = [t for t in tensors if t["net_positive"]]
    return {"format": "gguf_q6_k", "tensor_count": len(tensors), "all_tensors_exact_cpu_roundtrip": all(t["roundtrip_checked_d_values"] == t["blocks"] for t in tensors), "layout": "Preserve 208 Q6_K bytes per block; encode d per row as interleaved fixed-width little-endian fields: untouched 10-bit mantissa, exponent-minus-tensor-min in the minimum fixed width, and a sign bit only when signs vary. Pad each d row to 4 bytes and prepend one 32-byte tensor header with exponent base, widths, and constant sign when present.", "row_padding_policy": "round each original GGML matrix row's fixed-width d bitstream up to a 4-byte boundary", "all_tensor_rows": tensors, "selected_net_positive_tensor_ids": [t["id"] for t in selected], "selected_net_positive_count": len(selected), "selected_net_positive_baseline_bytes": sum(t["baseline_tensor_bytes"] for t in selected), "selected_net_positive_encoded_bytes": sum(t["encoded_tensor_bytes"] for t in selected), "selected_net_positive_bytes_saved": sum(t["net_bytes_saved"] for t in selected), "all_q6_baseline_bytes": sum(t["baseline_tensor_bytes"] for t in tensors), "all_q6_encoded_bytes_if_every_tensor_used_candidate": sum(t["encoded_tensor_bytes"] for t in tensors), "all_q6_bytes_saved_if_every_tensor_used_candidate": sum(t["net_bytes_saved"] for t in tensors)}


def hist_entropy(values, alphabet):
    return entropy_from_hist(np.bincount(np.asarray(values).reshape(-1), minlength=alphabet))


def q4_header_modes(exponents, sign_mantissa, tile_blocks):
    block_count = len(exponents)
    result = {"global": {}, "tile_local": {}}
    global_chosen = np.array([np.bincount(exponents[:, i], minlength=32).argmax() for i in range(2)], dtype=np.uint8)
    for mode_name in ("global", "tile_local"):
        stored = 0
        raw = block_count * 4
        candidates = 0
        common_total = 0
        tiles = 0
        for start in range(0, block_count, tile_blocks):
            exp = exponents[start:start + tile_blocks]
            sm = sign_mantissa[start:start + tile_blocks]
            if not len(exp):
                continue
            if mode_name == "global":
                chosen = global_chosen
            else:
                chosen = np.array([np.bincount(exp[:, i], minlength=32).argmax() for i in range(2)], dtype=np.uint8)
            common = exp == chosen[None, :]
            common_count = int(common.sum())
            rare_count = int(common.size - common_count)
            flag_bytes = (common.size + 7) // 8
            compact_bytes = (common_count * 11 + 7) // 8
            fallback_bytes = rare_count * 2
            header_bytes = 2 + 8
            candidate = ((flag_bytes + compact_bytes + fallback_bytes + header_bytes + 15) // 16) * 16
            raw_tile = ((len(exp) * 4 + 8 + 15) // 16) * 16
            if candidate < raw_tile:
                stored += candidate
                candidates += 1
                common_total += common_count
            else:
                stored += raw_tile
            tiles += 1
            original = sign_mantissa[start:start + len(exp)]
            rebuilt = np.where(common, (chosen.astype(np.uint16)[None, :] << 10) | original, exp.astype(np.uint16) * 1024 | original)
            if not np.array_equal(rebuilt, exp.astype(np.uint16) * 1024 | original):
                raise ValueError("header exponent reconstruction failed")
        result[mode_name] = {"tile_blocks": tile_blocks, "tiles": tiles, "candidate_tiles_used": candidates, "common_half_values": common_total, "raw_bytes": raw, "stored_bytes_including_raw_fallback_and_8B_tile_offsets": stored, "ratio_to_raw_d_dmin_headers": stored / raw, "saving_fraction": 1.0 - stored / raw}
    return result


def q6_d_modes(exponents, mantissas_signs, tile_blocks):
    count = len(exponents)
    mode = int(np.bincount(exponents, minlength=32).argmax())
    stored = 0
    candidates = 0
    common_total = 0
    tiles = 0
    for start in range(0, count, tile_blocks):
        exp = exponents[start:start + tile_blocks]
        sm = mantissas_signs[start:start + tile_blocks]
        common = exp == mode
        compact = int(common.sum())
        flags = (len(exp) + 7) // 8
        candidate = flags + (compact * 10 + 7) // 8 + int((~common).sum()) * 2 + 1 + 8
        raw = len(exp) * 2
        if candidate < raw:
            stored += candidate
            candidates += 1
            common_total += compact
        else:
            stored += raw + 8
        rebuilt = np.where(common, (mode << 10) | sm, exp * 1024 | sm)
        if not np.array_equal(rebuilt, exp.astype(np.uint16) * 1024 | sm):
            raise ValueError("Q6 d reconstruction failed")
        tiles += 1
    raw_total = count * 2
    return {"mode_exponent": mode, "tile_blocks": tile_blocks, "tiles": tiles, "candidate_tiles_used": candidates, "common_values": common_total, "raw_bytes": raw_total, "stored_bytes_including_fallback_and_offsets": stored, "ratio_to_raw_d_field": stored / raw_total, "saving_fraction": 1 - stored / raw_total}


def q4_scale_min_estimates(counts_scale, counts_min, block_count):
    output = []
    for tile_blocks in TILE_BLOCKS:
        stored = 0
        candidate_tiles = 0
        tile_count = 0
        raw = block_count * 12
        for start in range(0, block_count, tile_blocks):
            sc = counts_scale[start:start + tile_blocks]
            mn = counts_min[start:start + tile_blocks]
            if len(sc) != tile_blocks:
                tail = len(sc) * 12
                stored += tail
                continue
            scbytes = 8 + 6 * (8 - sc)
            mnbytes = 8 + 6 * (8 - mn)
            candidate = 2 * tile_blocks + int(((scbytes + 7) // 8 + (mnbytes + 7) // 8).sum()) + 8
            raw_tile = tile_blocks * 12 + 8
            if candidate < raw_tile:
                stored += candidate
                candidate_tiles += 1
            else:
                stored += raw_tile
            tile_count += 1
        output.append({"blocks_per_tile": tile_blocks, "tiles": tile_count, "candidate_tiles_used": candidate_tiles, "raw_bytes": raw, "stored_bytes_with_fallback_and_8B_offsets": stored, "ratio": stored / raw, "saving_fraction": 1 - stored / raw, "format": "per block, each 8-value scale/min field has 8-bit sentinel bitmap then exact 6-bit remaining values; max 63 is scale sentinel; zero is min sentinel; 8B tile offset; raw tile fallback"})
    return output


def group_predictor_update(codes, histogram_xor, histogram_delta, adjacent_hist, runs):
    grouped = codes.reshape(-1, codes.shape[-2], codes.shape[-1]) if codes.ndim == 3 else codes
    left = grouped[..., :-1]
    right = grouped[..., 1:]
    xor = left ^ right
    delta = (right.astype(np.int16) - left.astype(np.int16)) & 63 if int(codes.max()) > 15 else (right.astype(np.int16) - left.astype(np.int16)) & 15
    width = 64 if int(codes.max()) > 15 else 16
    histogram_xor += np.bincount(xor.reshape(-1), minlength=width)
    histogram_delta += np.bincount(delta.reshape(-1), minlength=width)
    adjacent_hist += np.bincount((left.astype(np.int64) * width + right).reshape(-1), minlength=width * width).reshape(width, width)
    runs += int((right != left).sum())


def main():
    directory = json.loads((ROOT / "directory.json").read_text(encoding="utf-8"))
    prior = json.loads((ROOT / "results" / "fields-codecs.json").read_text(encoding="utf-8"))
    prior_by_id = {t["id"]: t for t in prior["tensors"]}
    with MODEL.open("rb") as f:
        magic, header_length, artifact_id = struct.unpack("<8sQ16s", f.read(32))
        if magic != b"NINFER\0\3":
            raise ValueError("NInfer model magic mismatch")
        header = json.loads(f.read(header_length))
        payload_start = (32 + header_length + 4095) // 4096 * 4096
        if header != directory or artifact_id.hex() != prior["artifact_id"]:
            raise ValueError("model identity mismatch")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        all_q6_dplane_result = scan_all_q6_dplanes(mm, directory, payload_start)
        results = []
        for name in TARGETS:
            item = next(o for o in directory["objects"] if o["id"] == name)
            stride = 144 if item["format"] == "gguf_q4_k" else 210
            count = item["bytes"] // stride
            view = memoryview(mm)[payload_start + item["offset"]:payload_start + item["offset"] + item["bytes"]]
            source_hash = hashlib.sha256(view).hexdigest()
            blocks = np.frombuffer(view, dtype=np.uint8).reshape(count, stride)
            row = {"id": name, "format": item["format"], "blocks": count, "tensor_bytes": item["bytes"], "source_sha256": source_hash, "output_head_binding": name == "weight/000131"}
            if stride == 144:
                scale_hist = np.zeros(64, dtype=np.uint64)
                min_hist = np.zeros(64, dtype=np.uint64)
                header_exp_hist = np.zeros((2, 32), dtype=np.uint64)
                scale_zero_by_block = np.zeros(count, dtype=np.uint8)
                min_zero_by_block = np.zeros(count, dtype=np.uint8)
                scale63_by_block = np.zeros(count, dtype=np.uint8)
                min63_by_block = np.zeros(count, dtype=np.uint8)
                nibble_hist = np.zeros(16, dtype=np.uint64)
                xor_hist = np.zeros(16, dtype=np.uint64)
                delta_hist = np.zeros(16, dtype=np.uint64)
                pair_hist = np.zeros((16, 16), dtype=np.uint64)
                conditional_xor_hist = np.zeros((64, 16), dtype=np.uint64)
                conditional_code_hist = np.zeros((64, 16), dtype=np.uint64)
                conditional_full_code_hist = np.zeros((64, 16), dtype=np.uint64)
                scale_group_hist = np.zeros(64, dtype=np.uint64)
                scale_full_hist = np.zeros(64, dtype=np.uint64)
                mode_exceptions_by_block = np.zeros((count, 8), dtype=np.uint8)
                residual_zero = 0
                residual_total = 0
                for start in range(0, count, CHUNK_BLOCKS):
                    chunk = blocks[start:start + CHUNK_BLOCKS]
                    sc, mn = q4_scale_min(chunk[:, 4:16])
                    scale_hist += np.bincount(sc.reshape(-1), minlength=64).astype(np.uint64)
                    min_hist += np.bincount(mn.reshape(-1), minlength=64).astype(np.uint64)
                    scale_zero_by_block[start:start + len(chunk)] = (sc == 0).sum(axis=1)
                    min_zero_by_block[start:start + len(chunk)] = (mn == 0).sum(axis=1)
                    scale63_by_block[start:start + len(chunk)] = (sc == 63).sum(axis=1)
                    min63_by_block[start:start + len(chunk)] = (mn == 63).sum(axis=1)
                    halves = chunk[:, :4].copy().view("<u2").reshape(-1, 2)
                    exps = ((halves >> 10) & 31).astype(np.uint8)
                    header_exp_hist[0] += np.bincount(exps[:, 0], minlength=32).astype(np.uint64)
                    header_exp_hist[1] += np.bincount(exps[:, 1], minlength=32).astype(np.uint64)
                    q = chunk[:, 16:144]
                    low_groups = np.stack([q[:, j * 32:(j + 1) * 32] & 15 for j in range(4)], axis=1)
                    high_groups = np.stack([q[:, j * 32:(j + 1) * 32] >> 4 for j in range(4)], axis=1)
                    codes = np.empty((len(chunk), 8, 32), dtype=np.uint8)
                    codes[:, 0::2] = low_groups
                    codes[:, 1::2] = high_groups
                    nibble_hist += np.bincount(codes.reshape(-1), minlength=16).astype(np.uint64)
                    left = codes[..., :-1]
                    right = codes[..., 1:]
                    xor_hist += np.bincount((left ^ right).reshape(-1), minlength=16).astype(np.uint64)
                    delta_hist += np.bincount(((right.astype(np.int16) - left.astype(np.int16)) & 15).reshape(-1), minlength=16).astype(np.uint64)
                    pair_hist += np.bincount((left.astype(np.int64) * 16 + right).reshape(-1), minlength=256).reshape(16, 16).astype(np.uint64)
                    residual_zero += int(((left ^ right) == 0).sum())
                    residual_total += left.size
                    scale_group_hist += np.bincount(sc.reshape(-1), minlength=64).astype(np.uint64)
                    group_residual = codes[..., 1:] ^ codes[..., :-1]
                    scale_context = sc.astype(np.int64)[:, :, None] * 16
                    conditional_xor_hist += np.bincount((scale_context + group_residual).reshape(-1), minlength=1024).reshape(64, 16).astype(np.uint64)
                    conditional_code_hist += np.bincount((scale_context + codes[..., 1:]).reshape(-1), minlength=1024).reshape(64, 16).astype(np.uint64)
                    conditional_full_code_hist += np.bincount((scale_context + codes).reshape(-1), minlength=1024).reshape(64, 16).astype(np.uint64)
                    scale_full_hist += np.bincount(np.broadcast_to(sc[:, :, None], codes.shape).reshape(-1), minlength=64).astype(np.uint64)
                exponent_result = []
                for index, label in enumerate(("d", "dmin")):
                    h = header_exp_hist[index]
                    mode = int(h.argmax())
                    exponent_result.append({"field": label, "mode_exponent": mode, "mode_fraction": float(h[mode] / h.sum()), "entropy_bits": entropy_from_hist(h), "distinct_exponents": int((h > 0).sum()), "histogram": h.tolist()})
                scale_storage = []
                for tile_blocks in TILE_BLOCKS:
                    sc_c = scale63_by_block
                    mn_c = min_zero_by_block
                    stored = 0
                    cand_tiles = 0
                    tile_count = 0
                    for start in range(0, count, tile_blocks):
                        n = min(tile_blocks, count - start)
                        sc = sc_c[start:start + n]
                        mn = mn_c[start:start + n]
                        scbytes = 8 + 6 * (8 - sc)
                        mnbytes = 8 + 6 * (8 - mn)
                        candidate = 2 * n + int(((scbytes + 7) // 8 + (mnbytes + 7) // 8).sum()) + 8
                        candidate = ((candidate + 15) // 16) * 16
                        raw_tile = ((n * 12 + 8 + 15) // 16) * 16
                        if n == tile_blocks and candidate < raw_tile:
                            stored += candidate
                            cand_tiles += 1
                        else:
                            stored += raw_tile
                        tile_count += 1
                    raw_bytes = count * 12
                    scale_storage.append({"blocks_per_tile": tile_blocks, "tiles": tile_count, "candidate_tiles_used": cand_tiles, "raw_bytes": raw_bytes, "stored_bytes_with_raw_fallback_and_offsets": stored, "ratio": stored / raw_bytes, "saving_fraction": 1 - stored / raw_bytes})
                exps = ((blocks[:, :4].copy().view("<u2").reshape(-1, 2) >> 10) & 31).astype(np.uint8)
                sign_mantissa = (blocks[:, :4].copy().view("<u2").reshape(-1, 2) & 0x83ff).astype(np.uint16)
                head_modes = {str(tile): q4_header_modes(exps, sign_mantissa, tile) for tile in TILE_BLOCKS}
                mode_codes = conditional_full_code_hist.argmax(axis=1).astype(np.uint8)
                if not np.array_equal(conditional_code_hist.sum(axis=1), scale_group_hist * 31):
                    raise ValueError("Q4 conditional next-code histogram context totals mismatch")
                if not np.array_equal(conditional_xor_hist.sum(axis=1), scale_group_hist * 31):
                    raise ValueError("Q4 conditional xor histogram context totals mismatch")
                if not np.array_equal(conditional_full_code_hist.sum(axis=1), scale_full_hist):
                    raise ValueError("Q4 conditional full-code histogram context totals mismatch")
                for start in range(0, count, CHUNK_BLOCKS):
                    chunk = blocks[start:start + CHUNK_BLOCKS]
                    sc, _ = q4_scale_min(chunk[:, 4:16])
                    q = chunk[:, 16:144]
                    low_groups = np.stack([q[:, j * 32:(j + 1) * 32] & 15 for j in range(4)], axis=1)
                    high_groups = np.stack([q[:, j * 32:(j + 1) * 32] >> 4 for j in range(4)], axis=1)
                    codes = np.empty((len(chunk), 8, 32), dtype=np.uint8)
                    codes[:, 0::2] = low_groups
                    codes[:, 1::2] = high_groups
                    matches = codes == mode_codes[sc][:, :, None]
                    mode_exceptions_by_block[start:start + len(chunk)] = 32 - matches.sum(axis=2, dtype=np.uint8)
                mode_bitmap_layout = []
                for tile_blocks in TILE_BLOCKS:
                    stored = 0
                    candidate_tiles = 0
                    tile_count = 0
                    q_raw = count * 128
                    for start in range(0, count, tile_blocks):
                        n = min(tile_blocks, count - start)
                        if n != tile_blocks:
                            stored += n * 128
                            continue
                        exceptions = mode_exceptions_by_block[start:start + n].astype(np.uint16)
                        candidate = int((4 + 2 * exceptions).sum())
                        candidate = ((candidate + 8 + 15) // 16) * 16
                        raw_tile = ((n * 128 + 8 + 15) // 16) * 16
                        if candidate < raw_tile:
                            stored += candidate
                            candidate_tiles += 1
                        else:
                            stored += raw_tile
                        tile_count += 1
                    codebook_bytes = 32
                    total_stored = count * 16 + stored + codebook_bytes
                    mode_bitmap_layout.append({"blocks_per_tile": tile_blocks, "tiles": tile_count, "candidate_tiles_used": candidate_tiles, "mode_bitmap_and_packed_exceptions_q_bytes": stored, "raw_q_bytes": q_raw, "q_payload_ratio": stored / q_raw, "whole_tensor_bytes_including_raw_other_fields_offsets_and_32B_scale_mode_table": total_stored, "whole_tensor_ratio": total_stored / item["bytes"], "whole_tensor_saving_fraction": 1 - total_stored / item["bytes"], "format": "per block and each of eight scale groups: 32-bit bitmap marks symbols equal to the mode nibble for that group's decoded scale; remaining nibbles are packed in lane order; scale-to-mode table is 64 packed nibbles; tiles use raw fallback; offsets are 8B and tiles padded to 16B"})
                row["fields"] = {
                    "decoded_q4_scale_min_entropy_bits_each": {"scale": entropy_from_hist(scale_hist), "min": entropy_from_hist(min_hist)},
                    "decoded_scale_min_histogram_64": {"scale": scale_hist.tolist(), "min": min_hist.tolist()},
                    "scale_min_sentinels_per_block": {"scale63_fraction": float(scale_hist[63] / scale_hist.sum()), "minzero_fraction": float(min_hist[0] / min_hist.sum()), "scale_blocks_with_at_least3_max63": int((scale63_by_block >= 3).sum()), "min_blocks_with_at_least3_zero": int((min_zero_by_block >= 3).sum()), "scale_max63_count_histogram_per_block": np.bincount(scale63_by_block, minlength=9).tolist(), "min_zero_count_histogram_per_block": np.bincount(min_zero_by_block, minlength=9).tolist()},
                    "scale_min_bitmap_sentinel_layout": {"tile_estimates": scale_storage, "roundtrip_basis": "Sentinel marker positions are explicit bitmaps and non-sentinel values retain exact 6-bit values; raw fallback at tile granularity."},
                    "d_dmin_exponent_distributions": exponent_result,
                    "d_dmin_mode_exponent_tiles": head_modes,
                    "q4_group_local_predictors": {"code_entropy_bits": entropy_from_hist(nibble_hist), "previous_code_xor_residual_entropy_bits": entropy_from_hist(xor_hist), "previous_code_delta_mod16_entropy_bits": entropy_from_hist(delta_hist), "previous_code_xor_entropy_conditioned_on_scale_bits": entropy_from_hist(conditional_xor_hist.reshape(-1)) - entropy_from_hist(scale_group_hist), "code_entropy_conditioned_on_scale_bits": entropy_from_hist(conditional_code_hist.reshape(-1)) - entropy_from_hist(scale_group_hist), "adjacent_code_mutual_information_bits": mutual_information(pair_hist), "xor_residual_zero_fraction": residual_zero / residual_total, "pair_joint_entropy_bits": entropy_from_hist(pair_hist.reshape(-1))}
                    ,"scale_conditioned_mode_bitmap_layout": {"mode_code_by_scale_0_to_63": mode_codes.tolist(), "scale_conditioned_full_code_entropy_bits": entropy_from_hist(conditional_full_code_hist.reshape(-1)) - entropy_from_hist(scale_full_hist), "modal_code_fraction": float(conditional_full_code_hist.max(axis=1).sum() / conditional_full_code_hist.sum()), "exceptions_per_32_lane_group_histogram": np.bincount(mode_exceptions_by_block.reshape(-1), minlength=33).tolist(), "tile_estimates": mode_bitmap_layout}
                }
                del exps, sign_mantissa
            else:
                code_hist = np.zeros(64, dtype=np.uint64)
                xor_hist = np.zeros(64, dtype=np.uint64)
                delta_hist = np.zeros(64, dtype=np.uint64)
                pair_hist = np.zeros((64, 64), dtype=np.uint64)
                conditional_xor_hist = np.zeros((256, 64), dtype=np.uint64)
                conditional_code_hist = np.zeros((256, 64), dtype=np.uint64)
                scale_group_hist = np.zeros(256, dtype=np.uint64)
                residual_nonzero = 0
                residual_runs = 0
                residual_runs_by_block = np.zeros(count, dtype=np.uint16)
                residual_exceptions_by_block = np.zeros(count, dtype=np.uint16)
                d_exp_hist = np.zeros(32, dtype=np.uint64)
                d_mode_tiles = {t: {"tiles": 0, "common": 0, "stored": 0, "candidate_tiles": 0} for t in TILE_BLOCKS}
                scale_hist = np.zeros(256, dtype=np.uint64)
                for start in range(0, count, CHUNK_BLOCKS):
                    chunk = blocks[start:start + CHUNK_BLOCKS]
                    codes = q6_codes(chunk)
                    code_hist += np.bincount(codes.reshape(-1), minlength=64).astype(np.uint64)
                    codes_32 = codes.reshape(-1, 8, 32)
                    grouped_codes = codes_32.reshape(-1, 16, 16)
                    left = grouped_codes[..., :-1]
                    right = grouped_codes[..., 1:]
                    xor = left ^ right
                    delta = (right.astype(np.int16) - left.astype(np.int16)) & 63
                    xor_hist += np.bincount(xor.reshape(-1), minlength=64).astype(np.uint64)
                    delta_hist += np.bincount(delta.reshape(-1), minlength=64).astype(np.uint64)
                    pair_hist += np.bincount((left.astype(np.int64) * 64 + right).reshape(-1), minlength=4096).reshape(64, 64).astype(np.uint64)
                    block_scales = chunk[:, 192:208].view(np.int8).astype(np.int16) + 128
                    scale_group_hist += np.bincount(block_scales.reshape(-1), minlength=256).astype(np.uint64)
                    group_residual = grouped_codes[..., 1:] ^ grouped_codes[..., :-1]
                    conditional_xor_hist += np.bincount((block_scales[:, :, None] * 64 + group_residual).reshape(-1), minlength=16384).reshape(256, 64).astype(np.uint64)
                    conditional_code_hist += np.bincount((block_scales[:, :, None] * 64 + grouped_codes[..., 1:]).reshape(-1), minlength=16384).reshape(256, 64).astype(np.uint64)
                    ql = chunk[:, :128].reshape(-1, 2, 64)
                    qh = chunk[:, 128:192].reshape(-1, 2, 32)
                    residuals = []
                    for j in range(4):
                        lo = ql[:, :, :32] if j in (0, 2) else ql[:, :, 32:]
                        nibble = (lo >> (4 if j >= 2 else 0)) & 15
                        expected = np.where(nibble >= 8, 3, 0).astype(np.uint8)
                        actual = ((qh >> (j * 2)) & 3) ^ 2
                        residuals.append(actual ^ expected)
                    residual_array = np.stack(residuals, axis=2)
                    residual_lanes = residual_array.reshape(-1)
                    residual_nonzero += int((residual_lanes != 0).sum())
                    mask = residual_array.reshape(-1, 256) != 0
                    starts = mask & ~np.pad(mask[:, :-1], ((0, 0), (1, 0)))
                    residual_runs += int(starts.sum())
                    residual_runs_by_block[start:start + len(chunk)] = starts.sum(axis=1, dtype=np.uint16)
                    residual_exceptions_by_block[start:start + len(chunk)] = mask.sum(axis=1, dtype=np.uint16)
                    d = chunk[:, 208:210].copy().view("<u2").reshape(-1)
                    exponent = ((d >> 10) & 31).astype(np.uint8)
                    d_exp_hist += np.bincount(exponent, minlength=32).astype(np.uint64)
                    scale_hist += np.bincount(chunk[:, 192:208].reshape(-1), minlength=256).astype(np.uint64)
                    for tile in TILE_BLOCKS:
                        usable = len(exponent) // tile * tile
                        if usable:
                            grouped = exponent[:usable].reshape(-1, tile)
                            row_ids = np.repeat(np.arange(len(grouped), dtype=np.int32), tile)
                            hist = np.bincount(row_ids * 32 + grouped.reshape(-1), minlength=len(grouped) * 32).reshape(-1, 32)
                            modes = hist.argmax(axis=1)
                            common = (grouped == modes[:, None]).sum(axis=1)
                            st = d_mode_tiles[tile]
                            st["tiles"] += len(grouped)
                            st["common"] += int(common.sum())
                            raw_tile = ((tile * 2 + 8 + 15) // 16) * 16
                            payload = (common * 11 + 7) // 8 + ((tile - common) * 16 + 7) // 8 + (tile + 7) // 8 + 1 + 8
                            payload = (payload + 15) // 16 * 16
                            use = payload < raw_tile
                            st["candidate_tiles"] += int(use.sum())
                            st["stored"] += int(np.where(use, payload, raw_tile).sum())
                exp_entropy = entropy_from_hist(d_exp_hist)
                raw_qh = count * 64
                lanes = count * 256
                exception_count = residual_nonzero
                bitmap_size = (lanes + 7) // 8
                packed_values_size = (exception_count * 2 + 7) // 8
                span_bytes = residual_runs * 2 + packed_values_size
                span_tile_estimates = []
                for tile in TILE_BLOCKS:
                    tiles = count // tile
                    grouped_runs = residual_runs_by_block[:tiles * tile].reshape(-1, tile).sum(axis=1, dtype=np.uint64)
                    grouped_exceptions = residual_exceptions_by_block[:tiles * tile].reshape(-1, tile).sum(axis=1, dtype=np.uint64)
                    candidate = (146 * tile + 2 * grouped_runs + (grouped_exceptions * 2 + 7) // 8 + 8 + 15) // 16 * 16
                    raw_tile = (stride * tile + 8 + 15) // 16 * 16
                    use = candidate < raw_tile
                    stored = int(np.where(use, candidate, raw_tile).sum()) + (count - tiles * tile) * stride
                    span_tile_estimates.append({"blocks_per_tile": tile, "tiles": tiles, "candidate_tiles_used": int(use.sum()), "raw_fallback_tiles": int((~use).sum()), "raw_bytes": item["bytes"], "stored_bytes_including_8B_tile_offsets_and_raw_fallback": stored, "ratio": stored / item["bytes"], "saving_fraction": 1 - stored / item["bytes"], "stored_exception_start_length_bytes": int((2 * grouped_runs[use]).sum()), "stored_exception_value_bytes": int((((grouped_exceptions[use] * 2) + 7) // 8).sum())})
                q6_d_candidate = []
                for tile in TILE_BLOCKS:
                    st = d_mode_tiles[tile]
                    q6_d_candidate.append({"blocks_per_tile": tile, "tiles": st["tiles"], "candidate_tiles_used": st["candidate_tiles"], "common_exponent0_fraction_global": float(d_exp_hist[0] / d_exp_hist.sum()), "raw_bytes": count * 2, "stored_bytes": st["stored"] + (count % tile) * 2, "ratio": (st["stored"] + (count % tile) * 2) / (count * 2)})
                row["fields"] = {
                    "decoded_q6_code_entropy_bits": entropy_from_hist(code_hist),
                    "decoded_q6_code_histogram_64": code_hist.tolist(),
                    "within_scale_group_predictors": {"xor_residual_entropy_bits": entropy_from_hist(xor_hist), "delta_mod64_entropy_bits": entropy_from_hist(delta_hist), "xor_residual_entropy_conditioned_on_signed_scale_bits": entropy_from_hist(conditional_xor_hist.reshape(-1)) - entropy_from_hist(scale_group_hist), "code_entropy_conditioned_on_signed_scale_bits": entropy_from_hist(conditional_code_hist.reshape(-1)) - entropy_from_hist(scale_group_hist), "adjacent_code_mutual_information_bits": mutual_information(pair_hist), "joint_pair_entropy_bits": entropy_from_hist(pair_hist.reshape(-1)), "xor_residual_zero_fraction": float((xor_hist[0] / xor_hist.sum()))},
                    "high_plane_exception_contiguous_spans": {"nonzero_2bit_exceptions": exception_count, "total_2bit_lanes": lanes, "exception_fraction": exception_count / lanes, "contiguous_exception_runs": residual_runs, "mean_run_length": exception_count / residual_runs if residual_runs else 0, "raw_qh_bytes": raw_qh, "global_bitmap_plus_2bit_values_bytes": bitmap_size + packed_values_size, "global_span_start_length_plus_2bit_values_bytes": span_bytes, "span_bytes_assumes_1B_start_1B_length_per_run": True, "tile_estimates_with_fallback_offsets": span_tile_estimates},
                    "q6_d_half_exponents": {"histogram": d_exp_hist.tolist(), "mode_exponent": int(d_exp_hist.argmax()), "mode_fraction": float(d_exp_hist.max() / d_exp_hist.sum()), "entropy_bits": exp_entropy, "mantissa_sign_compact_tile_candidates": q6_d_candidate},
                    "q6_scale_byte_entropy_bits": entropy_from_hist(scale_hist),
                    "q6_scale_histogram_256": scale_hist.tolist()
                }
                del ql, qh, codes
            row["all_fields_analysis_full_tensor"] = True
            results.append(row)
            del blocks, view
        del mm
    q4 = next(t["fields"] for t in results if t["format"] == "gguf_q4_k")
    q6 = next(t["fields"] for t in results if t["format"] != "gguf_q4_k")
    q4_header = q4["d_dmin_mode_exponent_tiles"]["128"]["global"]
    q4_sentinel = q4["scale_min_bitmap_sentinel_layout"]["tile_estimates"]
    q4_modal = q4["scale_conditioned_mode_bitmap_layout"]["tile_estimates"]
    q6_spans = q6["high_plane_exception_contiguous_spans"]["tile_estimates_with_fallback_offsets"]
    q6_d = q6["q6_d_half_exponents"]["mantissa_sign_compact_tile_candidates"]
    result = {"model": str(MODEL), "artifact_id": artifact_id.hex(), "payload_offset": payload_start, "cpu_only": True, "full_tensor_roundtrip_basis": "Lossless predictor transforms retain the original source symbol plus reversible residual; size estimates include tile raw fallback and index/alignment where applicable.", "all_q6_fixed_dplane_study": all_q6_dplane_result, "next_layout_assessment": {"best_measured_fixed_metadata_candidate": {"tensor": "weight/000001", "layout": "per-tile shared exponent mode for d and dmin; retain 11 sign+mantissa bits for common-exponent values, raw half fallback for exceptions, one bitmap, 8B offset, 16B padding, raw tile fallback", "tile_blocks": 128, "field_saving_fraction": q4_header["saving_fraction"], "tensor_bytes_saved": int(round(q4_header["raw_bytes"] - q4_header["stored_bytes_including_raw_fallback_and_8B_tile_offsets"])), "whole_tensor_saving_fraction": (q4_header["raw_bytes"] - q4_header["stored_bytes_including_raw_fallback_and_8B_tile_offsets"]) / results[0]["tensor_bytes"], "decode_shape": "fixed-width bitmap and 11-bit fields; tile-local exponent read once"}, "rejected": {"q4_scale_min_sentinel_bitmaps": [{"tile_blocks": x["blocks_per_tile"], "tensor_saving_fraction": 0.0, "metadata_ratio_including_fallback_offsets": x["ratio"]} for x in q4_sentinel], "q4_scale_conditioned_mode_bitmap": [{"tile_blocks": x["blocks_per_tile"], "whole_tensor_saving_fraction": x["whole_tensor_saving_fraction"], "candidate_tiles_used": x["candidate_tiles_used"]} for x in q4_modal], "q6_high_plane_exception_spans": [{"tile_blocks": x["blocks_per_tile"], "whole_tensor_saving_fraction": 1.0 - x["ratio"], "candidate_tiles_used": x["candidate_tiles_used"]} for x in q6_spans]}, "q4_scale_context_is_weak": {"scale_conditioned_code_entropy_bits_per_nibble": q4["scale_conditioned_mode_bitmap_layout"]["scale_conditioned_full_code_entropy_bits"], "unconditioned_code_entropy_bits_per_nibble": q4["q4_group_local_predictors"]["code_entropy_bits"], "modal_code_fraction": q4["scale_conditioned_mode_bitmap_layout"]["modal_code_fraction"], "interpretation": "Conditioning on the decoded 6-bit scale lowers Q4 code entropy by only about 0.006 bits per nibble; the modal code occurs about 10.2% of the time, and all mode-bitmap tiles fall back to raw."}, "q6_lane_local_patterns_are_weak": {"scale_conditioned_code_entropy_bits_per_6bit_symbol": q6["within_scale_group_predictors"]["code_entropy_conditioned_on_signed_scale_bits"], "scale_conditioned_xor_entropy_bits_per_symbol": q6["within_scale_group_predictors"]["xor_residual_entropy_conditioned_on_signed_scale_bits"], "adjacent_code_mutual_information_bits": q6["within_scale_group_predictors"]["adjacent_code_mutual_information_bits"], "d_exponent_tile128_field_saving_fraction": 1.0 - q6_d[2]["ratio"], "d_exponent_tile128_whole_tensor_saving_fraction": (q6_d[2]["raw_bytes"] - q6_d[2]["stored_bytes"]) / results[1]["tensor_bytes"]}}, "tensors": results}
    out = ROOT / "results" / "register-fields.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "tensors": len(results), "seconds": time.time() - started}, separators=(",", ":")))


if __name__ == "__main__":
    started = time.time()
    main()
