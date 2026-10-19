import hashlib
import csv
import json
import math
import mmap
import pathlib
import struct
import time

import numpy as np


ROOT = pathlib.Path("/mnt/c/AI/tuning/ninfer-lossless-20260928")
MODEL = pathlib.Path("/mnt/c/AI/staging/qwen38-jonathan-ninfer/qwen38-jonathan-q4km-proposal.ninfer")
TENSORS = ("weight/000001", "weight/000060", "weight/000131")
TILE_BLOCKS = (32, 64, 128, 256)


def attach_short_stream_estimates(result):
    prior_path = ROOT / "alternatives-codecs.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    prior_by_id = {row["id"]: row for row in prior["tensors"]}
    ans_path = ROOT / "results" / "ans-weight-000060.csv"
    with ans_path.open("r", newline="", encoding="utf-8") as f:
        exact_ans_rows = [row for row in csv.DictReader(f) if row["blocks_per_tile"] in ("16", "64", "256") and row["streams"] == "32"]
    estimates = {}
    for tensor in result["tensors"]:
        old = prior_by_id.get(tensor["id"])
        q6 = tensor["format"] == "gguf_q6_k"
        if not old and not q6:
            continue
        entropy = tensor["fields"].get("quant_code_entropy_bits_per_symbol") if q6 else old["nibble_entropy_bits_symbol"]
        total_symbols = tensor["blocks"] * 256
        huffman_table = 32 if q6 else 8
        rans_table = 64 if q6 else 16
        rows = []
        for symbols in (32, 64, 128, 256):
            payload = math.ceil(symbols * entropy / 8)
            stream_count = (total_symbols + symbols - 1) // symbols
            if q6:
                raw = math.ceil(symbols * 6 / 8)
                huffman_byte = payload + 4
                huffman_16 = ((huffman_byte + 15) // 16) * 16
                rans_byte = payload + 8
                rans_16 = ((rans_byte + 15) // 16) * 16
                tans = stream_count * (((payload + 2 + 4 + 4 + 15) // 16) * 16)
                rows.append({"symbols_per_restart": symbols, "raw_6bit_code_bytes_per_stream": raw, "entropy_bits_per_symbol_full_tensor": entropy, "entropy_payload_bytes_lower_bound_per_stream": payload, "streams_per_tensor": stream_count, "shared_huffman_code_lengths_table_bytes_per_tensor": huffman_table, "huffman_restart_offset_bytes": 4, "huffman_byte_aligned_bytes_per_stream": huffman_byte, "huffman_byte_aligned_tensor_ratio": (stream_count * huffman_byte + huffman_table) / (total_symbols * 6 / 8), "huffman_16byte_aligned_bytes_per_stream": huffman_16, "huffman_16byte_aligned_tensor_ratio": (stream_count * huffman_16 + huffman_table) / (total_symbols * 6 / 8), "shared_rans_frequency_table_bytes_per_tensor": rans_table, "rans_state_and_restart_offset_bytes": 8, "rans_byte_aligned_bytes_per_stream": rans_byte, "rans_byte_aligned_tensor_ratio": (stream_count * rans_byte + rans_table) / (total_symbols * 6 / 8), "rans_16byte_aligned_bytes_per_stream": rans_16, "rans_16byte_aligned_tensor_ratio": (stream_count * rans_16 + rans_table) / (total_symbols * 6 / 8), "tans10bit_state_plus_pointer4B_and_offset4B_theoretical_bytes_per_stream": tans // stream_count, "tans10bit_state_theoretical_tensor_ratio": tans / (total_symbols * 6 / 8), "rates_are_entropy_based_estimates_not_coder_measurements": True})
            else:
                raw = (symbols + 1) // 2
                huffman_byte = payload + 4
                huffman_16 = ((huffman_byte + 15) // 16) * 16
                rans_byte = payload + 8
                rans_16 = ((rans_byte + 15) // 16) * 16
                tans = stream_count * (((payload + 2 + 4 + 4 + 15) // 16) * 16)
                rows.append({"symbols_per_restart": symbols, "raw_nibble_bytes_per_stream": raw, "entropy_bits_per_symbol_from_full_sample": entropy, "entropy_payload_bytes_lower_bound_per_stream": payload, "streams_per_tensor": stream_count, "shared_huffman_code_lengths_table_bytes_per_tensor": huffman_table, "huffman_restart_offset_bytes": 4, "huffman_byte_aligned_bytes_per_stream": huffman_byte, "huffman_byte_aligned_tensor_ratio": (stream_count * huffman_byte + huffman_table) / (total_symbols / 2), "huffman_16byte_aligned_bytes_per_stream": huffman_16, "huffman_16byte_aligned_tensor_ratio": (stream_count * huffman_16 + huffman_table) / (total_symbols / 2), "shared_rans_frequency_table_bytes_per_tensor": rans_table, "rans_state_and_restart_offset_bytes": 8, "rans_byte_aligned_bytes_per_stream": rans_byte, "rans_byte_aligned_tensor_ratio": (stream_count * rans_byte + rans_table) / (total_symbols / 2), "rans_16byte_aligned_bytes_per_stream": rans_16, "rans_16byte_aligned_tensor_ratio": (stream_count * rans_16 + rans_table) / (total_symbols / 2), "tans10bit_state_plus_pointer4B_and_offset4B_theoretical_bytes_per_stream": tans // stream_count, "tans10bit_state_theoretical_tensor_ratio": tans / (total_symbols / 2), "rates_are_entropy_based_estimates_not_coder_measurements": True})
        huffman = old.get("independent_tile_huffman", {}) if old else {}
        estimates[tensor["id"]] = {"source": "Full-tensor symbol entropy plus calculated ideal payload; codebook is shared per tensor, while offset/state and 16-byte alignment are charged per restart. No new coder was run for these rates.", "prior_independent_tile_huffman_ratio": huffman.get("ratio_to_sample_raw_bytes"), "prior_huffman_offset_bytes_per_tile": huffman.get("offset_bytes_per_tile_assumed"), "prior_huffman_alignment": huffman.get("tile_payload_alignment"), "short_streams": rows}
    estimates["weight/000060"]["prior_exact_full_tensor_rans_short_restart_rows"] = exact_ans_rows
    result["restart_entropy_estimates"] = estimates


def bits_pack(values, width):
    values = np.asarray(values, dtype=np.uint8).reshape(-1)
    out = bytearray((values.size * width + 7) // 8)
    for i, value in enumerate(values):
        bit = i * width
        out[bit >> 3] |= int(value) << (bit & 7)
    return bytes(out)


def bits_unpack(data, count, width):
    out = np.empty(count, dtype=np.uint8)
    for i in range(count):
        bit = i * width
        out[i] = (data[bit >> 3] >> (bit & 7)) & ((1 << width) - 1)
    return out


def pack2(values):
    values = np.asarray(values, dtype=np.uint8).reshape(-1)
    padded = np.pad(values, (0, (-values.size) % 4))
    quads = padded.reshape(-1, 4)
    return (quads[:, 0] | (quads[:, 1] << 2) | (quads[:, 2] << 4) | (quads[:, 3] << 6)).tobytes()


def unpack2(data, count):
    packed = np.frombuffer(data, dtype=np.uint8)
    lanes = ((packed[:, None] >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3).reshape(-1)
    return lanes[:count]


def q4_metadata_encode(tile):
    words = tile[:, :4].copy().view("<u2").reshape(-1, 2)
    exponents = words & 0x7c00
    same = (exponents[:, 0] == exponents[:, 1])
    signs = ((words >> 15) & 1).astype(np.uint8)
    mantissas = (words & 0x03ff).astype(np.uint16)
    flags = np.packbits(same.astype(np.uint8), bitorder="little").tobytes()
    compact_values = (((words[same, 0] >> 10) & 31).astype(np.uint32)
        | (signs[same, 0].astype(np.uint32) << 5)
        | (mantissas[same, 0].astype(np.uint32) << 6)
        | (signs[same, 1].astype(np.uint32) << 16)
        | (mantissas[same, 1].astype(np.uint32) << 17))
    compact_bits = np.unpackbits(compact_values.astype("<u4").view(np.uint8), bitorder="little").reshape(-1, 32)[:, :27]
    compact_payload = np.packbits(compact_bits, bitorder="little").tobytes()
    fallback_payload = words[~same].astype("<u2").tobytes()
    return flags, compact_payload, fallback_payload, same


def q4_metadata_decode(flags, compact_payload, fallback_payload, count):
    same = np.unpackbits(np.frombuffer(flags, dtype=np.uint8), bitorder="little")[:count].astype(bool)
    words = np.empty((count, 2), dtype=np.uint16)
    common_count = int(same.sum())
    bit_count = common_count * 27
    compact_bits = np.unpackbits(np.frombuffer(compact_payload, dtype=np.uint8), bitorder="little")[:bit_count].reshape(-1, 27).astype(np.uint64)
    values = (compact_bits << np.arange(27, dtype=np.uint64)).sum(axis=1)
    exponents = values & 31
    sign0 = (values >> 5) & 1
    mantissa0 = (values >> 6) & 1023
    sign1 = (values >> 16) & 1
    mantissa1 = (values >> 17) & 1023
    words[same, 0] = ((sign0 << 15) | (exponents << 10) | mantissa0).astype(np.uint16)
    words[same, 1] = ((sign1 << 15) | (exponents << 10) | mantissa1).astype(np.uint16)
    words[~same] = np.frombuffer(fallback_payload, dtype="<u2").reshape(-1, 2)
    if (bit_count + 7) // 8 != len(compact_payload):
        raise ValueError("Q4 metadata payload length mismatch")
    return words.astype("<u2").tobytes()


def q6_residual(blocks):
    ql = blocks[:, :128].reshape(-1, 2, 64)
    qh = blocks[:, 128:192].reshape(-1, 2, 32)
    residual = np.zeros_like(qh)
    for j in range(4):
        lo = ql[:, :, :32] if j in (0, 2) else ql[:, :, 32:]
        nibble = (lo >> (4 if j >= 2 else 0)) & 15
        actual = ((qh >> (j * 2)) & 3) ^ 2
        expected = np.where(nibble >= 8, 3, 0).astype(np.uint8)
        residual |= (actual ^ expected) << (j * 2)
    return residual.reshape(-1, 64)


def q6_sparse_encode(blocks, residual=None):
    if residual is None:
        residual = q6_residual(blocks)
    lanes = ((residual[:, :, None] >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3).reshape(-1)
    nz = np.flatnonzero(lanes)
    bitmap = np.packbits((lanes != 0).astype(np.uint8), bitorder="little").tobytes()
    values = pack2(lanes[nz])
    return bitmap, values, int(nz.size), lanes.size


def q6_sparse_decode(blocks, bitmap, values, lane_count):
    lanes = np.zeros(lane_count, dtype=np.uint8)
    mask = np.unpackbits(np.frombuffer(bitmap, dtype=np.uint8), bitorder="little")[:lane_count].astype(bool)
    lanes[mask] = unpack2(values, int(mask.sum()))
    residual = np.sum(lanes.reshape(-1, 4).astype(np.uint16) << np.array([0, 2, 4, 6], dtype=np.uint16), axis=1).astype(np.uint8).reshape(-1, 64)
    ql = blocks[:, :128].reshape(-1, 2, 64)
    qh = blocks[:, 128:192].reshape(-1, 2, 32).copy()
    for j in range(4):
        lo = ql[:, :, :32] if j in (0, 2) else ql[:, :, 32:]
        nibble = (lo >> (4 if j >= 2 else 0)) & 15
        expected = np.where(nibble >= 8, 3, 0).astype(np.uint8)
        predpart = expected ^ 2
        extra = (residual.reshape(-1, 2, 32) >> (j * 2)) & 3
        qh |= ((predpart ^ extra) << (j * 2)).astype(np.uint8)
    out = blocks.copy()
    out[:, 128:192] = qh.reshape(-1, 64)
    return out


def stream_entropy(values):
    hist = np.bincount(np.asarray(values, dtype=np.uint8).reshape(-1), minlength=256)
    return entropy_from_hist(hist)


def entropy_from_hist(hist):
    active = hist[hist != 0].astype(np.float64)
    p = active / active.sum()
    return float(-(p * np.log2(p)).sum())


def decode_q4_scale_min(packed):
    scales = np.empty((packed.shape[0], 8), dtype=np.uint8)
    mins = np.empty((packed.shape[0], 8), dtype=np.uint8)
    for j in range(4):
        scales[:, j] = packed[:, j] & 63
        mins[:, j] = packed[:, j + 4] & 63
        scales[:, j + 4] = (packed[:, j + 8] & 15) | ((packed[:, j] >> 6) << 4)
        mins[:, j + 4] = (packed[:, j + 8] >> 4) | ((packed[:, j + 4] >> 6) << 4)
    return scales, mins


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    directory = json.loads((ROOT / "directory.json").read_text(encoding="utf-8"))
    with MODEL.open("rb") as f:
        magic, header_length, artifact_id = struct.unpack("<8sQ16s", f.read(32))
        if magic != b"NINFER\0\3":
            raise ValueError("unexpected NInfer model magic")
        parsed = json.loads(f.read(header_length))
        payload_start = (32 + header_length + 4095) // 4096 * 4096
        if parsed != directory:
            raise ValueError("directory header does not match model")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        bindings = directory["bindings"]
        output_id = bindings["text/output_head"]["object"]
        tensors = []
        for name in TENSORS:
            item = next(t for t in directory["objects"] if t["id"] == name)
            stride = 144 if item["format"] == "gguf_q4_k" else 210
            count = item["bytes"] // stride
            start = payload_start + item["offset"]
            raw_view = memoryview(mm)[start:start + item["bytes"]]
            raw_hash = hashlib.sha256(raw_view).hexdigest()
            blocks = np.frombuffer(raw_view, dtype=np.uint8).reshape(count, stride)
            row = {"id": name, "format": item["format"], "blocks": count, "raw_bytes": item["bytes"], "sha256": raw_hash, "output_head_binding": name == output_id}
            if stride == 144:
                words = blocks[:, :4].copy().view("<u2").reshape(-1, 2)
                exp_equal = (words[:, 0] & 0x7c00) == (words[:, 1] & 0x7c00)
                flags, compact_payload, fallback_payload, _ = q4_metadata_encode(blocks)
                roundtrip_head = q4_metadata_decode(flags, compact_payload, fallback_payload, count)
                head_exact = roundtrip_head == blocks[:, :4].tobytes()
                decoded_scales, decoded_mins = decode_q4_scale_min(blocks[:, 4:16])
                fields = {
                    "header_shared_exponent": {"eligible_pairs": int(exp_equal.sum()), "pairs": int(exp_equal.size), "eligible_fraction": float(exp_equal.mean()), "raw_head_bytes": count * 4, "flag_bytes": len(flags), "compact_27bit_stream_bytes": len(compact_payload), "raw_fallback_bytes": len(fallback_payload), "total_with_flags_bytes": len(flags) + len(compact_payload) + len(fallback_payload) + 8, "format_header_bytes": 8, "total_fraction_of_head": (len(flags) + len(compact_payload) + len(fallback_payload) + 8) / (count * 4), "roundtrip_exact": head_exact},
                    "scales_12byte_entropy_bits": stream_entropy(blocks[:, 4:16]),
                    "decoded_scale_min_6bit_fields": {"scale_fields": int(decoded_scales.size), "scale_max_63_count": int((decoded_scales == 63).sum()), "scale_max_63_fraction": float((decoded_scales == 63).mean()), "min_fields": int(decoded_mins.size), "min_max_63_count": int((decoded_mins == 63).sum()), "min_max_63_fraction": float((decoded_mins == 63).mean()), "six_bit_packed_bytes_per_block": 12, "current_packed_bytes_per_block": 12, "max63_elision_without_extra_position_bits_saves_bytes": 0},
                    "scale_unique_pairs_in_first_256_blocks": int(np.unique(blocks[:min(count, 256), 4:16].reshape(-1, 2), axis=0).shape[0]),
                    "qdata_entropy_bits": stream_entropy(blocks[:, 16:144]),
                    "qdata_zero_fraction": float((blocks[:, 16:144] == 0).mean())
                }
                row["fields"] = fields
            else:
                total_lanes = 0
                total_nonzero = 0
                total_bitmap_bytes = 0
                total_value_bytes = 0
                roundtrip_hash = hashlib.sha256()
                quant_code_hist = np.zeros(64, dtype=np.uint64)
                tile_nonzero_counts = {tile_blocks: 0 for tile_blocks in TILE_BLOCKS}
                tile_storage_stats = {tile_blocks: {"stored_full_tile_bytes": 0, "candidate_tiles_used": 0, "tiles": 0, "bitmap_bytes_used": 0, "packed_value_bytes_used": 0} for tile_blocks in TILE_BLOCKS}
                q6_roundtrip_tile = 8192
                for offset in range(0, count, q6_roundtrip_tile):
                    tile = blocks[offset:offset + q6_roundtrip_tile]
                    residual = q6_residual(tile)
                    ql = tile[:, :128].reshape(-1, 2, 64)
                    qh = tile[:, 128:192].reshape(-1, 2, 32)
                    for j in range(4):
                        lo = ql[:, :, :32] if j in (0, 2) else ql[:, :, 32:]
                        nibble = (lo >> (4 if j >= 2 else 0)) & 15
                        high = ((qh >> (j * 2)) & 3) ^ 2
                        quant_code_hist += np.bincount((nibble | (high << 4)).reshape(-1), minlength=64).astype(np.uint64)
                    lane_values = ((residual[:, :, None] >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3).reshape(-1)
                    per_block = (lane_values != 0).reshape(-1, 256).sum(axis=1, dtype=np.uint16)
                    nonzero = int((lane_values != 0).sum())
                    lanes = lane_values.size
                    bitmap_bytes = (lanes + 7) // 8
                    value_bytes = (nonzero * 2 + 7) // 8
                    candidate_bytes = 146 * len(tile) + bitmap_bytes + value_bytes + 8
                    raw_tile_bytes = len(tile) * stride
                    if candidate_bytes < raw_tile_bytes:
                        bitmap, vals, _, _ = q6_sparse_encode(tile, residual)
                        roundtrip = q6_sparse_decode(tile, bitmap, vals, lanes)
                        roundtrip_bytes = roundtrip.tobytes()
                        total_bitmap_bytes += len(bitmap)
                        total_value_bytes += len(vals)
                    else:
                        roundtrip_bytes = tile.tobytes()
                        total_bitmap_bytes += bitmap_bytes
                        total_value_bytes += value_bytes
                    roundtrip_hash.update(roundtrip_bytes)
                    total_lanes += lanes
                    total_nonzero += nonzero
                    for tile_blocks in TILE_BLOCKS:
                        grouped = per_block[:len(per_block) // tile_blocks * tile_blocks].reshape(-1, tile_blocks)
                        nonzero_by_tile = grouped.sum(axis=1, dtype=np.uint64)
                        tile_nonzero_counts[tile_blocks] += int(nonzero_by_tile.sum())
                        packed_by_tile = (nonzero_by_tile + 3) // 4
                        candidate_by_tile = 178 * tile_blocks + packed_by_tile + 8
                        raw_by_tile = stride * tile_blocks + 8
                        use_candidate = candidate_by_tile < raw_by_tile
                        stats = tile_storage_stats[tile_blocks]
                        stats["stored_full_tile_bytes"] += int(np.where(use_candidate, candidate_by_tile, raw_by_tile).sum())
                        stats["candidate_tiles_used"] += int(use_candidate.sum())
                        stats["tiles"] += int(len(nonzero_by_tile))
                        stats["bitmap_bytes_used"] += int(tile_blocks * 32 * use_candidate.sum())
                        stats["packed_value_bytes_used"] += int(packed_by_tile[use_candidate].sum())
                raw_qh = blocks[:, 128:192].size
                encoded_qh = total_bitmap_bytes + total_value_bytes
                scale = blocks[:, 192:208].view(np.int8)
                row["fields"] = {
                    "sign_residual_bitmap": {"nonzero_2bit_lanes": total_nonzero, "lanes": total_lanes, "nonzero_fraction": total_nonzero / total_lanes, "raw_qh_bytes": raw_qh, "bitmap_bytes": total_bitmap_bytes, "packed_residual_bytes": total_value_bytes, "encoded_qh_bytes": encoded_qh, "fraction_of_raw_qh": encoded_qh / raw_qh, "roundtrip_exact": roundtrip_hash.hexdigest() == raw_hash},
                    "qh_entropy_bits": stream_entropy(blocks[:, 128:192]),
                    "qh_zero_fraction": float((blocks[:, 128:192] == 0).mean()),
                    "scale_entropy_bits": stream_entropy(blocks[:, 192:208]),
                    "scale_zero_fraction": float((scale == 0).mean()),
                    "scale_abs_max": int(np.abs(scale.astype(np.int16)).max()),
                    "d_exponent_histogram": np.bincount(((blocks[:, 208:210].copy().view("<u2").reshape(-1) >> 10) & 31), minlength=32).tolist(),
                    "quant_code_entropy_bits_per_symbol": entropy_from_hist(quant_code_hist),
                    "quant_code_histogram_64": quant_code_hist.tolist(),
                    "quant_code_count": int(quant_code_hist.sum())
                }
            if stride == 144:
                for tile_blocks in TILE_BLOCKS:
                    complete = count // tile_blocks
                    overhead = complete * 8
                    row.setdefault("tile_estimates", []).append({"blocks_per_tile": tile_blocks, "complete_tiles": complete, "tail_raw_bytes": (count - complete * tile_blocks) * stride, "tile_index_and_alignment_bytes": overhead, "raw_tiled_bytes": item["bytes"] + overhead})
            else:
                for tile_blocks in TILE_BLOCKS:
                    complete = count // tile_blocks
                    stats = tile_storage_stats[tile_blocks]
                    tail = (count - complete * tile_blocks) * stride
                    indexed_bytes = stats["stored_full_tile_bytes"]
                    row.setdefault("tile_estimates", []).append({"blocks_per_tile": tile_blocks, "tiles": stats["tiles"], "candidate_tiles_used": stats["candidate_tiles_used"], "tail_raw_bytes": tail, "indexed_bytes": indexed_bytes + tail, "ratio": (indexed_bytes + tail) / item["bytes"], "header_offset_padding_bytes": stats["tiles"] * 8, "bitmap_bytes": stats["bitmap_bytes_used"], "packed_nonzero_value_bytes": stats["packed_value_bytes_used"]})
            tensors.append(row)
            if stride == 210:
                del scale, tile, ql, qh
            del blocks, raw_view
        del mm
    result = {"model": str(MODEL), "artifact_id": artifact_id.hex(), "payload_offset": payload_start, "method": "CPU-only full tensor structured field study; mmap source reads; no GPU; no general-purpose compressor sweep", "tensors": tensors}
    attach_short_stream_estimates(result)
    result["all_full_tensor_roundtrips_exact"] = all(t.get("fields", {}).get("header_shared_exponent", {}).get("roundtrip_exact", t.get("fields", {}).get("sign_residual_bitmap", {}).get("roundtrip_exact", False)) for t in tensors)
    out = ROOT / "results" / "fields-codecs.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "tensors": len(tensors), "exact": all(t.get("fields", {}).get("header_shared_exponent", {}).get("roundtrip_exact", t.get("fields", {}).get("sign_residual_bitmap", {}).get("roundtrip_exact", False)) for t in tensors), "seconds": time.time() - started}, separators=(",", ":")))


if __name__ == "__main__":
    started = time.time()
    main()
