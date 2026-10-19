import hashlib
import json
import mmap
import pathlib
import struct
import time

import numpy as np


ROOT = pathlib.Path("/mnt/c/AI/tuning/ninfer-lossless-20260928")
MODEL = pathlib.Path("/mnt/c/AI/staging/qwen38-jonathan-ninfer/qwen38-jonathan-q4km-proposal.ninfer")
TARGETS = ("weight/000000", "weight/000131")
CHUNK_BLOCKS = 32768


def transpose_qh(qh):
    cells = qh.reshape(-1, 16, 4)
    out = np.empty_like(cells)
    for q in range(4):
        out[:, :, q] = np.sum(((cells >> (2 * q)) & 3) << (2 * np.arange(4, dtype=np.uint8)), axis=2, dtype=np.uint16).astype(np.uint8)
    return out.reshape(-1, 64)


def inverse_qh(packed):
    cells = packed.reshape(-1, 16, 4)
    out = np.empty_like(cells)
    for j in range(4):
        value = np.zeros(cells.shape[:2], dtype=np.uint8)
        for q in range(4):
            value |= (((cells[:, :, q] >> (2 * j)) & 3) << (2 * q)).astype(np.uint8)
        out[:, :, j] = value
    return out.reshape(-1, 64)


def signed_high_lut():
    signed = np.array([0xE0, 0xF0, 0x00, 0x10], dtype=np.uint8)
    table = np.zeros(256, dtype=np.uint32)
    values = np.arange(256, dtype=np.uint16)
    for j in range(4):
        field = (values >> (2 * j)) & 3
        table |= signed[field].astype(np.uint32) << (8 * j)
    return table


def prove_cells():
    patterns = np.empty((4 * 256, 4), dtype=np.uint8)
    row = 0
    for position in range(4):
        for value in range(256):
            patterns[row] = 0
            patterns[row, position] = value
            row += 1
    rng = np.random.default_rng(20260929)
    random_patterns = rng.integers(0, 256, size=(1000000, 4), dtype=np.uint8)
    data = np.concatenate((patterns, random_patterns), axis=0)
    source = np.zeros((len(data), 64), dtype=np.uint8)
    source[:, :4] = data
    encoded = transpose_qh(source)
    restored = inverse_qh(encoded)
    if not np.array_equal(restored, source):
        raise ValueError("QH transpose inverse fixture failure")
    lut = signed_high_lut()
    input_cells = data[:, :, None]
    direct = ((input_cells >> (2 * np.arange(4, dtype=np.uint8))) & 3).astype(np.uint8)
    mapped = np.empty_like(direct)
    encoded_cells = encoded[:, :4]
    for q in range(4):
        mapped[:, :, q] = ((lut[encoded_cells[:, q], None] >> (8 * np.arange(4, dtype=np.uint32))) & 255).astype(np.uint8)
    if not np.array_equal(mapped, np.array([0xE0, 0xF0, 0x00, 0x10], dtype=np.uint8)[direct]):
        raise ValueError("QH signed-high LUT fixture failure")
    low = np.arange(16, dtype=np.uint8)
    high = np.arange(4, dtype=np.uint8)
    direct_values = ((((high.astype(np.int16) << 4)[:, None]) + low.astype(np.int16)[None, :] - 32) & 255).astype(np.uint8)
    lut_values = np.array([0xE0, 0xF0, 0x00, 0x10], dtype=np.uint8)[high]
    if not np.array_equal(lut_values[:, None] | low[None, :], direct_values):
        raise ValueError("QH signed-high OR-low nibble fixture failure")
    return {"exhaustive_single_byte_patterns": int(len(patterns)), "random_four_byte_groups": int(len(random_patterns)), "roundtrip_groups": int(len(data)), "signed_high_field_values": ["0xE0", "0xF0", "0x00", "0x10"], "all_low_nibble_combinations_exact": 64}


def main():
    started = time.perf_counter()
    directory = json.loads((ROOT / "directory.json").read_text(encoding="utf-8"))
    with MODEL.open("rb") as f:
        magic, header_length, artifact_id = struct.unpack("<8sQ16s", f.read(32))
        if magic != b"NINFER\0\3":
            raise ValueError("NInfer model magic mismatch")
        header = json.loads(f.read(header_length))
        payload_start = (32 + header_length + 4095) // 4096 * 4096
        if header != directory:
            raise ValueError("NInfer model directory mismatch")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        fixture_proof = prove_cells()
        tensors = []
        lut = signed_high_lut()
        for name in TARGETS:
            item = next(obj for obj in directory["objects"] if obj["id"] == name)
            if item["format"] != "gguf_q6_k" or item["bytes"] % 210:
                raise ValueError("unexpected Q6 tensor format or byte count")
            blocks = item["bytes"] // 210
            source_hash = hashlib.sha256()
            encoded_hash = hashlib.sha256()
            restored_hash = hashlib.sha256()
            checked = 0
            for block_start in range(0, blocks, CHUNK_BLOCKS):
                block_count = min(CHUNK_BLOCKS, blocks - block_start)
                begin = payload_start + item["offset"] + block_start * 210
                end = begin + block_count * 210
                src = np.frombuffer(mm, dtype=np.uint8, count=block_count * 210, offset=begin).reshape(block_count, 210)
                source_hash.update(memoryview(src))
                transformed = src.copy()
                qh = src[:, 128:192]
                coded = transpose_qh(qh)
                transformed[:, 128:192] = coded
                encoded_hash.update(memoryview(transformed))
                decoded = inverse_qh(coded)
                if not np.array_equal(decoded, qh):
                    raise ValueError(f"full tensor QH inverse mismatch: {name} block {block_start}")
                transformed[:, 128:192] = decoded
                restored_hash.update(memoryview(transformed))
                checked += block_count
            src_digest = source_hash.hexdigest()
            restored_digest = restored_hash.hexdigest()
            if src_digest != restored_digest:
                raise ValueError(f"full tensor byte hash mismatch: {name}")
            tensors.append({"id": name, "shape": item["shape"], "format": item["format"], "source_offset": item["offset"], "source_bytes": item["bytes"], "blocks": blocks, "blocks_checked": checked, "source_sha256": src_digest, "encoded_layout_sha256": encoded_hash.hexdigest(), "decoded_original_sha256": restored_digest, "decoded_equals_source": True, "resident_bytes_after_transform": item["bytes"], "resident_bytes_saved": 0, "qh_source_range_in_210B_block": [128, 192], "qh_transform": "For each consecutive four QH bytes, transpose their four 2-bit cells into four bytes, one 2-bit bitplane per output byte; preserve QL[0:128], d/scales, and all other block bytes unchanged."})
        del src, transformed, qh, coded, decoded
        mm.close()
    result = {"method": "CPU-only exact bijective Q6_K QH four-byte group transpose proof using NInfer header and read-only mmap payload", "model": str(MODEL), "artifact_id": artifact_id.hex(), "payload_offset": payload_start, "layout_bytes_per_block": 210, "qh_bytes_per_block": 64, "codec_overhead_bytes_per_block": 0, "resident_size_change": 0, "device_lut_bytes_if_computed_once_per_CTA": 1024, "device_lut_is_resident_model_data": False, "fixtures": fixture_proof, "tensors": tensors, "elapsed_seconds": time.perf_counter() - started}
    output = ROOT / "results" / "q6-high-group-codec.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(output), "artifact_id": result["artifact_id"], "elapsed_seconds": result["elapsed_seconds"], "tensors": [{"id": t["id"], "blocks": t["blocks"], "bytes": t["source_bytes"], "exact": t["decoded_equals_source"]} for t in tensors]}, indent=2))


if __name__ == "__main__":
    main()
