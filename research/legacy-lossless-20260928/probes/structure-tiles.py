import bz2
import csv
import json
import mmap
import pathlib
import struct
import time
import zlib

import numpy as np


ROOT = pathlib.Path(r"C:\AI\tuning\ninfer-lossless-20260928")
MODEL = pathlib.Path(r"C:\AI\staging\qwen38-jonathan-ninfer\qwen38-jonathan-q4km-proposal.ninfer")
TARGETS = {"weight/000001", "weight/000060", "weight/000746", "weight/000131"}
REGION_BYTES = 1 << 20
TILE_BLOCKS = (16, 64, 256, 1024, 4096)
MAX_TILES_PER_REGION = 8


def load_header(f):
    magic, header_length, artifact_id = struct.unpack("<8sQ16s", f.read(32))
    if magic != b"NINFER\0\3":
        raise ValueError(f"unexpected magic {magic!r}")
    directory = json.loads(f.read(header_length))
    payload_offset = (32 + header_length + 4095) // 4096 * 4096
    return directory, payload_offset, artifact_id.hex(), header_length


def encode(raw, codec):
    start = time.perf_counter()
    if codec == "zlib1":
        encoded = zlib.compress(raw, 1)
    elif codec == "zlib6":
        encoded = zlib.compress(raw, 6)
    else:
        encoded = bz2.compress(raw, compresslevel=1)
    return len(encoded), time.perf_counter() - start


def column_delta(a, xor=False):
    if xor:
        return np.bitwise_xor(a, np.concatenate((a[:1], a[:-1]), axis=0))
    return np.diff(a.astype(np.int16), axis=0, prepend=a[:1].astype(np.int16)).astype(np.uint8)


def q4_nibble_layouts(blocks):
    qs = blocks[:, 16:144].reshape(-1, 4, 32)
    low = qs & 15
    high = qs >> 4
    group = np.stack((low, high), axis=2)
    return {
        "nibble_per_block": group.tobytes(),
        "nibble_group_major": group.transpose(1, 2, 0, 3).copy().tobytes(),
        "nibble_position_major": group.transpose(1, 2, 3, 0).copy().tobytes(),
    }, group


def groups_q4_scales(blocks):
    s = blocks[:, 4:16].astype(np.uint8)
    sc = np.empty((len(s), 8), dtype=np.uint8)
    mn = np.empty((len(s), 8), dtype=np.uint8)
    for j in range(8):
        if j < 4:
            sc[:, j] = s[:, j] & 63
            mn[:, j] = s[:, j + 4] & 63
        else:
            sc[:, j] = (s[:, j + 4] & 15) | ((s[:, j - 4] >> 6) << 4)
            mn[:, j] = (s[:, j + 4] >> 4) | ((s[:, j] >> 6) << 4)
    return sc, mn


def entropy_from_counts(c):
    p = c[c > 0].astype(np.float64) / c.sum()
    return float(-(p * np.log2(p)).sum())


def conditional_entropy(values, conditions, alph_size, cond_size):
    joint = np.bincount(conditions.astype(np.int64).reshape(-1) * alph_size + values.astype(np.int64).reshape(-1), minlength=cond_size * alph_size).reshape(cond_size, alph_size)
    totals = joint.sum(axis=1)
    nonzero = totals > 0
    result = 0.0
    total = totals.sum()
    for row in joint[nonzero]:
        result += row.sum() / total * entropy_from_counts(row)
    return result


def measure_tensor(mm, item, payload_offset):
    q4 = item["format"] == "gguf_q4_k"
    stride = 144 if q4 else 210
    blocks_total = item["bytes"] // stride
    bytes_per_region = min(REGION_BYTES // stride * stride, item["bytes"] // 4 // stride * stride)
    start_blocks = [int((blocks_total - bytes_per_region // stride) * frac) for frac in (0, 1 / 3, 2 / 3, 1.0)]
    regions = [np.frombuffer(mm, dtype=np.uint8, offset=payload_offset + item["offset"] + start * stride, count=bytes_per_region).copy().reshape(-1, stride) for start in start_blocks]
    output_rows = []
    tile_detail = []
    for tile_size in TILE_BLOCKS:
        aggregate = {}
        aggregate_time = {}
        aggregate_input_bytes = 0
        for region in regions:
            total_tiles = (len(region) + tile_size - 1) // tile_size
            selected_tiles = set(np.linspace(0, total_tiles - 1, min(total_tiles, MAX_TILES_PER_REGION), dtype=np.int64).tolist())
            for tile_index, first in enumerate(range(0, len(region), tile_size)):
                if tile_index not in selected_tiles:
                    continue
                tile = region[first:first + tile_size]
                raw = tile.tobytes()
                aggregate_input_bytes += len(raw)
                candidates = {"raw": [(raw, len(raw))]}
                if q4:
                    layouts, qgroups = q4_nibble_layouts(tile)
                    header = tile[:, :4]
                    scales = tile[:, 4:16]
                    qs = tile[:, 16:144]
                    candidates["block_byte_shuffle"] = [(tile.T.copy().tobytes(), len(raw))]
                    candidates["fields_all_reordered"] = [(np.concatenate((header.reshape(-1), scales.reshape(-1), qs.reshape(-1))).tobytes(), len(raw))]
                    candidates["headers_scales_qs_separate"] = [(header.T.copy().tobytes(), 4 * len(tile)), (scales.T.copy().tobytes(), 12 * len(tile)), (qs.T.copy().tobytes(), 128 * len(tile))]
                    candidates["header_scale_delta_qs_raw"] = [(column_delta(np.concatenate((header, scales), axis=1), True).T.copy().tobytes(), 16 * len(tile)), (qs.tobytes(), 128 * len(tile))]
                    candidates["header_scale_delta_qs_nibbles_group"] = [(column_delta(np.concatenate((header, scales), axis=1), True).T.copy().tobytes(), 16 * len(tile)), (qgroups.transpose(1, 2, 0, 3).copy().tobytes(), 128 * len(tile))]
                    candidates["metadata_qs_nibbles"] = [(header.tobytes(), 4 * len(tile)), (scales.tobytes(), 12 * len(tile)), (qgroups.tobytes(), 128 * len(tile))]
                    candidates["metadata_qs_nibbles_groups_separate"] = [(header.tobytes(), 4 * len(tile)), (scales.tobytes(), 12 * len(tile))] + [(qgroups[:, group, lane, :].tobytes(), 16 * len(tile)) for group in range(4) for lane in range(2)]
                else:
                    ql, qh, scales, d = tile[:, :128], tile[:, 128:192], tile[:, 192:208], tile[:, 208:210]
                    candidates["block_byte_shuffle"] = [(tile.T.copy().tobytes(), len(raw))]
                    candidates["fields_all_reordered"] = [(np.concatenate((ql.reshape(-1), qh.reshape(-1), scales.reshape(-1), d.reshape(-1))).tobytes(), len(raw))]
                    candidates["fields_separate"] = [(ql.tobytes(), 128 * len(tile)), (qh.tobytes(), 64 * len(tile)), (scales.tobytes(), 16 * len(tile)), (d.tobytes(), 2 * len(tile))]
                for candidate, streams in candidates.items():
                    for codec in ("zlib1", "zlib6", "bz2_1"):
                        outbytes = 0
                        duration = 0.0
                        for stream, represented in streams:
                            n, sec = encode(stream, codec)
                            outbytes += n
                            duration += sec
                        key = (candidate, codec)
                        aggregate[key] = aggregate.get(key, 0) + outbytes + len(raw) - sum(n for _, n in streams)
                        aggregate_time[key] = aggregate_time.get(key, 0.0) + duration
        sample_bytes = aggregate_input_bytes
        for (candidate, codec), total in aggregate.items():
            output_rows.append({"id": item["id"], "format": item["format"], "sample_bytes": sample_bytes, "independent_tile_blocks": tile_size, "candidate": candidate, "codec": codec, "effective_bytes_including_raw_residual": total, "ratio_vs_original": round(total / sample_bytes, 7), "aggregate_encode_seconds": round(aggregate_time[(candidate, codec)], 4)})
        best = min((r for r in output_rows if r["id"] == item["id"] and r["independent_tile_blocks"] == tile_size), key=lambda r: r["ratio_vs_original"])
        tile_detail.append({"tile_blocks": tile_size, "original_sample_bytes": sample_bytes, "best": best})
    info = {}
    if q4:
        sample = np.concatenate(regions)
        nibble_layout, group = q4_nibble_layouts(sample)
        flat = group.reshape(-1)
        global_counts = np.bincount(flat, minlength=16)
        position_entropies = [entropy_from_counts(np.bincount(group[:, g, lane, :].reshape(-1), minlength=16)) for g in range(4) for lane in range(2)]
        scale_codes, min_codes = groups_q4_scales(sample)
        condition = np.repeat(scale_codes[:, :, None], 32, axis=2).reshape(-1)
        qflat = group.reshape(-1)
        scale_h = conditional_entropy(qflat, condition, 16, 64)
        info = {"q4_global_nibble_entropy_bits": entropy_from_counts(global_counts), "q4_position_conditional_entropy_bits": float(np.mean(position_entropies)), "q4_entropy_per_subblock_bits": position_entropies, "q4_nibble_entropy_conditioned_on_decoded_scale_code_bits": scale_h, "q4_scale_code_entropy_bits": float(np.mean([entropy_from_counts(np.bincount(scale_codes[:, g], minlength=64)) for g in range(8)])), "q4_min_code_entropy_bits": float(np.mean([entropy_from_counts(np.bincount(min_codes[:, g], minlength=64)) for g in range(8)]))}
        q_mi = {}
        vals = flat.astype(np.int64)
        for lag in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            a = vals[:-lag]
            b = vals[lag:]
            joint = np.bincount(a * 16 + b, minlength=256).reshape(16, 16).astype(np.float64)
            joint /= joint.sum()
            pa = joint.sum(1)
            pb = joint.sum(0)
            i, j = np.nonzero(joint)
            q_mi[str(lag)] = float(np.sum(joint[i, j] * np.log2(joint[i, j] / (pa[i] * pb[j]))))
        info["q4_linear_nibble_mutual_information_by_lag_bits"] = q_mi
    return output_rows, {"id": item["id"], "format": item["format"], "sample_region_relative_block_starts": start_blocks, "sample_region_blocks": bytes_per_region // stride, "independent_tile_results": tile_detail, "conditional_information": info}


def main():
    directory = json.loads((ROOT / "directory.json").read_text(encoding="utf-8"))
    previous = json.loads((ROOT / "results" / "analysis.json").read_text(encoding="utf-8"))
    with MODEL.open("rb") as f:
        parsed, payload_offset, artifact_id, header_length = load_header(f)
        if parsed != directory or payload_offset != previous["payload_offset"] or artifact_id != previous["artifact_id"]:
            raise ValueError("model header does not agree with supplied directory/analysis files")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        rows = []
        summaries = []
        for item in directory["objects"]:
            if item["id"] in TARGETS:
                result_rows, summary = measure_tensor(mm, item, payload_offset)
                rows.extend(result_rows)
                summaries.append(summary)
                print(json.dumps({"id": summary["id"], "best": [x["best"] for x in summary["independent_tile_results"]], "conditional": summary["conditional_information"]}, separators=(",", ":")))
        mm.close()
    result = {"model": str(MODEL), "artifact_id": artifact_id, "header_bytes": header_length, "payload_offset_derived_from_header": payload_offset, "region_bytes_per_quadrant": REGION_BYTES, "tile_block_counts": TILE_BLOCKS, "max_tiles_per_region_per_size": MAX_TILES_PER_REGION, "compression_is_independent_per_tile_and_stream": True, "codec_names": ["zlib1", "zlib6", "bz2_1"], "tensors": summaries, "candidate_rows": rows}
    (ROOT / "structure-tiles.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (ROOT / "structure-tiles.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "format", "sample_bytes", "independent_tile_blocks", "candidate", "codec", "effective_bytes_including_raw_residual", "ratio_vs_original", "aggregate_encode_seconds"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
