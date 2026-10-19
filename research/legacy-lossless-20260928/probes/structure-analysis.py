import bz2
import hashlib
import json
import lzma
import mmap
import pathlib
import struct
import time
import zlib

import numpy as np


ROOT = pathlib.Path(r"C:\AI\tuning\ninfer-lossless-20260928")
MODEL = pathlib.Path(r"C:\AI\staging\qwen38-jonathan-ninfer\qwen38-jonathan-q4km-proposal.ninfer")
TARGETS = {"weight/000001", "weight/000060", "weight/000746", "weight/000131"}
SAMPLE_BYTES_PER_REGION = 1 << 20


def entropy(values, alphabet):
    counts = np.bincount(values.reshape(-1), minlength=alphabet).astype(np.float64)
    probs = counts[counts > 0] / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def mutual_information(a, b, alphabet):
    joint = np.bincount(a.reshape(-1).astype(np.int64) * alphabet + b.reshape(-1).astype(np.int64), minlength=alphabet * alphabet).reshape(alphabet, alphabet).astype(np.float64)
    joint /= joint.sum()
    pa = joint.sum(axis=1)
    pb = joint.sum(axis=0)
    nz = joint > 0
    ii, jj = np.nonzero(nz)
    return float(np.sum(joint[ii, jj] * np.log2(joint[ii, jj] / (pa[ii] * pb[jj]))))


def compressions(raw):
    out = {}
    for name, func in (("zlib1", lambda x: zlib.compress(x, 1)), ("zlib6", lambda x: zlib.compress(x, 6)), ("bz2_1", lambda x: bz2.compress(x, compresslevel=1)), ("lzma0", lambda x: lzma.compress(x, preset=0))):
        start = time.perf_counter()
        encoded = func(raw)
        out[name] = {"bytes": len(encoded), "ratio": len(encoded) / len(raw), "seconds": round(time.perf_counter() - start, 4)}
    return out


def q4_transforms(a):
    b = a.reshape(-1, 144)
    qs = b[:, 16:144]
    return {
        "block_byte_shuffle": b.T.copy().tobytes(),
        "fields_headers_scales_qs": np.concatenate((b[:, :4].reshape(-1), b[:, 4:16].reshape(-1), qs.reshape(-1))).tobytes(),
        "nibble_lanes_low_then_high": np.stack((qs & 15, qs >> 4), axis=1).reshape(-1).tobytes(),
        "nibble_lanes_high_then_low": np.stack((qs >> 4, qs & 15), axis=1).reshape(-1).tobytes(),
        "qs_block_byte_shuffle": qs.T.copy().tobytes(),
        "qs_adjacent_delta": np.diff(qs, axis=0, prepend=qs[:1]).astype(np.uint8).tobytes(),
        "qs_adjacent_xor": np.bitwise_xor(qs, np.concatenate((qs[:1], qs[:-1]), axis=0)).tobytes(),
    }


def q6_transforms(a):
    b = a.reshape(-1, 210)
    return {
        "block_byte_shuffle": b.T.copy().tobytes(),
        "fields_ql_qh_scales_d": np.concatenate((b[:, :128].reshape(-1), b[:, 128:192].reshape(-1), b[:, 192:208].reshape(-1), b[:, 208:210].reshape(-1))).tobytes(),
        "ql_byte_shuffle": b[:, :128].T.copy().tobytes(),
        "qh_byte_shuffle": b[:, 128:192].T.copy().tobytes(),
        "scale_byte_shuffle": b[:, 192:208].T.copy().tobytes(),
        "ql_adjacent_delta": np.diff(b[:, :128], axis=0, prepend=b[:1, :128]).astype(np.uint8).tobytes(),
        "ql_adjacent_xor": np.bitwise_xor(b[:, :128], np.concatenate((b[:1, :128], b[:-1, :128]), axis=0)).tobytes(),
        "qh_adjacent_delta": np.diff(b[:, 128:192], axis=0, prepend=b[:1, 128:192]).astype(np.uint8).tobytes(),
        "qh_adjacent_xor": np.bitwise_xor(b[:, 128:192], np.concatenate((b[:1, 128:192], b[:-1, 128:192]), axis=0)).tobytes(),
    }


def load_header(f):
    raw = f.read(32)
    magic, header_bytes, artifact_id = struct.unpack("<8sQ16s", raw)
    if magic != b"NINFER\0\3":
        raise ValueError(f"unexpected magic {magic!r}")
    directory_bytes = f.read(header_bytes)
    directory = json.loads(directory_bytes)
    payload_offset = ((32 + header_bytes + 4095) // 4096) * 4096
    return directory, payload_offset, artifact_id.hex(), header_bytes


def tensor_samples(mm, offset, nbytes, stride):
    blocks = nbytes // stride
    each = min(SAMPLE_BYTES_PER_REGION // stride * stride, nbytes // 4 // stride * stride)
    starts = [int((blocks - each // stride) * frac) * stride for frac in (0.0, 1 / 3, 2 / 3, 1.0)]
    pieces = []
    for start in starts:
        pieces.append(np.frombuffer(mm, dtype=np.uint8, offset=offset + start, count=each).copy())
    return np.concatenate(pieces), starts, each


def main():
    directory_path = ROOT / "directory.json"
    analysis_path = ROOT / "results" / "analysis.json"
    directory = json.loads(directory_path.read_text(encoding="utf-8"))
    previous = json.loads(analysis_path.read_text(encoding="utf-8"))
    with MODEL.open("rb") as f:
        parsed_directory, payload_offset, artifact_id, header_bytes = load_header(f)
        if parsed_directory != directory:
            raise ValueError("saved directory.json does not match model header")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        result = {"model": str(MODEL), "artifact_id": artifact_id, "header_bytes": header_bytes, "payload_offset_derived_from_header": payload_offset, "analysis_payload_offset": previous["payload_offset"], "sample_bytes_per_region": SAMPLE_BYTES_PER_REGION, "sample_regions": 4, "tensors": []}
        if payload_offset != previous["payload_offset"] or artifact_id != previous["artifact_id"]:
            raise ValueError("header-derived start or artifact identity disagrees with analysis.json")
        for tensor in directory["objects"]:
            if tensor["id"] not in TARGETS:
                continue
            stride = 144 if tensor["format"] == "gguf_q4_k" else 210
            absolute_offset = payload_offset + tensor["offset"]
            sample, starts, each = tensor_samples(mm, absolute_offset, tensor["bytes"], stride)
            blocks = sample.reshape(-1, stride)
            transforms = q4_transforms(blocks) if stride == 144 else q6_transforms(blocks)
            compressor_results = {"raw": {"identity": {"bytes": len(sample), "ratio": 1.0, "seconds": 0.0}}}
            compressor_results["raw"].update(compressions(sample.tobytes()))
            block_count = len(blocks)
            coverage = {name: stride for name in ("block_byte_shuffle", "fields_headers_scales_qs", "fields_ql_qh_scales_d") if name in transforms}
            if stride == 144:
                coverage.update({"nibble_lanes_low_then_high": 128, "nibble_lanes_high_then_low": 128, "qs_block_byte_shuffle": 128, "qs_adjacent_delta": 128, "qs_adjacent_xor": 128})
            else:
                coverage.update({"ql_byte_shuffle": 128, "qh_byte_shuffle": 64, "scale_byte_shuffle": 16, "ql_adjacent_delta": 128, "ql_adjacent_xor": 128, "qh_adjacent_delta": 64, "qh_adjacent_xor": 64})
            for name, transformed in transforms.items():
                compressor_results[name] = compressions(transformed)
                for codec in compressor_results[name].values():
                    codec["ratio"] = round(codec["ratio"], 7)
            for name, codecs in compressor_results.items():
                covered = len(sample) if name == "raw" else block_count * coverage[name]
                for codec in codecs.values():
                    codec["estimated_total_bytes_with_raw_untransformed_fields"] = int(codec["bytes"] + len(sample) - covered)
                    codec["estimated_full_tensor_ratio"] = round(codec["estimated_total_bytes_with_raw_untransformed_fields"] / len(sample), 7)
            if stride == 144:
                payload = blocks[:, 16:144]
                values = np.stack((payload & 15, payload >> 4), axis=-1)
                bitplanes = {str(bit): entropy(((values >> bit) & 1).astype(np.uint8), 2) for bit in range(4)}
                adjacent = mutual_information(values[..., 0], values[..., 1], 16)
                qsample = payload
            else:
                ql = blocks[:, :128]
                qh = blocks[:, 128:192]
                scales = blocks[:, 192:208]
                bitplanes = {"ql_bit_" + str(bit): entropy(((ql >> bit) & 1).astype(np.uint8), 2) for bit in range(8)}
                bitplanes.update({"qh_bit_" + str(bit): entropy(((qh >> bit) & 1).astype(np.uint8), 2) for bit in range(8)})
                adjacent = mutual_information(ql[:, :-1], ql[:, 1:], 256)
                qsample = ql
                qgroups = ql.reshape(-1, 2, 64)
                hgroups = qh.reshape(-1, 2, 32)
                low_lanes = []
                high_lanes = []
                q6_codes = []
                for chunk in range(4):
                    qlow = qgroups[:, chunk // 2, :32]
                    qhigh = (hgroups[:, chunk // 2, :] >> (2 * chunk)) & 3
                    qlow = (qlow >> (4 if chunk >= 2 else 0)) & 15
                    low_lanes.append(qlow)
                    high_lanes.append(qhigh)
                    q6_codes.append(qlow | (qhigh << 4))
                q6_codes = np.stack(q6_codes, axis=1)
                q6_low_high_mi = mutual_information(np.stack(low_lanes, axis=1), np.stack(high_lanes, axis=1), 16)
                q6_code_entropy = entropy(q6_codes, 64)
                q6_code_histogram = np.bincount(q6_codes.reshape(-1), minlength=64)
            qflat = qsample.reshape(-1)
            qbyte_mi = {str(lag): round(mutual_information(qflat[:-lag], qflat[lag:], 256), 6) for lag in (1, 2, 4, 8, 16, 32, 64)}
            qdelta = np.diff(qsample.astype(np.int16), axis=0, prepend=qsample[:1].astype(np.int16)).astype(np.int8).view(np.uint8)
            qxor = np.bitwise_xor(qsample, np.concatenate((qsample[:1], qsample[:-1]), axis=0))
            row = {
                "id": tensor["id"], "format": tensor["format"], "shape": tensor["shape"], "roles": next((x.get("roles", []) for x in previous["tensors"] if x["id"] == tensor["id"]), []), "tensor_bytes": tensor["bytes"], "absolute_file_offset": absolute_offset, "sample_region_starts_relative": starts, "sample_region_bytes": each, "sample_bytes": int(sample.nbytes), "sample_sha256": hashlib.sha256(sample).hexdigest(), "compressions": compressor_results, "quant_bitplane_entropy_bits": bitplanes, "quant_lane_or_neighbor_mi_bits": round(adjacent, 6), "quant_byte_mi_by_linear_lag_bits": qbyte_mi, "q_field_delta_and_xor": {"delta": compressions(qdelta.tobytes()), "xor": compressions(qxor.tobytes())}
            }
            if stride == 210:
                row["q6_decoded_code_entropy_bits"] = q6_code_entropy
                row["q6_decoded_code_histogram"] = q6_code_histogram.tolist()
                row["q6_low4_high2_mutual_information_bits"] = round(q6_low_high_mi, 6)
            row["best_codec_by_ratio"] = min(((v["estimated_full_tensor_ratio"], transform, codec) for transform, results in compressor_results.items() for codec, v in results.items()), key=lambda x: x[0])
            result["tensors"].append(row)
            print(json.dumps({"id": row["id"], "sample": row["sample_bytes"], "best": row["best_codec_by_ratio"], "raw_zlib1": compressor_results["raw"]["zlib1"], "shuffle_zlib1": compressor_results.get("block_byte_shuffle", {}).get("zlib1"), "mi": row["quant_lane_or_neighbor_mi_bits"], "byte_mi": qbyte_mi}, separators=(",", ":")))
        mm.close()
    (ROOT / "structure-results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    import csv
    with (ROOT / "structure-results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "format", "sample_bytes", "transform", "codec", "encoded_bytes", "effective_full_sample_bytes", "effective_full_sample_ratio", "codec_seconds"])
        writer.writeheader()
        for tensor in result["tensors"]:
            for transform, codecs in tensor["compressions"].items():
                for codec, measurement in codecs.items():
                    writer.writerow({"id": tensor["id"], "format": tensor["format"], "sample_bytes": tensor["sample_bytes"], "transform": transform, "codec": codec, "encoded_bytes": measurement["bytes"], "effective_full_sample_bytes": measurement["estimated_total_bytes_with_raw_untransformed_fields"], "effective_full_sample_ratio": measurement["estimated_full_tensor_ratio"], "codec_seconds": measurement["seconds"]})


if __name__ == "__main__":
    main()
