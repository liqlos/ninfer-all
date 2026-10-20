import argparse
import hashlib
import json
import math
import pathlib
import struct


K = 5120
PHYSICAL = 248320
LOGICAL = 248077
PACKED = 5760
SAMPLE = 502432
MAGIC = 0x315355534E454348
HEADER = struct.Struct("<Q8IQ")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bf16(bits):
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def bf16_cell(bits):
    require(bits & 0x7F80 != 0x7F80, "Nonfinite BF16 anchor")
    if bits & 0x7FFF == 0:
        return (-2.0 ** -134, 2.0 ** -134)
    if bits & 0x8000:
        low, high = bf16_cell(bits ^ 0x8000)
        return -high, -low
    value = bf16(bits)
    low = (bf16(bits - 1) + value) / 2.0
    high = (value + bf16(bits + 1)) / 2.0 if bits != 0x7F7F else 2.0 ** 128 - 2.0 ** 119
    return low, high


def signed_interval(coefficients, intervals):
    low = 0.0
    high = 0.0
    for coefficient, (a, b) in zip(coefficients, intervals):
        lower = coefficient * (a if coefficient >= 0 else b)
        upper = coefficient * (b if coefficient >= 0 else a)
        low = math.nextafter(low + math.nextafter(lower, -math.inf), -math.inf)
        high = math.nextafter(high + math.nextafter(upper, math.inf), math.inf)
    return low, high


def self_test():
    require(HEADER.size == 48, "Header ABI")
    require(SAMPLE == 32 + PACKED + PHYSICAL * 2, "Sample ABI")
    require(bf16_cell(0x3F80) == (0.998046875, 1.00390625), "Power-of-two cell asymmetry")
    require(bf16_cell(0xBF80) == (-1.00390625, -0.998046875), "Negative cell")
    require(bf16_cell(0) == bf16_cell(0x8000), "Signed zero enclosure")
    low, high = signed_interval([-2.0, 0.5], [(1.0, 2.0), (4.0, 6.0)])
    require(low <= -2.0 and high >= 1.0, "Signed anchor endpoint selection")
    require(-2.0 * 2.0 + 0.5 * 6.0 < high, "Upper endpoints alone are unsafe")
    return {"status": "CPU_FORMAT_AND_CELL_CONTRACT_PASS", "gpu_run": False,
            "native_lowering_proved": False, "performance_claim": False}


def analyze(metadata, binary):
    import numpy as np

    require(metadata["kind"] == "ninfer_lab_head_input_census" and metadata["error"] == 0,
            "Invalid or failed observer record")
    with binary.open("rb") as stream:
        fields = HEADER.unpack(stream.read(HEADER.size))
    magic, schema, epoch, count, k, physical, logical, packed, sample_bytes, payload_bytes = fields
    require((magic, schema, k, physical, logical, packed, sample_bytes) ==
            (MAGIC, 1, K, PHYSICAL, LOGICAL, PACKED, SAMPLE), "Capture format changed")
    require(0 < count <= 16 and epoch == metadata["epoch"] and
            count == len(metadata["records"]), "Epoch or record count mismatch")
    require(payload_bytes == LOGICAL * 4 + count * 7 * SAMPLE and
            binary.stat().st_size == HEADER.size + payload_bytes, "Capture length mismatch")
    dtype = np.dtype([("epoch", "<u4"), ("round", "<u4"), ("role", "<i4"),
        ("frontier", "<i4"), ("anchor", "<i4"), ("extent", "<i4"), ("weight", "<u8"),
        ("packed", "u1", (PACKED,)), ("logits", "<u2", (PHYSICAL,))])
    require(dtype.itemsize == SAMPLE, "NumPy sample ABI")
    initial_counts = np.memmap(binary, dtype="<i4", mode="r", offset=HEADER.size, shape=(LOGICAL,))
    counts = initial_counts.copy()
    require(bool(np.all(counts >= 0)), "Negative initial penalty counts")
    samples = np.memmap(binary, dtype=dtype, mode="r", offset=HEADER.size + LOGICAL * 4,
                        shape=(count, 7))
    report = []
    weight = int(samples[0, 0]["weight"])
    ledger = metadata["payload"]["ledger_rows"]
    require(len(ledger) >= count + 1, "Missing native ledger rows")
    committed = list(metadata["payload"]["first_ledger"])
    for index, row in enumerate(ledger):
        if index:
            committed.extend(row["committed_delta"])
        require(row["round"] == index + 1 and len(committed) == row["ledger_frontier"] and
                row["ledger_frontier"] == row["frontier"] + 1 and
                committed[-1] == row["anchor"] and all(0 <= token < LOGICAL for token in
                (row["committed_delta"] if index else committed)), "Native ledger reconstruction")

    def raw(sample):
        return (sample["logits"].astype(np.uint32) << 16).view(np.float32)[:LOGICAL]

    def adjusted(values, prefix):
        present = counts > 0
        for token in prefix:
            present[token] = True
        return np.subtract(values, np.where(present, np.float32(1.5), np.float32(0)), dtype=np.float32)

    def top2(values):
        first = int(np.argmax(values))
        work = values.copy()
        work[first] = -np.inf
        return [first, int(np.argmax(work))]

    def check_scores(column, values, processed, prefix):
        for name, array in (("raw", values), ("adjusted", processed)):
            ids = top2(array)
            require(ids == [entry["id"] for entry in column[name]], "Full-logit top2 mismatch")
            for entry in column[name]:
                token = entry["id"]
                require(int(values.view(np.uint32)[token]) == entry["raw_bits"] and
                        int(processed.view(np.uint32)[token]) == entry["adjusted_bits"] and
                        int(counts[token]) == entry["base_count"] and
                        prefix.count(token) == entry["prefix_count"], "Score/count bits mismatch")

    def inputs(record):
        vectors = []
        for sample in record:
            data = sample["packed"]
            qs = data[:K].view(np.int8).astype(np.float64)
            scales = data[K:].view("<f2").reshape(K // 32, 2)[:, 0].astype(np.float64)
            require(bool(np.all(np.isfinite(scales))), "Nonfinite native Q8 scale")
            vectors.append((qs.reshape(K // 32, 32) * scales[:, None]).reshape(K))
        return np.stack(vectors)

    def geometry(vectors, valid):
        results = []
        for groups in (1, 5, 20, 32):
            indices = (np.arange(K).reshape(5, 32, 32).transpose(1, 0, 2).reshape(32, 160)
                       if groups == 32 else np.arange(K).reshape(groups, K // groups))
            columns = []
            for column in range(valid):
                residual2 = 0.0
                inflation2 = 0.0
                ranks = []
                for index in indices:
                    basis = vectors[:3, index].T
                    x = vectors[3 + column, index]
                    alpha, _, rank, _ = np.linalg.lstsq(basis, x, rcond=None)
                    remainder = x - basis @ alpha
                    residual2 += float(remainder @ remainder)
                    inflation2 += float(np.abs(alpha) @ np.linalg.norm(basis, axis=0)) ** 2
                    ranks.append(int(rank))
                norm = float(np.linalg.norm(vectors[3 + column]))
                columns.append({"column": column, "residual_norm": math.sqrt(residual2),
                    "target_norm": norm, "coefficient_inflation_norm": math.sqrt(inflation2),
                    "min_rank": min(ranks)})
            results.append({"groups": groups, "columns": columns})
        return results

    for i, record in enumerate(metadata["records"]):
        require(record["round"] == i + 2 and record["producer_round"] == i + 1,
                "Nonconsecutive producer/target rounds")
        require(ledger[i + 1]["round"] == record["round"] and
                ledger[i + 1]["frontier"] == record["frontier"] and
                ledger[i + 1]["anchor"] == record["anchor"], "Native ledger join")
        if i:
            require(ledger[i + 1]["committed_delta"] == metadata["records"][i - 1]["licensed"],
                    "Native ledger delta is not previous licensed span")
        for role, sample in enumerate(samples[i]):
            target = role >= 3
            require(int(sample["epoch"]) == epoch and int(sample["role"]) == role and
                    int(sample["round"]) == i + (2 if target else 1) and
                    int(sample["frontier"]) == record["frontier"] and
                    int(sample["anchor"]) == record["anchor"] and
                    int(sample["extent"]) == record["extent" if target else "q_extent"] and
                    int(sample["weight"]) == weight != 0, "Sample role/frontier/weight binding")
        drafts = record["drafts"]
        for step, q in enumerate(record["q"]):
            values = raw(samples[i, step])
            require(bool(np.all(np.isfinite(values))), "Nonfinite logical Q logits")
            processed = adjusted(values, drafts[:step])
            check_scores(q, values, processed, drafts[:step])
            require(int(np.argmax(values)) == q["actual"] == drafts[step], "Native Q argmax")
        winners = []
        for column, p in enumerate(record["p"]):
            values = raw(samples[i, 3 + column])
            require(bool(np.all(np.isfinite(values))), "Nonfinite logical P logits")
            processed = adjusted(values, drafts[:column])
            check_scores(p, values, processed, drafts[:column])
            require(int(np.argmax(values)) == p["actual_raw"], "Native P raw argmax")
            winners.append(int(np.argmax(processed)))
        accepted = 0
        while accepted < len(drafts) and drafts[accepted] == winners[accepted]:
            accepted += 1
        licensed = drafts[:accepted] + [winners[accepted]]
        require(accepted == record["accepted"] and licensed == record["licensed"],
                "Full-logit native acceptance mismatch")
        vectors = inputs(samples[i])
        target = vectors[3:record["extent"] + 4]
        drift = np.linalg.norm(target[:, None, :] - vectors[None, :3, :], axis=2)
        target_norm = np.linalg.norm(target, axis=1)
        report.append({"round": record["round"], "accepted": accepted,
            "pairwise_drift_norm": drift.tolist(), "target_norm": target_norm.tolist(),
            "pairwise_drift_over_target_norm": [[float(value) / float(norm) if norm else None
                for value in row] for row, norm in zip(drift, target_norm)],
            "oracle_processed_top1_bits": [p["adjusted"][0]["adjusted_bits"] for p in record["p"]],
            "oracle_raw_top1_bits": [p["raw"][0]["raw_bits"] for p in record["p"]],
            "oracle_thresholds_for_negative_filter_only": True,
            "geometry": geometry(vectors, record["extent"] + 1)})
        for token in licensed:
            counts[token] += 1
    with binary.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"status": "CAPTURE_NATIVE_REPLAY_AND_GEOMETRY_PASS", "epoch": epoch,
            "records": count, "binary_sha256": digest, "per_row_certificate": False,
            "geometry_is_nonrigorous_diagnostic": True, "native_lowering_proved": False,
            "performance_claim": False, "rounds": report}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=pathlib.Path)
    parser.add_argument("--binary", type=pathlib.Path)
    args = parser.parse_args()
    require(bool(args.metadata) == bool(args.binary), "Metadata and binary must be supplied together")
    result = analyze(json.loads(args.metadata.read_text()), args.binary) if args.metadata else self_test()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
