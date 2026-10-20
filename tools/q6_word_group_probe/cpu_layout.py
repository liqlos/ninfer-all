import argparse
import collections
import dataclasses
import hashlib
import json
import pathlib
import random


@dataclasses.dataclass(frozen=True)
class View:
    parent_rows: int
    parent_B: int
    row_begin: int
    rows: int
    payload_bytes: int
    final_slab_blocks: int

    def validate(self):
        n = self.parent_rows * self.parent_B
        if (self.parent_rows <= 0 or self.parent_B <= 0 or self.rows <= 0 or
                self.row_begin < 0 or self.row_begin + self.rows > self.parent_rows or
                self.payload_bytes != ((n + 15) // 16) * 3360 or
                self.final_slab_blocks != (n - 1) % 16 + 1):
            raise ValueError("invalid parent-backed row view")

    def block(self, row, kb):
        self.validate()
        if not 0 <= row < self.rows or not 0 <= kb < self.parent_B:
            raise ValueError("row or K block is outside the view")
        return (self.row_begin + row) * self.parent_B + kb


def native_codes(block):
    if len(block) != 210:
        raise ValueError("native Q6 block must contain 210 bytes")
    values = [0] * 256
    for half in range(2):
        for lane in range(32):
            lo0 = block[half * 64 + lane]
            lo1 = block[half * 64 + 32 + lane]
            hi = block[128 + half * 32 + lane]
            values[half * 128 + lane] = (lo0 & 15) | ((hi & 3) << 4)
            values[half * 128 + 32 + lane] = (lo1 & 15) | (((hi >> 2) & 3) << 4)
            values[half * 128 + 64 + lane] = (lo0 >> 4) | (((hi >> 4) & 3) << 4)
            values[half * 128 + 96 + lane] = (lo1 >> 4) | (((hi >> 6) & 3) << 4)
    return values


def native_encode(values, metadata):
    if len(values) != 256 or len(metadata) != 18 or any(not 0 <= q < 64 for q in values):
        raise ValueError("invalid code or metadata extent")
    result = bytearray(210)
    for k, value in enumerate(values):
        half, within = divmod(k, 128)
        quarter, lane = divmod(within, 32)
        result[half * 64 + (quarter % 2) * 32 + lane] |= (value & 15) << (4 * (quarter // 2))
        result[128 + half * 32 + lane] |= (value >> 4) << (2 * quarter)
    result[192:210] = metadata
    return bytes(result)


def group_encode(values):
    if len(values) != 16 or any(not 0 <= q < 64 for q in values):
        raise ValueError("group must contain sixteen 6-bit codes")
    low = bytes((values[t] & 15) | ((values[t + 4] & 15) << 4)
                for base in (0, 8) for t in range(base, base + 4))
    high = bytes(sum((values[t + 4 * j] >> 4) << (2 * j) for j in range(4))
                 for t in range(4))
    return low, high


def group_words(low, high):
    if len(low) != 8 or len(high) != 4:
        raise ValueError("group fields require QL64 and QH32")
    lo0 = int.from_bytes(low[:4], "little")
    lo1 = int.from_bytes(low[4:], "little")
    hi = int.from_bytes(high, "little")
    return (
        (lo0 & 0x0f0f0f0f) | ((hi << 4) & 0x30303030),
        ((lo0 >> 4) & 0x0f0f0f0f) | ((hi << 2) & 0x30303030),
        (lo1 & 0x0f0f0f0f) | (hi & 0x30303030),
        ((lo1 >> 4) & 0x0f0f0f0f) | ((hi >> 2) & 0x30303030),
    )


def group_decode(low, high):
    return list(b"".join(word.to_bytes(4, "little") for word in group_words(low, high)))


def offsets(block, group=0):
    slab, slot = divmod(block, 16)
    base = slab * 3360
    return (base + slot * 128 + group * 8,
            base + 2048 + slot * 64 + group * 4,
            base + 3072 + slot * 16 + group,
            base + 3328 + slot * 2)


def pack_parent(native, rows, blocks_per_row):
    blocks = rows * blocks_per_row
    if rows <= 0 or blocks_per_row <= 0 or len(native) != blocks * 210:
        raise ValueError("native parent extent mismatch")
    result = bytearray(((blocks + 15) // 16) * 3360)
    for block in range(blocks):
        original = native[block * 210:(block + 1) * 210]
        values = native_codes(original)
        for group in range(16):
            lo, hi = group_encode(values[group * 16:(group + 1) * 16])
            ql, qh, _, _ = offsets(block, group)
            result[ql:ql + 8], result[qh:qh + 4] = lo, hi
        _, _, sc, d = offsets(block)
        result[sc:sc + 16], result[d:d + 2] = original[192:208], original[208:210]
    view = View(rows, blocks_per_row, 0, rows, len(result), (blocks - 1) % 16 + 1)
    view.validate()
    return bytes(result), view


def unpack_block(payload, block):
    values = []
    for group in range(16):
        ql, qh, _, _ = offsets(block, group)
        values.extend(group_decode(payload[ql:ql + 8], payload[qh:qh + 4]))
    _, _, sc, d = offsets(block)
    return native_encode(values, payload[sc:sc + 16] + payload[d:d + 2])


def unpack_view(payload, view):
    view.validate()
    if len(payload) != view.payload_bytes:
        raise ValueError("payload extent differs from descriptor")
    return b"".join(unpack_block(payload, view.block(row, kb))
                    for row in range(view.rows) for kb in range(view.parent_B))


def assert_decode(payload, view, native):
    view.validate()
    for row in range(view.rows):
        for kb in range(view.parent_B):
            block = view.block(row, kb)
            original = native[block * 210:(block + 1) * 210]
            expected = native_codes(original)
            actual = []
            for group in range(16):
                ql, qh, sc, d = offsets(block, group)
                assert ql % 8 == 0 and qh % 4 == 0 and d % 2 == 0
                assert ql + 8 <= len(payload) and qh + 4 <= len(payload)
                assert payload[sc] == original[192 + group]
                words = group_words(payload[ql:ql + 8], payload[qh:qh + 4])
                for word in words:
                    centered = word ^ 0xe0e0e0e0 ^ ((word & 0x20202020) * 6)
                    signed = [q if q < 128 else q - 256 for q in centered.to_bytes(4, "little")]
                    unsigned = list(word.to_bytes(4, "little"))
                    assert signed == [q - 32 for q in unsigned]
                    actual.extend(unsigned)
            assert actual == expected
            _, _, _, d = offsets(block)
            assert payload[d:d + 2] == original[208:210]
        for lane in range(32):
            iteration = 0
            while lane + 32 * iteration < view.parent_B * 8:
                kb, sub = divmod(lane + 32 * iteration, 8)
                block = view.block(row, kb)
                original = native_codes(native[block * 210:(block + 1) * 210])
                actual = []
                for group in (2 * sub, 2 * sub + 1):
                    ql, qh, _, _ = offsets(block, group)
                    actual.extend(group_decode(payload[ql:ql + 8], payload[qh:qh + 4]))
                assert actual == original[sub * 32:(sub + 1) * 32]
                iteration += 1


def ownership_proof():
    stores = []
    transactions = 0
    for i0 in range(0, 128, 16):
        for warp in range(8):
            lane_words = []
            for lane in range(32):
                row = i0 + 2 * warp + lane // 16
                group = lane % 16
                words = [row * 76 + group * 4 + j for j in range(4)]
                assert (4 * words[0]) % 16 == 0
                lane_words.append(words)
                stores.extend(words)
            for first in range(0, 32, 8):
                banks = [word % 32 for words in lane_words[first:first + 8] for word in words]
                assert sorted(banks) == list(range(32))
                transactions += 1
    expected = {row * 76 + word for row in range(128) for word in range(64)}
    assert set(stores) == expected and len(stores) == len(expected) == 8192
    scales = [(i0 + warp * 8 + lane // 4) % 128 * 76 + 65 + lane % 4
              for i0 in range(0, 128, 64) for warp in range(8) for lane in range(32)]
    assert len(scales) == len(set(scales)) == 512
    d = [(warp * 32 + lane) % 128 * 76 + 64 for warp in range(8) for lane in range(32)]
    assert set(collections.Counter(d).values()) == {2}
    assert not set(stores).intersection(d + scales)
    return {"code_words_once": len(stores), "scale_words_once": len(scales),
            "native_duplicate_d_writes_per_row": 2,
            "modeled_128B_transactions_without_bank_conflicts": transactions}


def self_test():
    rng = random.Random(186)
    exhaustive = 0
    for position in range(16):
        for value in range(64):
            codes = [(i * 13 + 5) % 64 for i in range(16)]
            codes[position] = value
            lo, hi = group_encode(codes)
            assert group_decode(lo, hi) == codes
            exhaustive += 1
    basis = []
    for byte in range(210):
        for bit in range(8):
            block = bytearray(210)
            block[byte] = 1 << bit
            basis.append(bytes(block))
    raw = b"".join(basis)
    packed, view = pack_parent(raw, len(basis), 1)
    assert unpack_view(packed, view) == raw
    assert_decode(packed, view, raw)
    cases = 0
    invalid = 0
    for rows, B in ((1, 1), (1, 3), (2, 5), (3, 17), (7, 3), (17, 5),
                    (19, 20), (3, 68), (2, 4), (9, 1), (33, 7)):
        raw = rng.randbytes(rows * B * 210)
        packed, parent = pack_parent(raw, rows, B)
        assert unpack_view(packed, parent) == raw
        for origin in sorted({0, min(1, rows - 1), rows // 2, rows - 1}):
            for count in sorted({1, rows - origin}):
                view = dataclasses.replace(parent, row_begin=origin, rows=count)
                assert unpack_view(packed, view) == raw[origin * B * 210:(origin + count) * B * 210]
                assert_decode(packed, view, raw)
                cases += 1
        for block in range(rows * B, (rows * B + 15) // 16 * 16):
            ql, qh, sc, d = offsets(block)
            assert packed[ql:ql + 128] == bytes(128)
            assert packed[qh:qh + 64] == bytes(64)
            assert packed[sc:sc + 16] == bytes(16)
            assert packed[d:d + 2] == bytes(2)
        for bad in (dataclasses.replace(parent, row_begin=rows),
                    dataclasses.replace(parent, parent_B=B + 1),
                    dataclasses.replace(parent, payload_bytes=len(packed) - 1),
                    dataclasses.replace(parent, final_slab_blocks=0)):
            try:
                bad.validate()
            except ValueError:
                invalid += 1
            else:
                raise AssertionError("invalid descriptor was accepted")
    return {"group_position_code_cases": exhaustive, "all_native_bit_basis_cases": len(basis),
            "odd_parent_and_origin_view_cases": cases, "invalid_descriptors_rejected": invalid,
            "ownership": ownership_proof()}


def fixture_samples(path, rows, B):
    expected_bytes = rows * B * 210
    if path.stat().st_size != expected_bytes:
        raise ValueError("fixture extent mismatch: " + str(path))
    row_origins = sorted({0, 1, 7, rows // 2, rows - 4})
    sample_hash = hashlib.sha256()
    samples = []
    with path.open("rb") as stream:
        for origin in row_origins:
            count = min(4, rows - origin)
            stream.seek(origin * B * 210)
            raw = stream.read(count * B * 210)
            if len(raw) != count * B * 210:
                raise ValueError("truncated fixture sample")
            sample_hash.update(raw)
            packed, view = pack_parent(raw, count, B)
            assert unpack_view(packed, view) == raw
            assert_decode(packed, view, raw)
            child = dataclasses.replace(view, row_begin=1, rows=count - 1)
            assert unpack_view(packed, child) == raw[B * 210:]
            assert_decode(packed, child, raw)
            samples.append({"source_row": origin, "rows": count, "bytes": len(raw)})
    return {"path": str(path.resolve()), "original_bytes": expected_bytes,
            "sample_sha256": sample_hash.hexdigest(), "samples": samples,
            "full_fixture_hash_checked": False, "full_fixture_roundtrip": "deferred"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-root", type=pathlib.Path)
    args = parser.parse_args()
    result = {"cpu_proof": self_test(), "fixtures": [],
              "cuda_compile": "not_run", "gpu": "not_run",
              "fp32_outputs": "not_qualified", "performance": "not_measured"}
    if args.fixture_root:
        for relative, rows, B in (
                ("operators/q6-full-k17408.bin", 5120, 68),
                ("weights/q6-head-full-k5120.bin", 248320, 20)):
            result["fixtures"].append(fixture_samples(args.fixture_root / relative, rows, B))
    result["status"] = "PASS_BOUNDED_CPU_CODEC_AND_MAPPING"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
