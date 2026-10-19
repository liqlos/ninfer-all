import bz2
import hashlib
import json
import mmap
import pathlib
import struct
import zlib

import numpy as np


ROOT = pathlib.Path(r"C:\AI\tuning\ninfer-lossless-20260928")
MODEL = pathlib.Path(r"C:\AI\staging\qwen38-jonathan-ninfer\qwen38-jonathan-q4km-proposal.ninfer")
CHECKS = ("weight/000001", "weight/000131")
TILE_BLOCKS = (16, 64, 256, 1024)


def unpack_model(f):
    magic, header_length, artifact_id = struct.unpack("<8sQ16s", f.read(32))
    if magic != b"NINFER\0\3":
        raise ValueError(f"unexpected magic {magic!r}")
    directory = json.loads(f.read(header_length))
    payload_offset = (32 + header_length + 4095) // 4096 * 4096
    return directory, payload_offset, artifact_id.hex(), header_length


def zip_bytes(data, codec):
    if codec == "zlib1":
        return zlib.compress(data, 1)
    return bz2.compress(data, compresslevel=1)


def unzip_bytes(data, codec):
    if codec == "zlib1":
        return zlib.decompress(data)
    return bz2.decompress(data)


def q4_pack(tile):
    h = tile[:, :4]
    s = tile[:, 4:16]
    q = tile[:, 16:144]
    return [h.T.copy().tobytes(), s.T.copy().tobytes(), q.T.copy().tobytes()]


def q4_unpack(parts, count):
    h = np.frombuffer(parts[0], dtype=np.uint8).reshape(4, count).T.copy()
    s = np.frombuffer(parts[1], dtype=np.uint8).reshape(12, count).T.copy()
    q = np.frombuffer(parts[2], dtype=np.uint8).reshape(128, count).T.copy()
    return np.concatenate((h, s, q), axis=1).tobytes()


def q6_pack(tile):
    return [tile[:, :128].T.copy().tobytes(), tile[:, 128:192].T.copy().tobytes(), tile[:, 192:208].T.copy().tobytes(), tile[:, 208:210].T.copy().tobytes()]


def q6_unpack(parts, count):
    ql = np.frombuffer(parts[0], dtype=np.uint8).reshape(128, count).T.copy()
    qh = np.frombuffer(parts[1], dtype=np.uint8).reshape(64, count).T.copy()
    sc = np.frombuffer(parts[2], dtype=np.uint8).reshape(16, count).T.copy()
    d = np.frombuffer(parts[3], dtype=np.uint8).reshape(2, count).T.copy()
    return np.concatenate((ql, qh, sc, d), axis=1).tobytes()


def main():
    directory = json.loads((ROOT / "directory.json").read_text(encoding="utf-8"))
    analysis = json.loads((ROOT / "results" / "analysis.json").read_text(encoding="utf-8"))
    with MODEL.open("rb") as f:
        parsed, payload_offset, artifact_id, header_length = unpack_model(f)
        if parsed != directory or payload_offset != analysis["payload_offset"] or artifact_id != analysis["artifact_id"]:
            raise ValueError("model header identity mismatch")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        results = []
        for name in CHECKS:
            item = next(t for t in directory["objects"] if t["id"] == name)
            stride = 144 if item["format"] == "gguf_q4_k" else 210
            codec_candidates = (("q4_separate_fields", q4_pack, q4_unpack), ("q4_whole_block_byte_shuffle", lambda a: [a.T.copy().tobytes()], lambda p, n: np.frombuffer(p[0], dtype=np.uint8).reshape(stride, n).T.copy().tobytes())) if stride == 144 else (("q6_separate_fields", q6_pack, q6_unpack), ("q6_whole_block_byte_shuffle", lambda a: [a.T.copy().tobytes()], lambda p, n: np.frombuffer(p[0], dtype=np.uint8).reshape(stride, n).T.copy().tobytes()))
            for tile_blocks in TILE_BLOCKS:
                relative_start = ((item["bytes"] // stride) // 3 // tile_blocks) * tile_blocks
                tile = np.frombuffer(mm, dtype=np.uint8, offset=payload_offset + item["offset"] + relative_start * stride, count=tile_blocks * stride).copy().reshape(tile_blocks, stride)
                original = tile.tobytes()
                for label, pack, unpack in codec_candidates:
                    streams = pack(tile)
                    for codec in ("zlib1", "bz2_1"):
                        encoded = [zip_bytes(part, codec) for part in streams]
                        decoded = [unzip_bytes(part, codec) for part in encoded]
                        reconstructed = unpack(decoded, tile_blocks)
                        results.append({"tensor": name, "format": item["format"], "tile_blocks": tile_blocks, "transform": label, "codec": codec, "original_bytes": len(original), "encoded_stream_bytes": sum(map(len, encoded)), "restored_bytes": len(reconstructed), "source_sha256": hashlib.sha256(original).hexdigest(), "restored_sha256": hashlib.sha256(reconstructed).hexdigest(), "exact_match": reconstructed == original})
        mm.close()
    if not all(r["exact_match"] for r in results):
        raise ValueError("one or more independently compressed tiles failed exact roundtrip")
    result = {"model": str(MODEL), "artifact_id": artifact_id, "header_bytes": header_length, "payload_offset_derived_from_header": payload_offset, "checks": results, "all_exact": True}
    (ROOT / "structure-roundtrip.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"checks": len(results), "all_exact": result["all_exact"], "payload_offset": payload_offset}, separators=(",", ":")))


if __name__ == "__main__":
    main()
