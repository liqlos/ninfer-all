import argparse
import difflib
import hashlib
import json
import pathlib
import struct
import sys


SOURCE_DIRECTORY = pathlib.Path("third_party/ggml-quants/src/ggml-cuda")
CHANGED_HEADER = "mmq-vec-dot.cuh"
SWITCH = "NINFER_GGUF_Q4_MMQ_BIASED_DOT"
EXPECTED_SHA256 = {
    "mmq.cuh": "493ce59f8be4ff269fd0a2a8386f9d1e9a80bd2ee9c7478083d36bc2ac9ccf45",
    "mmq-load-tiles.cuh": "80a7e02c1b2019db4ebdc1ef398a405c0249174dcf13a3d74b5d29af0347e3cf",
    "mmq-vec-dot.cuh": "e47f3be4edcf96341b9ad48df24bd87480051c50bf7cabeef251abcb6ae39c66",
}
ORIGINAL = """#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n][k01/QI8_1], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    float2 dm;
#if defined(NINFER_GGUF_Q4_MMQ_PACKED_DM) && __CUDA_ARCH__ == 860
                    if constexpr (type == GGML_TYPE_Q4_K && J == 128) {
                        dm = __half22float2(dmA[n][l/2][k01/QI8_1]);
                    } else
#endif
                    {
                        dm = dmA[n][l/2][k01/QI8_1];
                    }
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dm.x*dsB[l%2].x*C.x[l];
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dm.y*dsB[l%2].y;
                }
            }
"""
REPLACEMENT = """#ifndef NINFER_GGUF_Q4_MMQ_BIASED_DOT
#define NINFER_GGUF_Q4_MMQ_BIASED_DOT 0
#endif
#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
#if NINFER_GGUF_Q4_MMQ_BIASED_DOT && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 860)
                if constexpr (type == GGML_TYPE_Q4_K && J == 128) {
#pragma unroll
                    for (int l = 0; l < tile_C::ne; ++l) {
                        C.x[l] = 0x4b400000;
                    }
                }
#endif
                mma(C, A[n][k01/QI8_1], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    float2 dm;
#if defined(NINFER_GGUF_Q4_MMQ_PACKED_DM) && __CUDA_ARCH__ == 860
                    if constexpr (type == GGML_TYPE_Q4_K && J == 128) {
                        dm = __half22float2(dmA[n][l/2][k01/QI8_1]);
                    } else
#endif
                    {
                        dm = dmA[n][l/2][k01/QI8_1];
                    }
#if NINFER_GGUF_Q4_MMQ_BIASED_DOT && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 860)
                    if constexpr (type == GGML_TYPE_Q4_K && J == 128) {
                        const float dot = __fsub_rn(__int_as_float(C.x[l]), 12582912.0f);
                        sum[(j0/tile_C::J + n)*tile_C::ne + l] += dm.x*dsB[l%2].x*dot;
                    } else
#endif
                    {
                        sum[(j0/tile_C::J + n)*tile_C::ne + l] += dm.x*dsB[l%2].x*C.x[l];
                    }
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dm.y*dsB[l%2].y;
                }
            }
"""


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def prepare(source_root):
    originals = {}
    for name, expected in EXPECTED_SHA256.items():
        data = (source_root / SOURCE_DIRECTORY / name).read_bytes()
        if sha256(data) != expected:
            raise ValueError("source SHA256 mismatch: " + name)
        originals[name] = data
    source = originals[CHANGED_HEADER]
    newline = "\r\n" if b"\r\n" in source else "\n"
    old = ORIGINAL.replace("\n", newline).encode("utf-8")
    new = REPLACEMENT.replace("\n", newline).encode("utf-8")
    if source.count(old) != 1:
        raise ValueError("expected exactly one generic dot32 consumer fragment")
    if SWITCH.encode("ascii") in source:
        raise ValueError("source already contains biased dot")
    start = source.index(old)
    end = start + len(old)
    generated = source[:start] + new + source[end:]
    if generated.count(new) != 1 or generated.replace(new, old, 1) != source:
        raise ValueError("generated source does not round-trip through the single fragment")
    files = dict(originals)
    files[CHANGED_HEADER] = generated
    line = source[:start].count(b"\n") + 1
    metadata = {
        "status": "PASS_SINGLE_FRAGMENT_TRANSFORM",
        "switch": SWITCH,
        "default": 0,
        "changed_header": CHANGED_HEADER,
        "source_sha256": EXPECTED_SHA256[CHANGED_HEADER],
        "generated_header_sha256": sha256(generated),
        "source_headers": dict(EXPECTED_SHA256),
        "generated_headers": {name: sha256(data) for name, data in files.items()},
        "changed_fragments": 1,
        "original_lines": [line, line + old.count(b"\n") - 1],
        "generated_lines": [line, line + new.count(b"\n") - 1],
        "original_fragment_sha256": sha256(old),
        "generated_fragment_sha256": sha256(new),
        "unchanged_prefix_sha256": sha256(source[:start]),
        "unchanged_suffix_sha256": sha256(source[end:]),
        "siblings_byte_identical": all(
            files[name] == originals[name] for name in EXPECTED_SHA256 if name != CHANGED_HEADER
        ),
    }
    return originals, files, metadata


def self_test():
    bias_bits = 0x4B400000
    bias = struct.unpack("<f", struct.pack("<I", bias_bits))[0]
    lower = 32 * 15 * -128
    upper = 32 * 15 * 127
    if bias != 12582912.0 or not 0 < bias_bits + lower <= bias_bits + upper < 2**31:
        raise ValueError("invalid bias or INT32 range")
    for dot in range(lower, upper + 1):
        encoded = struct.unpack("<f", struct.pack("<I", bias_bits + dot))[0]
        if encoded != bias + dot:
            raise ValueError("biased word is not the exact FP32 integer at " + str(dot))
        actual = struct.pack("<f", encoded - bias)
        expected = struct.pack("<f", float(dot))
        if actual != expected:
            raise ValueError("FP32 conversion differs at " + str(dot))
    return {
        "status": "PASS",
        "dot_terms": 32,
        "weight_range": [0, 15],
        "activation_range": [-128, 127],
        "dot_range": [lower, upper],
        "integers_checked": upper - lower + 1,
        "bias_bits": hex(bias_bits),
        "bias_value": bias,
        "bitwise_fp32_equal": True,
        "positive_zero_equal": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, default=pathlib.Path("/src"))
    parser.add_argument("--output-root", type=pathlib.Path,
                        default=pathlib.Path("/build/q4-biased-dot/generated"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--diff", action="store_true")
    args = parser.parse_args()
    originals, files, metadata = prepare(args.source_root.resolve())
    if args.self_test:
        metadata["arithmetic_check"] = self_test()
    if not args.check_only:
        destination = args.output_root.resolve() / "ggml-cuda"
        source_directory = (args.source_root.resolve() / SOURCE_DIRECTORY).resolve()
        if destination == source_directory or destination.is_relative_to(args.source_root.resolve()):
            raise ValueError("generated files must be outside the source tree")
        destination.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            path = destination / name
            if path.exists():
                if path.read_bytes() != data:
                    raise ValueError("refusing to overwrite a different generated header: " + name)
            else:
                with path.open("xb") as handle:
                    handle.write(data)
            if sha256(path.read_bytes()) != metadata["generated_headers"][name]:
                raise ValueError("generated header verification failed: " + name)
        metadata["output_root"] = str(destination.parent)
    metadata["check_only"] = args.check_only
    if args.diff:
        sys.stdout.writelines(difflib.unified_diff(
            originals[CHANGED_HEADER].decode("utf-8").splitlines(keepends=True),
            files[CHANGED_HEADER].decode("utf-8").splitlines(keepends=True),
            fromfile="original/ggml-cuda/" + CHANGED_HEADER,
            tofile="generated/ggml-cuda/" + CHANGED_HEADER,
        ))
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
