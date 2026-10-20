import argparse
import difflib
import hashlib
import json
import pathlib
import sys


CUDA_DIRECTORY = pathlib.Path("third_party/ggml-quants/src/ggml-cuda")
EXPECTED_SHA256 = {
    "third_party/ggml-quants/src/ggml-cuda/mmq.cuh": "493ce59f8be4ff269fd0a2a8386f9d1e9a80bd2ee9c7478083d36bc2ac9ccf45",
    "third_party/ggml-quants/src/ggml-cuda/mmq-load-tiles.cuh": "80a7e02c1b2019db4ebdc1ef398a405c0249174dcf13a3d74b5d29af0347e3cf",
    "third_party/ggml-quants/src/ggml-cuda/mmq-vec-dot.cuh": "e47f3be4edcf96341b9ad48df24bd87480051c50bf7cabeef251abcb6ae39c66",
    "src/ops/linear/gguf/ggml_bridge.cu": "670d0910869d8b1183b02d37f7ab2f554def75b9ff7ba7152562ce65327608e6",
    "src/ops/linear/gguf/ggml_bridge_mmq.cuh": "2513cb559034db1422841508c1c251e060ec087fe7b92643b2b539f687a7b15b",
    "tests/ops/linear/test_gguf_mmq_store.cu": "c88bcb861c57546cfe83cec1135019dafce50b165f814389921bc3f915d137b5",
}
SWITCH = "NINFER_GGUF_Q6_MMQ_ALIGNED_RAW32"
START = "template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q6_K("
END = "template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq1_s("
INSERT = """#ifndef NINFER_GGUF_Q6_MMQ_ALIGNED_RAW32
#define NINFER_GGUF_Q6_MMQ_ALIGNED_RAW32 0
#endif
    const auto q6_raw32_load = [](const void * p, const int i32) {
#if NINFER_GGUF_Q6_MMQ_ALIGNED_RAW32 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 860)
        if constexpr (type == GGML_TYPE_Q6_K && J == 128) {
            const auto * address = static_cast<const uint8_t *>(p) + 4*i32;
            int result;
            asm volatile (
                "{ .reg .pred aligned; .reg .u32 bits, low, high; "
                "cvt.u32.u64 bits, %1; and.b32 bits, bits, 3; "
                "setp.eq.u32 aligned, bits, 0; "
                "@aligned ld.global.nc.u32 low, [%1]; "
                "@!aligned ld.global.nc.u16 low, [%1]; "
                "@!aligned ld.global.nc.u16 high, [%1+2]; "
                "@!aligned mad.lo.u32 low, high, 65536, low; "
                "mov.b32 %0, low; }"
                : "=r"(result) : "l"(address) : "memory");
            return result;
        }
#endif
        return get_int_b2(p, i32);
    };
"""
CALLS = (
    "get_int_b2(bxi->ql, txi)",
    "get_int_b2(bxi->qh, (QI6_K/4) * (txi / (QI6_K/2)) + txi % (QI6_K/4))",
    "get_int_b2(bxi->scales, threadIdx.x % (MMQ_TILE_NE_K/8))",
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def span(data, start, end):
    first = start.encode()
    last = end.encode()
    if data.count(first) != 1 or data.count(last) != 1:
        raise ValueError("source markers are not unique")
    begin = data.index(first)
    stop = data.index(last, begin)
    return begin, stop, data[begin:stop]


def prepare(source_root):
    originals = {}
    for name, expected in EXPECTED_SHA256.items():
        data = (source_root / name).read_bytes()
        if sha256(data) != expected:
            raise ValueError("source SHA256 mismatch: " + name)
        originals[name] = data
    key = str(CUDA_DIRECTORY / "mmq-load-tiles.cuh").replace("\\", "/")
    source = originals[key]
    begin, stop, old = span(source, START, END)
    if SWITCH.encode() in source:
        raise ValueError("source already contains aligned raw32")
    newline = b"\r\n" if b"\r\n" in source else b"\n"
    marker = b"    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();"
    if old.count(marker) != 1:
        raise ValueError("unexpected Q6 loader prologue")
    insert = INSERT.replace("\n", newline.decode()).encode()
    new = old.replace(marker, insert + marker, 1)
    for call in CALLS:
        before = call.encode()
        if old.count(before) != 1:
            raise ValueError("expected exactly one Q6 load call: " + call)
        new = new.replace(before, call.replace("get_int_b2", "q6_raw32_load", 1).encode(), 1)
    generated = source[:begin] + new + source[stop:]
    if generated.count(new) != 1 or generated.replace(new, old, 1) != source:
        raise ValueError("bounded Q6 loader transform did not round-trip")
    reverse = new.replace(insert, b"", 1)
    for call in CALLS:
        reverse = reverse.replace(call.replace("get_int_b2", "q6_raw32_load", 1).encode(),
                                  call.encode(), 1)
    if reverse != old:
        raise ValueError("unexpected edit within the Q6 loader")
    files = {
        "ggml-cuda/mmq.cuh": originals[str(CUDA_DIRECTORY / "mmq.cuh").replace("\\", "/")],
        "ggml-cuda/mmq-load-tiles.cuh": generated,
        "ggml-cuda/mmq-vec-dot.cuh": originals[str(CUDA_DIRECTORY / "mmq-vec-dot.cuh").replace("\\", "/")],
    }
    bridge = originals["src/ops/linear/gguf/ggml_bridge.cu"]
    _, _, producer = span(bridge, "constexpr int kMatrixQuantizeThreads = 128;",
                         "template <mmq_q8_1_ds_layout layout_a, mmq_q8_1_ds_layout layout_b>")
    oracle_source = originals["tests/ops/linear/test_gguf_mmq_store.cu"]
    _, _, decoder = span(oracle_source, "double q6_runtime_weight_value(",
                        "int check_q6_runtime_oracle(")
    _, _, input_maker = span(oracle_source, "std::vector<__nv_bfloat16> make_input(",
                            "bool compare_rounding(")
    files["q6_native_producer.cuh"] = producer
    files["q6_independent_oracle.cuh"] = input_maker + decoder
    metadata = {
        "status": "PASS_BOUNDED_Q6_LOADER_TRANSFORM",
        "switch": SWITCH,
        "default": 0,
        "source_sha256": sha256(source),
        "source_loader_sha256": sha256(source),
        "generated_header_sha256": sha256(generated),
        "source_files": dict(EXPECTED_SHA256),
        "generated_headers": {name: sha256(data) for name, data in files.items()},
        "changed_function": "ggml_cuda_mmq_load_tiles_q6_K",
        "changed_load_calls": 3,
        "original_lines": [source[:begin].count(b"\n") + 1, source[:stop].count(b"\n")],
        "original_function_sha256": sha256(old),
        "generated_function_sha256": sha256(new),
        "unchanged_prefix_sha256": sha256(source[:begin]),
        "unchanged_suffix_sha256": sha256(source[stop:]),
        "siblings_byte_identical": all(
            files["ggml-cuda/" + name] == originals[str(CUDA_DIRECTORY / name).replace("\\", "/")]
            for name in ("mmq.cuh", "mmq-vec-dot.cuh")
        ),
        "native_producer_sha256": sha256(producer),
        "independent_q6_decoder_sha256": sha256(decoder),
        "input_maker_sha256": sha256(input_maker),
        "native_producer_entry": "quantize_matrix_kernel<MMQ_Q8_1_DS_LAYOUT_D4>",
        "performance_claim": False,
    }
    return source, generated, files, metadata


def self_test():
    counts = {"addresses": 0, "aligned4": 0, "fallback2": 0, "odd_starts": 0}
    for blocks in (20, 68):
        row_bytes = blocks * 210
        for start in range(blocks):
            counts["odd_starts"] += start & 1
            for row in (0, 1, 2, 127, 128):
                for block in range(start, blocks):
                    origin = row * row_bytes + block * 210
                    for offset, words in ((0, 32), (128, 16), (192, 4)):
                        for index in range(words):
                            address = origin + offset + 4 * index
                            if address & 1:
                                raise ValueError("Q6 address lost half-word alignment")
                            data = bytes(((address + i) * 73 + 19) & 255 for i in range(4))
                            old = int.from_bytes(data[:2], "little") | (
                                int.from_bytes(data[2:], "little") << 16)
                            new = int.from_bytes(data, "little") if address % 4 == 0 else old
                            if new != old:
                                raise ValueError("aligned word differs from two half words")
                            counts["addresses"] += 1
                            counts["aligned4" if address % 4 == 0 else "fallback2"] += 1
    if not counts["aligned4"] or not counts["fallback2"] or not counts["odd_starts"]:
        raise ValueError("address audit did not cover both paths")
    return {"status": "PASS_CPU_ADDRESS_EQUIVALENCE", **counts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, default=pathlib.Path("/src"))
    parser.add_argument("--output-root", type=pathlib.Path,
                        default=pathlib.Path("/build/q6-raw32/generated"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--diff", action="store_true")
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    original, generated, files, metadata = prepare(source_root)
    if args.self_test:
        metadata["address_check"] = self_test()
    if not args.check_only:
        destination = args.output_root.resolve()
        allowed = pathlib.Path("/build/q6-raw32").resolve()
        if not destination.is_relative_to(allowed) or destination.is_relative_to(source_root):
            raise ValueError("generated output must stay under /build/q6-raw32 outside source")
        for name, data in files.items():
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if path.read_bytes() != data:
                    raise ValueError("refusing a different existing generated file: " + name)
            else:
                with path.open("xb") as handle:
                    handle.write(data)
            if sha256(path.read_bytes()) != metadata["generated_headers"][name]:
                raise ValueError("generated file verification failed: " + name)
        metadata["output_root"] = str(destination)
    if args.diff:
        sys.stdout.writelines(difflib.unified_diff(
            original.decode().splitlines(keepends=True), generated.decode().splitlines(keepends=True),
            fromfile="original/ggml-cuda/mmq-load-tiles.cuh",
            tofile="generated/ggml-cuda/mmq-load-tiles.cuh"))
    metadata["check_only"] = args.check_only
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
