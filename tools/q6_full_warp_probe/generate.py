import argparse
import difflib
import hashlib
import json
import pathlib
import re
import sys


VECTOR = "src/ops/linear/gguf/ggml_bridge_vec.cuh"
SWITCH = "NINFER_Q6_FULL_WARP_SLICES"
HEADER = "q6_full_warp_generated.cuh"
EXPECTED_SHA256 = {
    VECTOR: "0a0c82d6cede1707f91044efa70aeb39a536eeecc53b479b5c976e92c82e6596",
    "src/ops/linear/gguf/ggml_bridge_internal.cuh":
        "8b78c42a2cb01b10e1227987cae22b3d0a030684070d6f35a213b033fb3c9d90",
    "src/ops/linear/gguf/ggml_bridge.h":
        "63dc7967143129a4e454824d0176cfbf9e6c8502b4b88423acdd291e74856b76",
    "src/ops/common/memory.cuh":
        "68d4ce555d86761aa4c21d49726be2e48d82fcbdc4f6dbe4fc8b7356f945e205",
    "src/ops/common/mma.cuh":
        "4f3f05cfb7ecce39a650cf53588587997a9e8bee4faf992e23564cf8c696d94d",
    "third_party/ggml-quants/src/ggml-cuda/common.cuh":
        "6010078d639db19b1e8a2f78c4661ad240659b7bfa5a22864e315a2cbbc8da9b",
    "third_party/ggml-quants/src/ggml-cuda/vecdotq.cuh":
        "a7fd4281ad9f0a6d9373c776325411d2522e0b62ffae2c16e91e667a1af251fe",
    "third_party/ggml-quants/src/ggml-cuda/mmq.cuh":
        "493ce59f8be4ff269fd0a2a8386f9d1e9a80bd2ee9c7478083d36bc2ac9ccf45",
}
PROBE_FILES = ("generate.py", "variant_impl.cuh", "variant_off.cu", "variant_on.cu",
               "build.sh", "sass_gate.py")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def span(data, first, following):
    if data.count(first) != 1 or data.count(following) != 1:
        raise ValueError("bounded source anchors are not unique")
    begin, end = data.index(first), data.index(following)
    if begin >= end:
        raise ValueError("bounded source anchors are reversed")
    return begin, end, data[begin:end]


def replace(data, old, new, count=1):
    if old == new or data.count(old) != count or new in data:
        raise ValueError("replacement does not have its expected unique shape")
    result = data.replace(old, new)
    if result.count(new) != count or result.replace(new, old) != data:
        raise ValueError("replacement does not reverse exactly")
    return result


def prepare(source_root):
    originals = {}
    for name, expected in EXPECTED_SHA256.items():
        data = (source_root / name).read_bytes()
        if sha(data) != expected:
            raise ValueError("pinned source changed: " + name)
        originals[name] = data
    source = originals[VECTOR]
    newline = b"\r\n" if b"\r\n" in source else b"\n"
    enc = lambda text: text.replace("\n", newline.decode()).encode()
    if b"FullWarpSlices" in source or SWITCH.encode() in source:
        raise ValueError("native source already contains the candidate")
    regions = []
    begin, end, old = span(source, enc("template <>\nstruct Decoder<GGML_TYPE_Q6_K> {"),
                           enc("template <>\nstruct BaselineDecoder<GGML_TYPE_Q6_K> {"))
    new = replace(old,
                  enc("    __device__ __forceinline__ static void decode("),
                  enc("    template <bool FullWarpSlices = false>\n"
                      "    __device__ __forceinline__ static void decode("))
    new = replace(new, enc("        const unsigned mask = __activemask();"),
                  enc("        const unsigned mask = FullWarpSlices ? 0xffffffffu : __activemask();"))
    regions.append(("q6_decode_mask", begin, end, old, new))
    begin, end, old = span(source,
                           enc("template <ggml_type type, int T, bool Fused, int KS, bool Prefetch2 = true,"),
                           enc("template <ggml_type type, int T, bool Fused, int KS>\nvoid launch_baseline("))
    new = replace(old,
                  enc("          bool Baseline = false, int Rows = Baseline ? 2 : vector_rows_per_warp<type, T, Fused>()>"),
                  enc("          bool Baseline = false, int Rows = Baseline ? 2 : vector_rows_per_warp<type, T, Fused>(),\n"
                      "          bool FullWarpSlices = false>"))
    new = replace(new,
                  enc("__launch_bounds__(kVecWarps * 32) __global__ void kernel(VecArgs p) {"),
                  enc("__launch_bounds__(kVecWarps * 32) __global__ void kernel(VecArgs p) {\n"
                      "    static_assert(!FullWarpSlices || (NINFER_Q6_FULL_WARP_SLICES == 1 &&\n"
                      "        type == GGML_TYPE_Q6_K && (T == 1 || T == 4) && !Fused &&\n"
                      "        KS == 0 && !Baseline && Rows == 2));"))
    decode = "D::decode(rowp[r] + std::int64_t(32 / kPerBlock) * it * D::kBlockBytes, sub, table, sl);"
    new = replace(new, enc("                " + decode),
                  enc("                if constexpr (FullWarpSlices) {\n"
                      "                    D::template decode<true>(rowp[r] + std::int64_t(32 / kPerBlock) * it * D::kBlockBytes, sub, table, sl);\n"
                      "                } else {\n"
                      "                    " + decode + "\n"
                      "                }"))
    loop = ("        } else {\n"
            "            for (int it = 0; lane + 32 * it < slices; ++it) { slice(it); }\n"
            "        }")
    new = replace(new, enc(loop),
                  enc("        } else if constexpr (FullWarpSlices) {\n"
                      "            for (int it = 0; it < k / 1024; ++it) { slice(it); }\n" + loop), 2)
    regions.append(("runtime_loop_and_decode_selection", begin, end, old, new))
    generated = source
    for _, begin, end, old, new in reversed(regions):
        if generated[begin:end] != old:
            raise ValueError("transformed regions overlap")
        generated = generated[:begin] + new + generated[end:]
    include_pairs = (
        ('#include "ggml_bridge_internal.cuh"',
         '#include "ops/linear/gguf/ggml_bridge_internal.cuh"'),
        ('#include "../../common/memory.cuh"', '#include "ops/common/memory.cuh"'),
        ('#include "../../common/mma.cuh"', '#include "ops/common/mma.cuh"'),
    )
    for old, new in include_pairs:
        generated = replace(generated, enc(old), enc(new))
    native_flags = set(re.findall(rb"NINFER_GGUF_[A-Z0-9_]+", source))
    prohibited = sorted(flag.decode() for flag in native_flags
                        if flag != b"NINFER_GGUF_Q6_COOPERATIVE_DECODE")
    guard = (
        "#pragma once\n"
        "#ifndef NINFER_Q6_FULL_WARP_PROBE_INCLUDE\n"
        '#error "the generated vector header is for this isolated probe only"\n'
        "#endif\n"
        "#ifndef NINFER_Q6_FULL_WARP_SLICES\n"
        "#define NINFER_Q6_FULL_WARP_SLICES 0\n"
        "#endif\n"
        "#if NINFER_Q6_FULL_WARP_SLICES != 0 && NINFER_Q6_FULL_WARP_SLICES != 1\n"
        '#error "invalid FullWarpSlices switch"\n'
        "#endif\n"
        "#if !defined(NINFER_GGUF_Q6_COOPERATIVE_DECODE) || NINFER_GGUF_Q6_COOPERATIVE_DECODE != 1\n"
        '#error "both variants require the same native Q6 cooperative decode"\n'
        "#endif\n"
        "#if " + " || \\\n    ".join("defined(" + name + ")" for name in prohibited) + "\n"
        '#error "native vector header already included or unrelated vector switch enabled"\n'
        "#endif\n"
        "#define NINFER_Q6_FULL_WARP_GENERATED_HEADER 1\n"
    )
    prefix = enc(guard)
    generated = prefix + generated
    reverse = generated[len(prefix):]
    for old, new in reversed(include_pairs):
        reverse = replace(reverse, enc(new), enc(old))
    for _, _, _, old, new in reversed(regions):
        reverse = replace(reverse, new, old)
    if reverse != source:
        raise ValueError("complete header transform did not round-trip byte exactly")
    protected = {}
    for label, first, following in (
        ("row_ownership_and_activation",
         "    for (int row0 = (blockIdx.x * kVecWarps + warp) * step; row0 < p.rows;",
         "                " + decode),
        ("reduction_and_store",
         "#pragma unroll\n        for (int r = 0; r < kRows; ++r) {\n#pragma unroll\n            for (int j = 0; j < T; ++j) {\n#pragma unroll\n                for (int o = 16;",
         "template <ggml_type type, int T, bool Fused, int KS>\nvoid launch_baseline("),
        ("native_launch_guards",
         "template <ggml_type type, int T, bool Fused, int KS>\nvoid launch_baseline(",
         "#define NINFER_GGUF_VECTOR_INSTANCE(TYPE)"),
    ):
        _, _, data = span(source, enc(first), enc(following))
        if generated.count(data) != 1:
            raise ValueError("protected native region changed: " + label)
        protected[label] = sha(data)
    probe_root = pathlib.Path(__file__).resolve().parent
    metadata = {
        "status": "PASS_BOUNDED_FULL_WARP_SOURCE_TRANSFORM",
        "source_root": str(source_root), "probe_root": str(probe_root),
        "source_files": EXPECTED_SHA256,
        "probe_files": {name: sha((probe_root / name).read_bytes()) for name in PROBE_FILES},
        "generated_header": HEADER, "generated_header_sha256": sha(generated),
        "reverse_roundtrip_byte_exact": True, "protected_native_regions": protected,
        "changed_regions": [
            {"name": name, "native_lines": [source[:begin].count(b"\n") + 1,
                                           source[:end].count(b"\n")],
             "native_sha256": sha(old), "generated_sha256": sha(new)}
            for name, begin, end, old, new in regions],
        "switch": SWITCH, "default": 0, "columns": [1, 4], "KS": 0,
        "admission": "SM86, Q6 cooperative, !Fused, positive K divisible by 1024",
        "unselected_probe_calls": "return false; caller retains the native fallback",
        "numerical_validation_run": False, "performance_claim": False,
    }
    return source, generated, metadata


def self_test():
    checked_slices = 0
    for k in (1024, 2048, 5120, 17408, 32768):
        for lane in range(32):
            native = list(range(lane, k // 32, 32))
            candidate = [lane + 32 * it for it in range(k // 1024)]
            if native != candidate:
                raise ValueError("ordered per-lane native slices changed")
            checked_slices += len(native)
    partial = {}
    for k in (256, 768, 1280, 4864, 17152):
        counts = [len(range(lane, k // 32, 32)) for lane in range(32)]
        if len(set(counts)) != 2:
            raise ValueError("partial-warp rejection witness failed")
        partial[k] = sorted(set(counts))
    return {"status": "PASS_ORDERED_SLICE_PARTICIPATION_PROOF",
            "checked_slices": checked_slices, "rejected_partial_warp_iteration_counts": partial}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, default=pathlib.Path("/src"))
    parser.add_argument("--output-root", type=pathlib.Path,
                        default=pathlib.Path("/build/q6-full-warp/generated"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--diff", action="store_true")
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    original, generated, metadata = prepare(source_root)
    if args.self_test:
        metadata["slice_participation"] = self_test()
    if not args.check_only:
        destination = args.output_root.resolve()
        if (not destination.is_relative_to(pathlib.Path("/build/q6-full-warp").resolve())
                or destination.is_relative_to(source_root)):
            raise ValueError("generated output must stay under /build/q6-full-warp outside source")
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / HEADER
        if path.exists():
            if path.read_bytes() != generated:
                raise ValueError("refusing a different existing generated header")
        else:
            with path.open("xb") as handle:
                handle.write(generated)
        metadata["generated_path"] = str(path)
    if args.diff:
        sys.stdout.writelines(difflib.unified_diff(
            original.decode().splitlines(keepends=True), generated.decode().splitlines(keepends=True),
            fromfile="native/ggml_bridge_vec.cuh", tofile=HEADER))
    metadata["check_only"] = args.check_only
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
