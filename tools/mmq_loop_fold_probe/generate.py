import argparse
import hashlib
import json
import pathlib
import re
import sys

import proof


CUDA = "third_party/ggml-quants/src/ggml-cuda/"
SWITCH = "NINFER_LAB_MMQ_K128_LOOP_FOLD"
PINS = {
    CUDA + "mmq.cuh": "493ce59f8be4ff269fd0a2a8386f9d1e9a80bd2ee9c7478083d36bc2ac9ccf45",
    CUDA + "mmq-load-tiles.cuh": "80a7e02c1b2019db4ebdc1ef398a405c0249174dcf13a3d74b5d29af0347e3cf",
    CUDA + "mmq-vec-dot.cuh": "e47f3be4edcf96341b9ad48df24bd87480051c50bf7cabeef251abcb6ae39c66",
    CUDA + "mma.cuh": "08a18ccf4400990e9b2a651fdf702f49f0a521e427f151443e4359c912ac4fab",
    CUDA + "common.cuh": "6010078d639db19b1e8a2f78c4661ad240659b7bfa5a22864e315a2cbbc8da9b",
    CUDA + "mmq-config-ampere.cuh": "6de9336a6c2f65c0ad83cbe9303dcd03e8cbb6185d15346133a31421000f6408",
    "third_party/ggml-quants/src/ggml-common.h": "0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c",
    "third_party/ggml-quants/include/ggml.h": "12ee71f99db7db9b353bc02b1fbb57c344ee17c01ac5fb7952b41a637a747ea9",
}
LOOP_START = """        for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        if constexpr (child_cta) { ggml_cuda_mmq_load_tiles_q4_K<type, J, fallback, 64>(x, tile_x, offset_x + kb0 + child_row_offset*stride_row_x, tile_x_max_i-child_row_offset, stride_row_x); }
        else { load_tiles(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x); }
"""
LOOP_END = "#if defined(NINFER_GGUF_Q4_MMQ_ASYNC_X)\n    }\n#endif"
FIRST_OFFSET = "y + ncols_y * (kb0 * qk / ne_block) * sz"
SECOND_OFFSET = "y + ncols_y * ((kb0 * qk / ne_block) * sz + sz)"
FIRST_DOT = """        if constexpr (child_cta) { ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma<type, J, fallback, true>(tile_x, tile_y, sum, 0); }
        else { vec_dot(tile_x, tile_y, sum, 0); }"""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(text, newline):
    return text.replace("\n", newline.decode()).encode()


def span(data, first, last, newline):
    start, end = encoded(first, newline), encoded(last, newline)
    require(data.count(start) == 1, "source start is not unique")
    begin = data.index(start)
    stop = data.find(end, begin + len(start))
    require(stop >= 0, "source end is missing")
    return begin, stop, data[begin:stop]


def prepare(root):
    originals = {}
    for name, digest in PINS.items():
        raw = (root / name).read_bytes()
        require(sha(raw) == digest, "source pin changed: " + name)
        originals[name] = raw
    native = originals[CUDA + "mmq.cuh"]
    newline = b"\r\n" if b"\r\n" in native else b"\n"
    require(SWITCH.encode() not in native, "candidate already present in native source")
    begin, end, loop = span(native, LOOP_START, LOOP_END, newline)
    tail = encoded("        }\n", newline)
    require(loop.endswith(tail), "native K loop closing brace differs")
    body = loop[len(encoded(LOOP_START, newline)):-len(tail)]
    offset2 = encoded(SECOND_OFFSET, newline)
    marker2 = encoded("        {\n            const int * by0 = ", newline) + offset2
    require(body.count(marker2) == 1, "native second half is ambiguous")
    split = body.index(marker2)
    half0, half1 = body[:split], body[split:]
    normalized_half1 = half1.replace(offset2, encoded(FIRST_OFFSET, newline))
    normalized_half1 = normalized_half1.replace(b"sum, MMQ_TILE_NE_K)", b"sum, 0)")
    require(half0 == normalized_half1 + newline, "native K128 halves differ beyond source offsets and separator")
    require(half0.count(encoded(FIRST_DOT, newline)) == 1, "native dot call differs")
    folded_half = half0.replace(encoded(FIRST_OFFSET, newline),
        encoded("y + ncols_y * ((kb0 * qk / ne_block) * sz + phase * sz)", newline))
    folded_half = folded_half.replace(encoded(FIRST_DOT, newline),
        encoded("        vec_dot(tile_x, tile_y, sum, phase * MMQ_TILE_NE_K);", newline))
    require(folded_half.count(b"phase") == 2, "phase reached an unintended source expression")
    prefix = """#if NINFER_LAB_MMQ_K128_LOOP_FOLD
        if constexpr (type == GGML_TYPE_Q4_K && J == 128 && !fallback && !fixup &&
                      !child_cta && !bf16_gate_product && prec_src1 == GGML_PREC_Q8) {
            static_assert(I == 128 && nwarps == 8 && blocks_per_iter == 1);
            for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
                load_tiles(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
#pragma unroll 1
                for (int phase = 0; phase < 2; ++phase) {
"""
    suffix = """                }
            }
        } else
#endif
"""
    addition = encoded(prefix, newline) + folded_half + encoded(suffix, newline)
    candidate = native[:begin] + addition + native[begin:]
    require(candidate.replace(addition, b"", 1) == native, "transform failed byte-exact reversal")
    require(candidate[begin + len(addition):begin + len(addition) + len(loop)] == loop,
            "native fallback or fixup K loop changed")
    marker = "static __global__ void mul_mat_q("
    require(native.count(marker.encode()) == 1, "native kernel declaration is ambiguous")
    signature_start = native.index(marker.encode()) + len(marker)
    signature_end = native.index(b") {", signature_start)
    signature = native[signature_start:signature_end].decode().replace("\r\n", "\n")
    require(signature.count(",") == 24, "native kernel parameter count changed")
    experiments = sorted(set(re.findall(rb"\bNINFER_[A-Z0-9_]+", b"\n".join(originals.values()))))
    forbidden = [name.decode() for name in experiments if name != b"NINFER_GGUF_MMQ_STORE_FUSION"]
    variant = "#ifndef " + SWITCH + "\n#define " + SWITCH + " 0\n#endif\n\n"
    variant += "#if " + SWITCH + " != 0 && " + SWITCH + " != 1\n#error Invalid_loop_fold_switch\n#endif\n"
    variant += "#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 860\n#error Loop_fold_requires_SM86\n#endif\n"
    for macro in forbidden:
        variant += "#if defined(" + macro + ")\n#error Unrelated_experiment_must_be_undefined\n#endif\n"
    variant += "#define NINFER_GGUF_MMQ_STORE_FUSION 1\n#include \"ggml-cuda/mmq.cuh\"\n\n"
    variant += "static_assert(sizeof(block_q4_K) == 144 && sizeof(block_q8_1_mmq) == 144);\n"
    variant += "#if defined(__CUDA_ARCH__)\n"
    variant += "static_assert(ggml_cuda_mmq_get_sram_stride(GGML_TYPE_Q4_K, 128, false) == 76);\n"
    variant += "static_assert(ggml_cuda_mmq_get_I(GGML_TYPE_Q4_K, 128, false) == 128);\n"
    variant += "static_assert(ggml_cuda_mmq_get_nthreads(GGML_TYPE_Q4_K, 128, false) == 256);\n"
    variant += "static_assert(ggml_cuda_mmq_get_occupancy(GGML_TYPE_Q4_K, 128, false) == 1);\n"
    variant += "static_assert(ggml_cuda_mmq_get_rows_per_warp(GGML_TYPE_Q4_K, 128, false) == 32);\n#endif\n"
    for store, accumulate in ((False, False), (True, False), (True, True)):
        values = str(store).lower() + ", " + str(accumulate).lower()
        variant += "template __global__ void mul_mat_q<GGML_TYPE_Q4_K, 128, false, GGML_PREC_Q8, "
        variant += values + ", false, false>(" + signature + ");\n"
    files = {"on/ggml-cuda/mmq.cuh": candidate, "off/ggml-cuda/mmq.cuh": native,
             "variant.cu": variant.encode()}
    for name in ("mmq-load-tiles.cuh", "mmq-vec-dot.cuh"):
        files["on/ggml-cuda/" + name] = originals[CUDA + name]
    cpu = proof.run()
    receipt = {"switch": SWITCH, "default": 0, "scope": proof.SCOPE,
               "source_sha256": PINS, "generated_sha256": {name: sha(data) for name, data in files.items()},
               "native_loop": {"byte_begin": begin, "byte_end": end, "sha256": sha(loop)},
               "native_K128_half_sha256": sha(half0), "addition_sha256": sha(addition),
               "OFF_header_byte_exact": files["off/ggml-cuda/mmq.cuh"] == native,
               "ON_reversal_byte_exact": True, "native_dot_and_load_headers_unchanged": True,
               "phase_only_changes_shared_and_global_source_offsets": True,
               "source_fixup_branch_unchanged": True, "cpu_proof": cpu,
               "host_harness": False, "GPU_or_compile_run": False}
    return files, receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=pathlib.Path)
    args = parser.parse_args()
    _, result = prepare(args.source_root.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
