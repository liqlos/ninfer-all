import argparse
import hashlib
import json
import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
VECTOR = "src/ops/linear/gguf/ggml_bridge_vec.cuh"
MMQ = "third_party/ggml-quants/src/ggml-cuda/mmq.cuh"
PINS = {
    VECTOR: "0a0c82d6cede1707f91044efa70aeb39a536eeecc53b479b5c976e92c82e6596",
    "src/ops/linear/gguf/ggml_bridge_internal.cuh":
        "8b78c42a2cb01b10e1227987cae22b3d0a030684070d6f35a213b033fb3c9d90",
    "src/ops/linear/gguf/ggml_bridge.h":
        "63dc7967143129a4e454824d0176cfbf9e6c8502b4b88423acdd291e74856b76",
    "src/ops/linear/gguf/ggml_bridge_mmq.cuh":
        "2513cb559034db1422841508c1c251e060ec087fe7b92643b2b539f687a7b15b",
    "src/ops/common/memory.cuh":
        "68d4ce555d86761aa4c21d49726be2e48d82fcbdc4f6dbe4fc8b7356f945e205",
    "src/ops/common/mma.cuh":
        "4f3f05cfb7ecce39a650cf53588587997a9e8bee4faf992e23564cf8c696d94d",
    "third_party/ggml-quants/src/ggml-cuda/common.cuh":
        "6010078d639db19b1e8a2f78c4661ad240659b7bfa5a22864e315a2cbbc8da9b",
    "third_party/ggml-quants/src/ggml-cuda/vecdotq.cuh":
        "a7fd4281ad9f0a6d9373c776325411d2522e0b62ffae2c16e91e667a1af251fe",
    MMQ: "493ce59f8be4ff269fd0a2a8386f9d1e9a80bd2ee9c7478083d36bc2ac9ccf45",
    "third_party/ggml-quants/src/ggml-cuda/mmq-load-tiles.cuh":
        "80a7e02c1b2019db4ebdc1ef398a405c0249174dcf13a3d74b5d29af0347e3cf",
    "third_party/ggml-quants/src/ggml-cuda/mmq-vec-dot.cuh":
        "e47f3be4edcf96341b9ad48df24bd87480051c50bf7cabeef251abcb6ae39c66",
    "third_party/ggml-quants/src/ggml-cuda/mmq-config-ampere.cuh":
        "6de9336a6c2f65c0ad83cbe9303dcd03e8cbb6185d15346133a31421000f6408",
    "third_party/ggml-quants/src/ggml-cuda/mma.cuh":
        "08a18ccf4400990e9b2a651fdf702f49f0a521e427f151443e4359c912ac4fab",
}


def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def bounded(text, first, following):
    if text.count(first) != 1 or text.count(following) != 1:
        raise ValueError("source anchors are not unique")
    begin, end = text.index(first), text.index(following)
    if begin >= end:
        raise ValueError("source anchors are reversed")
    return text[begin:end]


class Transform:
    def __init__(self, text):
        self.original = text
        self.text = text
        self.edits = []

    def replace(self, old, new, count=1):
        if old == new or self.text.count(old) != count or new in self.text:
            raise ValueError(f"replacement shape mismatch: {old[:100]!r}")
        before = self.text
        self.text = before.replace(old, new)
        if self.text.replace(new, old) != before:
            raise ValueError("replacement is not reversible")
        self.edits.append((old, new, count))

    def receipt(self):
        restored = self.text
        for old, new, count in reversed(self.edits):
            if restored.count(new) != count:
                raise ValueError("reverse substitution count changed")
            restored = restored.replace(new, old)
        if restored != self.original:
            raise ValueError("unlisted source change")
        return {
            "normalized_native_sha256": sha(self.original),
            "generated_region_sha256": sha(self.text),
            "only_listed_reversible_substitutions": True,
            "substitutions": [
                {"before": old, "after": new, "count": count}
                for old, new, count in self.edits
            ],
        }


def prepare(root):
    sources = {}
    for name, expected in PINS.items():
        raw = (root / name).read_bytes()
        if sha(raw) != expected:
            raise ValueError("pinned source changed: " + name)
        sources[name] = raw.decode().replace("\r\n", "\n")
    vector = Transform(bounded(
        sources[VECTOR],
        "template <ggml_type type, int T, bool Fused, int KS, bool Prefetch2 = true,",
        "template <ggml_type type, int T, bool Fused, int KS>\nvoid launch_baseline("))
    vector.replace(
        "__launch_bounds__(kVecWarps * 32) __global__ void kernel(VecArgs p) {",
        "__launch_bounds__(kVecWarps * 32) __global__ void q6_word_group_vector_kernel("
        "VecArgs p, Q6WGView operand) {\n"
        "    static_assert(type == GGML_TYPE_Q6_K && (T == 1 || T == 4) && !Fused &&\n"
        "                  KS == 0 && Prefetch2 && !Baseline && Rows == 2);")
    vector.replace(
        "    const std::int64_t lane_offset = std::int64_t(lane / kPerBlock) * D::kBlockBytes;",
        "    const std::uint64_t lane_block = lane / kPerBlock;")
    row_setup = bounded(vector.text, "        const std::uint8_t* rowp[kRows];",
                        "        float acc[kRows][T];")
    vector.replace(
        row_setup,
        "        std::uint64_t row_block[kRows];\n"
        "#pragma unroll\n"
        "        for (int r = 0; r < kRows; ++r) {\n"
        "            row_block[r] = (std::uint64_t(operand.row_begin) +\n"
        "                min(row0 + r, p.rows - 1)) * operand.parent_B + lane_block;\n"
        "        }\n")
    vector.replace(
        "D::decode(rowp[r] + std::int64_t(32 / kPerBlock) * it * D::kBlockBytes, sub, table, sl);",
        "q6_word_group_decode(operand, row_block[r] + std::uint64_t(32 / kPerBlock) * it, sub, sl);")
    process = Transform(bounded(
        sources[MMQ],
        "template <ggml_type type, int J, bool fallback, bool fixup, ggml_prec prec_src1 = GGML_PREC_Q8,",
        '// The mul_mat_q kernel implements "stream-k"'))
    process.replace("void mul_mat_q_process_tile(", "void q6_word_group_process_tile(")
    process.replace(
        "        const char * __restrict__ x, const int offset_x, const int * __restrict__ y,",
        "        Q6WGView operand, const int offset_x, const int * __restrict__ y,")
    process.replace(
        "        const int gate_stride) {\n",
        "        const int gate_stride) {\n"
        "    static_assert(type == GGML_TYPE_Q6_K && J == 128 && !fallback &&\n"
        "                  prec_src1 == GGML_PREC_Q8 && !bf16_gate_product && !child_cta);\n"
        "    const char* x = reinterpret_cast<const char*>(operand.base);\n")
    process.replace(
        "load_tiles(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);",
        "q6_word_group_load_tiles<type, J, fallback>(operand, tile_x, offset_x + kb0, "
        "tile_x_max_i, stride_row_x);", 2)
    kernel = Transform(bounded(
        sources[MMQ],
        "template <ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8,\n"
        "          bool bf16_store = false, bool bf16_accumulate = false, bool bf16_gate_product = false, bool child_cta = false>\n"
        "__launch_bounds__",
        "template <ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8>\n"
        "__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, fallback, prec_src1)/2, 1)\n"
        "static __global__ void mul_mat_q_stream_k_fixup("))
    kernel.replace("void mul_mat_q(", "void q6_word_group_mul_mat_q(")
    kernel.replace(
        "        const char * __restrict__ x, const int * __restrict__ y, const int32_t * __restrict__ ids_dst,",
        "        Q6WGView operand, const int * __restrict__ y, const int32_t * __restrict__ ids_dst,")
    kernel.replace("mul_mat_q_process_tile<", "q6_word_group_process_tile<", 3)
    kernel.replace("(x, offset_x, y + offset_y, ids_dst_shared, tile_dst,",
                   "(operand, offset_x, y + offset_y, ids_dst_shared, tile_dst,", 3)
    flags = set()
    for source in sources.values():
        flags.update(re.findall(r"NINFER_GGUF_[A-Z0-9_]+", source))
    flags.discard("NINFER_GGUF_Q6_COOPERATIVE_DECODE")
    guard = (
        "#pragma once\n#include \"probe_api.h\"\n"
        "#if NINFER_Q6_WORD_GROUP_PROBE != 0 && NINFER_Q6_WORD_GROUP_PROBE != 1\n"
        "#error Invalid_candidate_switch\n#endif\n"
        "#if !defined(NINFER_GGUF_Q6_COOPERATIVE_DECODE) || NINFER_GGUF_Q6_COOPERATIVE_DECODE != 1\n"
        "#error Native_Q6_cooperative_decode_control_is_required\n#endif\n"
        "#if " + " || \\\n    ".join(f"defined({flag})" for flag in sorted(flags)) +
        "\n#error Unrelated_experiments_must_be_undefined\n#endif\n"
        "#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 860\n"
        "#error This_probe_requires_sm86\n#endif\n")
    vector_header = (
        '#pragma once\n#include "q6_word_group_guards.cuh"\n'
        '#include "ops/linear/gguf/ggml_bridge_vec.cuh"\n'
        '#if NINFER_Q6_WORD_GROUP_PROBE\n#include "vector_decode.cuh"\n'
        "namespace ninfer::ops::gguf::detail::vec {\n" + vector.text + "}\n#endif\n")
    mmq_header = (
        '#pragma once\n#include "q6_word_group_guards.cuh"\n'
        '#include "ops/linear/gguf/ggml_bridge_mmq.cuh"\n'
        '#if NINFER_Q6_WORD_GROUP_PROBE\n#include "mmq_loader.cuh"\n' +
        process.text + kernel.text + "#endif\n")
    files = {
        "q6_word_group_guards.cuh": guard,
        "q6_word_group_vector.cuh": vector_header,
        "q6_word_group_mmq.cuh": mmq_header,
    }
    receipt = {
        "schema": 1,
        "candidate": "q6-word-group-slab16",
        "default_enabled": False,
        "source_pins": PINS,
        "generated_sha256": {name: sha(text) for name, text in files.items()},
        "transforms": {
            "vector_kernel": vector.receipt(),
            "mmq_process": process.receipt(),
            "mmq_stream_k_kernel": kernel.receipt(),
        },
        "unchanged": [
            "vector Q8 loads, native Q6 dot, runtime K loop, reduction, epilogue",
            "MMQ activation reads, vec_dot, K order, barriers, shared extents, epilogues",
            "MMQ Stream-K scheduling, fixup kernel and FP32 fixup order",
            "native scale values, literal half d bits, unsigned code minus 32",
        ],
        "host_link_requires": ["ninfer::ops::gguf::detail::device_facts()",
                               "vendored ggml support symbols required by native headers"],
        "qualified": {"cpu": False, "cuda_compile": False, "gpu": False,
                      "numerical_outputs": False, "performance": False},
    }
    return files, receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--output-dir", type=pathlib.Path, default=HERE / "generated")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    files, receipt = prepare(args.source_root.resolve())
    if not args.check_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (args.output_dir / name).write_bytes(text.encode())
        (args.output_dir / "source_receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    else:
        for name, text in files.items():
            path = args.output_dir / name
            if path.exists() and path.read_bytes() != text.encode():
                raise ValueError("generated header is stale: " + str(path))
    print(json.dumps({"source_check": "pass", "pinned_sources": len(PINS),
                      "generated_headers": list(files), "cuda_compile": "not_run",
                      "gpu": "not_run"}, separators=(",", ":")))


if __name__ == "__main__":
    main()
