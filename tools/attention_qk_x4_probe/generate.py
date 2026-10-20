import argparse
import hashlib
import json
import pathlib
import re
import sys


SOURCE = "src/ops/softmax_attention/dense/causal_cache/prompt_i8_fast.cuh"
GENERATED = "attention_qk_x4_generated.cuh"
SWITCH = "NINFER_PROMPT_QK_B_X4"
PINS = {
    SOURCE: "f7c887c6c0dec7de242822b1ab831eecd8712761510d1bc6e83a27b847955f6b",
    "src/ops/softmax_attention/dense/causal_cache/prompt_common.cuh":
        "0749aa221b829b53de02ae7a62ccb0b72ea8e1d4ca767fdf1773b2554b0978ce",
    "src/ops/common/mma.cuh":
        "4f3f05cfb7ecce39a650cf53588587997a9e8bee4faf992e23564cf8c696d94d",
    "src/ops/common/memory.cuh":
        "68d4ce555d86761aa4c21d49726be2e48d82fcbdc4f6dbe4fc8b7356f945e205",
    "src/ops/softmax_attention/dense/causal_cache/geometry.cuh":
        "cccf46c4e0704cfa5893a45f9bb9a0e51b7f300c921b873046671f9e44788368",
    "src/ops/softmax_attention/common/head_mapping.cuh":
        "c11948e0e5bf38caceb8b7681e1c73bfb1871bca4761a331d57fe741237bdcdc",
    "src/ops/kernel/paged_kv_address.cuh":
        "df4463d9d3c15caeb18fa511abdc982dd9142828d958282299fe1c0e44cae77e",
    "src/ops/softmax_attention/dense/causal_cache/prompt.cu":
        "02fa6e5675c9b8097de366027ec43b5acdce37aa86cf5a038cab240f2a09cdaa",
    "src/ops/softmax_attention/dense/causal_cache/launch.h":
        "702aacf8e88138acb0ce96ceb171329ccd23854958119ea5fd8dfa50a01a00ee",
    "src/ops/kv_cache/int8_g64_codec.cuh":
        "5acb97a519b740905d6ab4e475d086922479c30eb530ac396fadb9a76a60e62a",
    "src/core/paged_kv_cache.h":
        "8e7e015ce06c7354d33ae22387b7fa061337046ba49d9c6e219eb1e08d0e1f06",
    "src/core/paged_kv_storage.h":
        "a9bb8920c61cc66ac08774e8bb8cda9fb03b0e9ea0e3a9886ea2ad0c31242025",
    "src/core/tensor.h":
        "99bdbd15aab32e189165b98ede5efd94e2c900c20ce1313acb41b4173b0f188e",
}
NATIVE_LOOP = """#pragma unroll
                for (int kk = 0; kk < GroupKc; ++kk) {
                    const int brow = nt * 8 + a_rin;
                    const int bcol = (grp * GroupKc + kk) * 16 + b_koff;
                    unsigned bf[2];
                    ldmatrix_x2(bf[0], bf[1],
                                smem_addr(&k_b16[brow * DB16 + causal_prompt_swz(brow, bcol)]));
                    mma_s8(c0, c1, c2, c3, af[kk][0], af[kk][1], af[kk][2], af[kk][3], bf[0],
                           bf[1]);
                }
"""
ENABLED_PREFIX = """#if NINFER_PROMPT_QK_B_X4
                if constexpr (D == 256 && Warps == 8 && !PackedValues &&
                              Keys == KvKeyCoding::Int8 &&
                              std::is_same_v<Geometry, CausalD256H24Kv4> &&
                              std::is_same_v<Metadata, PagedKVBatchMetadata<false>>) {
                    const int brow = nt * 8 + a_rin;
                    const int bcol = (grp * GroupKc + (lane >> 4)) * 16 + b_koff;
                    unsigned bf[4];
                    ldmatrix_x4(bf[0], bf[1], bf[2], bf[3],
                                smem_addr(&k_b16[brow * DB16 + causal_prompt_swz(brow, bcol)]));
#pragma unroll
                    for (int kk = 0; kk < GroupKc; ++kk) {
                        mma_s8(c0, c1, c2, c3, af[kk][0], af[kk][1], af[kk][2], af[kk][3],
                               bf[2 * kk], bf[2 * kk + 1]);
                    }
                } else {
#endif
"""
ENABLED_SUFFIX = """#if NINFER_PROMPT_QK_B_X4
                }
#endif
"""
PROVENANCE = """#define NINFER_ATTENTION_QK_X4_GENERATED 1
#ifndef NINFER_PROMPT_QK_B_X4
#define NINFER_PROMPT_QK_B_X4 0
#endif
#if NINFER_PROMPT_QK_B_X4 != 0 && NINFER_PROMPT_QK_B_X4 != 1
#error NINFER_PROMPT_QK_B_X4_must_be_zero_or_one
#endif
"""


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(data, old, new):
    require(data.count(old) == 1, "expected one exact source fragment")
    return data.replace(old, new, 1)


def prepare(source_root):
    inputs = {}
    for path, expected in PINS.items():
        inputs[path] = (source_root / path).read_bytes()
        require(sha256(inputs[path]) == expected, "source pin mismatch: " + path)
    original = inputs[SOURCE]
    newline = b"\r\n" if b"\r\n" in original else b"\n"
    require(newline == b"\n" or b"\n" not in original.replace(b"\r\n", b""),
            "mixed source newlines")

    def encode(text):
        return text.encode("utf-8").replace(b"\n", newline)

    replacements = (
        (encode("#pragma once\n"), encode("#pragma once\n" + PROVENANCE)),
        (encode("#include <cstdint>\n"), encode("#include <cstdint>\n#include <type_traits>\n")),
        (encode(NATIVE_LOOP), encode(ENABLED_PREFIX + NATIVE_LOOP + ENABLED_SUFFIX)),
    )
    generated = original
    for old, new in replacements:
        generated = replace_once(generated, old, new)
    restored = generated
    for old, new in reversed(replacements):
        restored = replace_once(restored, new, old)
    require(restored == original, "inverse transform is not byte exact")
    disabled = replace_once(generated, encode(ENABLED_PREFIX), b"")
    disabled = replace_once(disabled, encode(ENABLED_SUFFIX), b"")
    disabled = replace_once(disabled, encode(PROVENANCE), b"")
    disabled = replace_once(disabled, encode("#include <type_traits>\n"), b"")
    require(disabled == original, "OFF projection differs from pinned native source")
    native_mma = re.findall(r"mma_s8\([^;]+;", NATIVE_LOOP)
    candidate_mma = re.findall(r"mma_s8\([^;]+;", ENABLED_PREFIX)
    require(len(native_mma) == len(candidate_mma) == 1, "MMA source call is not unique")
    restored_mma = candidate_mma[0].replace("bf[2 * kk + 1]", "bf[1]").replace(
        "bf[2 * kk]", "bf[0]")
    require(re.sub(r"\s+", "", restored_mma) == re.sub(r"\s+", "", native_mma[0]),
            "candidate changes native MMA operands or accumulator order")
    kk_loop = "for (int kk = 0; kk < GroupKc; ++kk)"
    require(NATIVE_LOOP.count(kk_loop) == ENABLED_PREFIX.count(kk_loop) == 1,
            "candidate changes native kk accumulation order")
    scoped = "\n".join(line for line in
                       (ENABLED_PREFIX + NATIVE_LOOP + ENABLED_SUFFIX).splitlines()
                       if not line.lstrip().startswith("#"))
    depth = 0
    for token in scoped:
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
            require(depth >= 0, "ON replacement closes an outer native scope")
    require(depth == 0, "ON replacement leaks a scope into the FP32 suffix")
    require(ENABLED_PREFIX.rstrip().endswith("} else {\n#endif"),
            "native fallback must have an explicit compound statement")
    following = "                const int keya  = nt * 8 + 2 * lid;\n"
    require(generated.count(encode(ENABLED_PREFIX + NATIVE_LOOP + ENABLED_SUFFIX + following)) == 1,
            "keya declaration is not immediately after the closed replacement scope")
    require(generated.replace(encode(ENABLED_PREFIX), b"").count(b"mma_s8(") ==
            original.count(b"mma_s8("), "native MMA sites changed")
    for token in (b"__fmaf_rn(", b"__syncthreads(", b"cp_async", b"causal_prompt_i8_fast_mma_f16_acc(",
                  b"causal_prompt_i8_fast_decode_v_pair(", b"tile_shift", b"__float2half"):
        require(generated.count(token) == original.count(token), "non-QK-load source changed")
    return generated, {
        "source_pins": PINS,
        "generated_sha256": sha256(generated),
        "inverse_byte_exact": True,
        "off_projection_byte_exact": True,
        "native_arithmetic_and_pv_source_unchanged": True,
        "mma_call_exact_after_proven_b_register_substitution": True,
        "kk_accumulation_loop_byte_identical": True,
        "fallback_scope_closed_before_native_keya": True,
        "exact_replacement_sites": len(replacements),
        "generated_include": GENERATED,
        "sibling_include_strategy": "unique direct include; native dependencies stay native",
    }


def swizzle(row, column):
    return (((column >> 3) ^ (row & 7)) << 3) | (column & 7)


def row_address(stage, nt, grp, lane, kk):
    row = nt * 8 + (lane & 7)
    column = (grp * 2 + kk) * 16 + (((lane >> 3) & 1) << 3)
    return 32768 + stage * 33792 + 2 * (row * 128 + swizzle(row, column))


def fragment_addresses(addresses, matrix, lane):
    first = addresses[8 * matrix + lane // 4] + 4 * (lane % 4)
    return tuple(first + byte for byte in range(4))


def fragment_proof():
    memory = {}
    for stage in range(2):
        for row in range(64):
            for dimension in range(256):
                address = 32768 + stage * 33792 + 2 * (
                    row * 128 + swizzle(row, dimension // 2)) + dimension % 2
                require(address not in memory, "native K staging overlaps")
                require(address < 100352, "K staging exceeds unchanged shared arena")
                memory[address] = (stage, row, dimension)
    compared_registers = 0
    compared_bytes = 0
    checked_row_addresses = 0
    naive_x4_mismatches = 0
    swapped_pair_mismatches = 0
    observed = set()
    for stage in range(2):
        for grp in range(4):
            for nt in range(8):
                native = [[row_address(stage, nt, grp, lane, kk)
                           for lane in range(32)] for kk in range(2)]
                fused = [row_address(stage, nt, grp, lane, lane >> 4)
                         for lane in range(32)]
                for lane in range(32):
                    require(fused[lane] == native[lane >> 4][lane & 15],
                            "x4 row provider differs from corresponding x2 provider")
                    require(fused[lane] % 16 == 0, "ldmatrix row alignment changed")
                    require(fused[lane] + 15 in memory, "ldmatrix row escapes K staging")
                    checked_row_addresses += 1
                    for matrix in range(4):
                        old = fragment_addresses(native[matrix // 2], matrix % 2, lane)
                        new = fragment_addresses(fused, matrix, lane)
                        expected = tuple((stage, nt * 8 + lane // 4,
                                          grp * 64 + matrix * 16 + 4 * (lane % 4) + byte)
                                         for byte in range(4))
                        require(old == new, "x4 register differs from concatenated x2 fragments")
                        require(tuple(memory[address] for address in new) == expected,
                                "loaded bytes differ from independent logical K coordinates")
                        naive_x4_mismatches += fragment_addresses(
                            native[0], matrix, lane) != old
                        swapped_pair_mismatches += fragment_addresses(
                            fused, matrix ^ 2, lane) != old
                        observed.update(new)
                        compared_registers += 1
                        compared_bytes += 4
    require(observed == set(memory), "fragments do not cover both complete K stages")
    require(naive_x4_mismatches == 4096, "naive x2-to-x4 negative control escaped")
    require(swapped_pair_mismatches == 8192, "MMA-pair permutation negative control escaped")
    native_mma = tuple((grp, nt, kk, (2 * kk, 2 * kk + 1))
                       for grp in range(4) for nt in range(8) for kk in range(2))
    require(len(native_mma) == 64, "native per-page IMMA count changed")
    return {
        "status": "PASS_CPU_SYMBOLIC_BYTE_ADDRESS_AND_FRAGMENT_PROOF",
        "ptx_reference": "https://docs.nvidia.com/cuda/parallel-thread-execution/"
                         "index.html#warp-level-matrix-instructions-ldmatrix",
        "stages": 2,
        "groups": 4,
        "n8_tiles": 8,
        "lanes": 32,
        "row_addresses": checked_row_addresses,
        "register_fragments": compared_registers,
        "symbolic_bytes": compared_bytes,
        "unique_staged_k_bytes": len(observed),
        "negative_controls_rejected": {
            "naive_x4_without_upper_lane_address_change": naive_x4_mismatches,
            "swapped_mma_register_pairs": swapped_pair_mismatches,
        },
        "quantification": "all byte values; each byte represented by its unique source coordinate",
        "register_mapping": "x4[0:2] = x2(kk=0); x4[2:4] = x2(kk=1)",
        "mma_order": "grp then nt then kk=0,1; same A and C registers and exact B bytes",
        "mma_per_warp_live_page": len(native_mma),
        "qk_b_loads_per_warp_live_page_off": 64,
        "qk_b_loads_per_warp_live_page_on": 32,
        "fp32_order": "unchanged donor suffix after both IMMA calls in each group",
        "proof_limit": "PTX data movement and source order; no CUDA compiler or GPU execution",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parents[2])
    parser.add_argument("--output-root", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    generated, result = prepare(args.source_root.resolve())
    result.update({
        "switch": SWITCH,
        "default": 0,
        "geometry": "CausalD256H24Kv4 = CausalAttentionGeometry<24,4,1>",
        "metadata": "PagedKVBatchMetadata<false>",
        "cache_storage": "Int8Group64",
        "warps": 8,
        "packed_values": False,
        "key_coding": "KvKeyCoding::Int8",
        "threads": 256,
        "query_rows_per_cta": 128,
        "shared_bytes": 100352,
        "extra_workspace_bytes": 0,
        "register_lifetime_change": "two extra B words before first IMMA; second pair live across it",
        "compiler_gate": {
            "toolchain": "CUDA 13.1.115",
            "architecture": "sm_86",
            "no_increase": ["stack", "local", "spill stores", "spill loads"],
            "unchanged": ["100352B dynamic shared", "IMMA", "PV", "FP32 rounding/order"],
            "required_replacement": "one non-transposed LDSM.x4 for each pair of QK LDSM.x2",
        },
        "codegen_qualified": False,
        "gpu_correctness_qualified": False,
        "performance_claim": False,
    })
    if args.self_test:
        result["cpu_proof"] = fragment_proof()
    if args.output_root is not None:
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / GENERATED).write_bytes(generated)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
