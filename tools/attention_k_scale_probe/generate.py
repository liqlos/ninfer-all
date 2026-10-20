import argparse
import collections
import hashlib
import json
import pathlib
import re
import struct
import sys


SOURCE = "src/ops/softmax_attention/dense/causal_cache/prompt_i8_fast.cuh"
GENERATED = "attention_k_scale_generated.cuh"
SWITCH = "NINFER_PROMPT_K_SCALE_FP32"
PINS = {
    SOURCE: "f7c887c6c0dec7de242822b1ab831eecd8712761510d1bc6e83a27b847955f6b",
    "src/ops/softmax_attention/dense/causal_cache/prompt_common.cuh":
        "0749aa221b829b53de02ae7a62ccb0b72ea8e1d4ca767fdf1773b2554b0978ce",
    "src/ops/common/mma.cuh":
        "4f3f05cfb7ecce39a650cf53588587997a9e8bee4faf992e23564cf8c696d94d",
    "src/ops/common/memory.cuh":
        "68d4ce555d86761aa4c21d49726be2e48d82fcbdc4f6dbe4fc8b7356f945e205",
    "src/ops/common/warp.cuh":
        "7555cb64c0bfd8738209fc25ebe5972f4cb032fee03ea4fc769770da4ec2d839",
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
    "include/ninfer/ops/attention_geometry.h":
        "32cc81e045773e2b3119f2971030ad4878a8090a1b6556cb6fb45ad2230249f4",
    "include/ninfer/ops/kv_cache_append.h":
        "65e69bfe36ecfecb381de42b017cefb8063aab75bd9f64fffce48d603fd401c9",
    "include/ninfer/ops/softmax_attention.h":
        "9ee1fd2636b28274d0f0e12049ec41545702eabd64446df6290889f65765ad94",
    "include/ninfer/types.h":
        "b8f831d75158a530d8a9b27195ff91ba3dc1c7e406ff00a66b106b0c37f43975",
    "src/core/arena.h":
        "869c06ffe4afe0bc88c8a9dcfdfd8f296ddd3202867a6d021aacc3ee3577e21e",
    "src/core/cyclic_kv_cache.h":
        "1048ed7a1395691936bbe20350880dff26555342679244b74fd51be787921ed3",
    "src/core/device.h":
        "f4b87ac4b493a58d29970ab233f69d0ddc5a2a6f5a8526893cd16f7315cabbd0",
    "src/core/dtype.h":
        "13ab4e833debe24baf9294a9d8eefc3f61a182b914e8a128d1725be3d93574ed",
    "src/core/layout.h":
        "2fa3dbde6e507d5940c1ea9cff08606900cb71ab27ca888a8b5b66ea9235e400",
    "src/ops/common/device_route.h":
        "3e8f073d5f51a495bfd3eb9686a917182ea2a85541858d54bb900ebf24da9bbe",
    "src/ops/common/math.cuh":
        "339939e6189695afdc027f4541f816c014061c4a37965550cbba72f96a2ad2c1",
    "src/ops/common/math.h":
        "2f61fe7900d5584e700c824add008fe5c1d21e7d32a31e93ac9089b9f861f330",
    "src/ops/kv_cache/append/launch.h":
        "de41fcac3ddc21a2b76a0ef83741c74c8d8fb875cfe0f0a19f06185d4a7ab1d5",
    "src/ops/kv_cache/d256_profile.h":
        "e02bf4a088d46bc4b48e5612a79e52289aa1d0b2dc9e9662b1b2e52fb306b64d",
    "src/ops/kv_cache/e8_root_codec.cuh":
        "372c6b3dbf7decefe654e1e32d9ca89a89ea269a9710b27ddc73484a21d70090",
    "src/ops/kv_cache/hadamard_d256.cuh":
        "3d00813a0a3c112484ef5a12c0011ec7ee8d0848aa17fadccaf9fbdeb0b14fa7",
    "src/ops/softmax_attention/dense/causal_cache/prompt_bf16.cuh":
        "ef5583d348b11f749752065d02caf6538905f189f90650c2f9fa46af71a58b52",
    "src/ops/softmax_attention/dense/causal_cache/prompt_i8.cuh":
        "7e795dacee7c79bec61d5217b9c9082fbbd72dcf3ba3e823c08a86ea6e05682f",
}
PROVENANCE = """#define NINFER_ATTENTION_K_SCALE_GENERATED 1
#ifndef NINFER_PROMPT_K_SCALE_FP32
#define NINFER_PROMPT_K_SCALE_FP32 0
#endif
#if NINFER_PROMPT_K_SCALE_FP32 != 0 && NINFER_PROMPT_K_SCALE_FP32 != 1
#error NINFER_PROMPT_K_SCALE_FP32_must_be_zero_or_one
#endif
"""
SHARED = "    extern __shared__ __align__(16) unsigned char smem_raw[];\n"
LAYOUT = SHARED + """    constexpr bool UseFp32KScale =
        D == 256 && Warps == 8 && !PackedValues && Keys == KvKeyCoding::Int8 &&
        std::is_same_v<Geometry, CausalD256H24Kv4> &&
        std::is_same_v<Metadata, PagedKVBatchMetadata<false>>;
    using KScaleT = std::conditional_t<UseFp32KScale, float, __half>;
    constexpr int KScaleExtraBytes = UseFp32KScale ? kCausalPromptI8FastScaleBytes : 0;
"""
STAGE = "        return smem_raw + Shape::QBytes + stage * Stage::Bytes;\n"
FP32_STAGE = "        return smem_raw + Shape::QBytes + stage * (Stage::Bytes + KScaleExtraBytes);\n"
ISSUE_POINTERS = """        __half* ks_s        = reinterpret_cast<__half*>(base + Stage::KScaleOffset);
        __half* vs_s        = ks_s + Bc * Groups;
"""
FP32_ISSUE_POINTERS = """        KScaleT* ks_s       = reinterpret_cast<KScaleT*>(base + Stage::KScaleOffset);
        __half* vs_s        = reinterpret_cast<__half*>(
            base + Stage::KScaleOffset + kCausalPromptI8FastScaleBytes + KScaleExtraBytes);
"""
COPY = """            cp_async_zfill<8>(ks_s + tid * Groups,
                              cache_k_scale +
                                  kv_cache_int8_quant_scale_index<Geometry>(page, kv_head, 0, tid),
                              tid < valid ? 8 : 0);
"""
PRODUCER = """            if constexpr (UseFp32KScale) {
                uint2 raw_scales = make_uint2(0u, 0u);
                if (tid < valid) {
                    raw_scales = load_vec<uint2>(
                        cache_k_scale +
                        kv_cache_int8_quant_scale_index<Geometry>(page, kv_head, 0, tid));
                }
                const __half2 h01 = load_vec<__half2>(&raw_scales.x);
                const __half2 h23 = load_vec<__half2>(&raw_scales.y);
                const float4 expanded = make_float4(
                    __half2float(__low2half(h01)), __half2float(__high2half(h01)),
                    __half2float(__low2half(h23)), __half2float(__high2half(h23)));
                store_vec(ks_s + tid * Groups, expanded);
            } else {
""" + COPY + "            }\n"
QK_POINTERS = """        const __half* ks_s = reinterpret_cast<const __half*>(base + Stage::KScaleOffset);
        const __half* vs_s = ks_s + Bc * Groups;
"""
FP32_QK_POINTERS = """        const KScaleT* ks_s = reinterpret_cast<const KScaleT*>(base + Stage::KScaleOffset);
        const __half* vs_s = reinterpret_cast<const __half*>(
            base + Stage::KScaleOffset + kCausalPromptI8FastScaleBytes + KScaleExtraBytes);
"""
CONSUMER = """                const float ks0 = __half2float(ks_s[keya * Groups + grp]);
                const float ks1 = __half2float(ks_s[(keya + 1) * Groups + grp]);
"""
FP32_CONSUMER = """                float ks0, ks1;
                if constexpr (UseFp32KScale) {
                    ks0 = ks_s[keya * Groups + grp];
                    ks1 = ks_s[(keya + 1) * Groups + grp];
                } else {
                    ks0 = __half2float(ks_s[keya * Groups + grp]);
                    ks1 = __half2float(ks_s[(keya + 1) * Groups + grp]);
                }
"""
PV_POINTER = """        const __half* vs_s =
            reinterpret_cast<const __half*>(v_s + kCausalPromptI8FastTileBytes) + Bc * Groups;
"""
FP32_PV_POINTER = """        const __half* vs_s = reinterpret_cast<const __half*>(
            v_s + kCausalPromptI8FastTileBytes + kCausalPromptI8FastScaleBytes + KScaleExtraBytes);
"""
LOOP = """    issue_tile(0);
    commit_keys(0);
#pragma unroll 1
    for (int kb = 0; kb < key_blocks; ++kb) {
        cp_wait<0>();
        __syncthreads();
        if (kb + 1 < key_blocks) { issue_tile(kb + 1); }
        qk_softmax(kb);
        pv(kb);
        if (kb + 1 < key_blocks) { commit_keys(kb + 1); }
    }
"""
CHANGES = ((SHARED, LAYOUT), (STAGE, FP32_STAGE), (ISSUE_POINTERS, FP32_ISSUE_POINTERS),
           (COPY, PRODUCER), (QK_POINTERS, FP32_QK_POINTERS), (CONSUMER, FP32_CONSUMER),
           (PV_POINTER, FP32_PV_POINTER))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(data, old, new):
    require(data.count(old) == 1, "expected one exact source fragment: " + repr(old[:90]))
    return data.replace(old, new, 1)


def guarded(old, new):
    return "#if " + SWITCH + "\n" + new + "#else\n" + old + "#endif\n"


def project(source, enabled):
    output, active = [], [True]
    for line in source.splitlines(keepends=True):
        directive = line.strip()
        if directive == "#if " + SWITCH:
            active.append(active[-1] and enabled)
        elif directive == "#else":
            require(len(active) > 1, "unmatched feature else")
            active[-1] = active[-2] and not active[-1]
        elif directive == "#endif":
            require(len(active) > 1, "unmatched feature endif")
            active.pop()
        elif all(active):
            output.append(line)
    require(len(active) == 1, "unterminated feature conditional")
    return "".join(output)


def scope_proof(source):
    source = re.sub(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/', "", source, flags=re.DOTALL)
    depth = 0
    for token in source:
        depth += (token == "{") - (token == "}")
        require(depth >= 0, "replacement closes an outer scope")
    require(depth == 0, "replacement leaks scope")


def line_range(source, fragment):
    require(source.count(fragment) == 1, "line-range anchor not unique")
    first = source[:source.index(fragment)].count("\n") + 1
    return [first, first + fragment.count("\n") - 1]


def prepare(source_root):
    inputs = {path: (source_root / path).read_bytes() for path in PINS}
    for path, expected in PINS.items():
        require(sha256(inputs[path]) == expected, "source pin mismatch: " + path)
    original = inputs[SOURCE]
    newline = b"\r\n" if b"\r\n" in original else b"\n"
    require(newline == b"\n" or b"\n" not in original.replace(b"\r\n", b""), "mixed newlines")
    encode = lambda value: value.encode("utf-8").replace(b"\n", newline)
    changes = [("#pragma once\n", "#pragma once\n" + PROVENANCE)]
    changes += [(old, guarded(old, new)) for old, new in CHANGES]
    generated = original
    for old, new in changes:
        generated = replace_once(generated, encode(old), encode(new))
    restored = generated
    for old, new in reversed(changes):
        restored = replace_once(restored, encode(new), encode(old))
    require(restored == original, "inverse transform differs from raw native header")
    body = generated.replace(encode(PROVENANCE), b"", 1).decode("utf-8")
    require(project(body, False).encode("utf-8") == original, "OFF projection changed native bytes")
    enabled = project(body, True)
    for _, new in CHANGES:
        scope_proof(new)
    normalized = enabled.encode("utf-8")
    for old, new in reversed(CHANGES):
        normalized = replace_once(normalized, encode(new), encode(old))
    require(normalized == original, "ON changed bytes outside selected layout/copy/read sites")
    require(encode(LOOP) in original and encode(LOOP) in enabled.encode("utf-8"), "pipeline loop changed")
    for token in ("__syncthreads();", "cp_wait<0>();", "cp_commit();", "mma_s8(",
                  "causal_prompt_i8_fast_mma_f16_acc(", "__fmaf_rn("):
        require(enabled.count(token) == original.decode("utf-8").count(token), "math/barrier count changed")
    require(enabled.count("__syncthreads();") == 1, "one existing CTA barrier required")
    require("warp_active" not in PRODUCER and "return" not in PRODUCER, "producer participation changed")
    text = generated.decode("utf-8").replace("\r\n", "\n")
    qk = text[text.index("    const auto qk_softmax"):
              text.index("        const bool full_tile")]
    return generated, {
        "source_pins": PINS,
        "generated_sha256": sha256(generated),
        "generated_include": GENERATED,
        "inverse_byte_exact": True,
        "off_projection_byte_exact": True,
        "unchanged_bytes_outside_seven_selected_sites": True,
        "replacement_scopes_balanced": True,
        "native_QK_PV_arithmetic_and_pipeline_loop_preserved": True,
        "cta_barriers_per_page": 1,
        "exact_replacement_sites": len(changes),
        "generated_line_ranges": {
            "producer": line_range(text, PRODUCER),
            "qk": line_range(text, qk),
            "loop": line_range(text, LOOP),
        },
    }


def address_proof():
    rows = copies = zero_rows = scalar_reads = 0
    for stage in range(2):
        base = 32768 + stage * 34304
        ks, vs, end = base + 32768, base + 33792, base + 34304
        require((base | ks | vs | end) % 16 == 0, "shared alignment")
        require(vs - ks == 1024 and end - vs == 512, "scale extents")
        require(base + 16384 + 16384 + 512 + 512 == vs, "PV V-scale pointer")
        value_writes = [vs + (tid - 64) * 8 + byte for tid in range(64, 128) for byte in range(8)]
        require(value_writes == list(range(vs, end)), "native V-scale ownership/extent")
        for page in (0, 3, 1023):
            for head in range(4):
                source_base = (page * 4 + head) * 64 * 4 * 2
                require(source_base % 8 == 0, "source head alignment")
                for valid in range(65):
                    writes = collections.Counter()
                    for tid in range(64):
                        source = source_base + tid * 8
                        destination = ks + tid * 16
                        require(source % 8 == 0 and destination % 16 == 0, "vector alignment")
                        require(ks <= destination < destination + 16 <= vs, "K-scale write extent")
                        writes.update(range(destination, destination + 16))
                        if tid < valid:
                            require(source + 8 <= source_base + valid * 8, "tail read past valid key")
                            for group in range(4):
                                actual = 2 * (4 * 64 * (head + 4 * page) + 4 * tid + group)
                                require(actual == source + 2 * group, "native packed-half address")
                            copies += 1
                        else:
                            zero_rows += 1
                        rows += 1
                    require(len(writes) == 1024 and set(writes.values()) == {1}, "K-scale ownership")
        owners = collections.Counter()
        for warp in range(8):
            for lane in range(32):
                for nt in range(8):
                    for group in range(4):
                        for adjacent in range(2):
                            key = nt * 8 + 2 * (lane & 3) + adjacent
                            address = ks + (key * 4 + group) * 4
                            require(ks <= address <= vs - 4, "QK consumer address")
                            owners[key, group] += 1
                            scalar_reads += 1
        require(len(owners) == 256 and set(owners.values()) == {64}, "consumer scale multiplicity")
    require(32768 + 2 * 34304 == 101376, "total dynamic shared")
    guards = 0
    for warps in (4, 8):
        for packed in (False, True):
            for int8_keys in (False, True):
                for geometry in (False, True):
                    for metadata in (False, True):
                        selected = warps == 8 and not packed and int8_keys and geometry and metadata
                        guards += selected
    require(guards == 1, "fixed-shape guard is not exclusive")
    query_blocks = 0
    for width in (1, 33, 127, 129, 1280):
        for base_position in (0, 63, 31488):
            for query_block in range((width + 127) // 128):
                q0 = query_block * 128
                last = base_position + q0 + min(128, width - q0) - 1
                pages = last // 64 + 1
                tail = min(64, last + 1 - (pages - 1) * 64)
                require(1 <= tail <= 64 and (pages - 1) * 64 + tail == last + 1, "query tail mapping")
                query_blocks += 1
    return {"status": "PASS", "valid_keys_exhaustive": [0, 64], "stages": 2,
            "physical_pages": [0, 3, 1023], "heads": 4, "producer_threads": [0, 63],
            "vector_rows_checked": rows, "valid_8B_reads": copies, "zero_filled_16B_rows": zero_rows,
            "QK_scalar_reads_checked": scalar_reads, "consumers_per_stored_scale_full_CTA": 64,
            "query_blocks_checked": query_blocks, "query_widths": [1, 33, 127, 129, 1280],
            "stage_bytes_off_on": [33792, 34304], "shared_bytes_off_on": [100352, 101376],
            "partial_key_copy": "none: one whole 8B scale vector per valid key, including odd tails",
            "new_static_shared_bytes": 0, "global_workspace_bytes": 0}


def lifetime_proof():
    checked = 0
    for pages in (1, 2, 3, 4, 5, 512):
        for tile in range(pages):
            write_phase = 0 if tile == 0 else 3 * (tile - 1) + 2
            publish_phase = 3 * tile + 1
            read_phase = 3 * tile + 2
            require(write_phase < publish_phase < read_phase, "publish order")
            if tile + 2 < pages:
                all_readers_done_barrier = 3 * (tile + 1) + 1
                next_stage_write = 3 * (tile + 1) + 2
                require(read_phase < all_readers_done_barrier < next_stage_write, "stage reuse race")
                require((tile & 1) == ((tile + 2) & 1), "double-buffer stage")
            checked += 1
    overlapping_half_rows = []
    for producer in range(64):
        writes = set(range(producer * 16, producer * 16 + 16))
        for reader in range(64):
            reads = set(range(reader * 8, reader * 8 + 8))
            if producer != reader and writes & reads:
                overlapping_half_rows.append([producer, reader])
    require(overlapping_half_rows, "in-place expansion negative control missed its race")
    return {"status": "PASS_SOURCE_ORDER_MODEL", "tile_lifetimes_checked": checked,
            "producer_reads": "immutable global half vectors only; no staged-half reads",
            "publish": "producer normal shared stores precede the existing all-thread CTA barrier",
            "reuse": "barrier at tile t+1 follows all reads of t and precedes issue_tile(t+2)",
            "in_place_half_expansion_overlap_negative_control": overlapping_half_rows[:4],
            "compiler_order_requires_PTX_SASS_review": True,
            "async_tradeoff": "K-scale cp.async replaced by synchronous global load/conversion/store; K/V and V-scale async copies unchanged",
            "shared_payload_tradeoff": "K-scale writes grow from 512B to 1024B per page; each selected QK scale read grows from 2B to 4B"}


def half_bits_to_float_bits(bits):
    sign, exponent, fraction = (bits & 0x8000) << 16, (bits >> 10) & 31, bits & 1023
    if exponent == 0:
        if fraction == 0:
            return sign
        unbiased = -14
        while (fraction & 1024) == 0:
            fraction <<= 1
            unbiased -= 1
        return sign | ((unbiased + 127) << 23) | ((fraction & 1023) << 13)
    return sign | ((exponent - 15 + 127) << 23) | (fraction << 13)


def numeric_proof():
    finite = 0
    for bits in range(65536):
        if ((bits >> 10) & 31) == 31:
            continue
        native_value = struct.unpack("<e", struct.pack("<H", bits))[0]
        native_float = struct.pack("<f", native_value)
        expanded = struct.pack("<I", half_bits_to_float_bits(bits))
        require(expanded == native_float, "finite half expansion differs: " + hex(bits))
        require(struct.pack("<f", struct.unpack("<f", expanded)[0]) == native_float, "FP32 shared round trip")
        finite += 1
    require(finite == 63488, "finite-half coverage")
    boundary = [0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x47ff, 0x4800, 0x4801,
                0x4bff, 0x4c00, 0x4c01, 0x7bff, 0xfbff]
    for offset in range(len(boundary)):
        groups = [boundary[(offset + group) % len(boundary)] for group in range(4)]
        raw = struct.pack("<4H", *groups)
        x, y = struct.unpack("<2I", raw)
        unpacked = [x & 65535, x >> 16, y & 65535, y >> 16]
        require(unpacked == groups, "uint2 half2 low/high group order")
    return {"status": "PASS_ALL_FINITE_HALF_BITS", "finite_encodings": finite,
            "uint2_half2_float4_boundary_vectors": len(boundary),
            "includes": ["both signed zeros", "all signed subnormals", "8/16 boundaries", "signed maximum finite 65504"],
            "operation": "same __half2float conversion performed earlier, FP32 store/load exact",
            "scale_recomputation": False, "QK_multiply_FMA_order_changed": False,
            "NaN_or_infinity_contract_qualified": False, "GPU_math_qualified": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[2])
    parser.add_argument("--output-root", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    generated, result = prepare(args.source_root.resolve())
    result.update({
        "switch": SWITCH, "default": 0, "geometry": "CausalD256H24Kv4",
        "metadata": "PagedKVBatchMetadata<false>", "cache_storage": "Int8Group64",
        "warps": 8, "packed_values": False, "key_coding": "KvKeyCoding::Int8",
        "dynamic_shared_bytes_off_on": [100352, 101376], "additional_static_shared_bytes": 0,
        "candidate_resource_gate": {"max_registers": 255, "stack_bytes": 0,
                                    "spill_store_bytes": 0, "spill_load_bytes": 0,
                                    "local_bytes": 0, "max_total_shared_bytes": 101376},
        "known_native_baseline_stack_spill_store_spill_load_bytes": [8, 4, 4],
        "codegen_qualified": False, "gpu_correctness_qualified": False, "performance_claim": False,
        "probe_source_pins": {name: sha256((pathlib.Path(__file__).parent / name).read_bytes())
                              for name in ("generate.py", "variant.cu", "probe_api.h", "build.sh", "analyze.py")},
    })
    if args.self_test:
        result["cpu_address_proof"] = address_proof()
        result["cpu_lifetime_proof"] = lifetime_proof()
        result["cpu_numeric_proof"] = numeric_proof()
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
