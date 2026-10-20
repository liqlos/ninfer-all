import argparse
import collections
import hashlib
import json
import math
import pathlib
import re
import struct
import sys


SOURCE = "src/ops/softmax_attention/dense/causal_cache/prompt_i8_fast.cuh"
GENERATED = "attention_v_shift_generated.cuh"
SWITCH = "NINFER_PROMPT_V_SHIFT_CTA"
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
}
PROVENANCE = """#define NINFER_ATTENTION_V_SHIFT_GENERATED 1
#ifndef NINFER_PROMPT_V_SHIFT_CTA
#define NINFER_PROMPT_V_SHIFT_CTA 0
#endif
#if NINFER_PROMPT_V_SHIFT_CTA != 0 && NINFER_PROMPT_V_SHIFT_CTA != 1
#error NINFER_PROMPT_V_SHIFT_CTA_must_be_zero_or_one
#endif
"""
SHARED_ANCHOR = "    extern __shared__ __align__(16) unsigned char smem_raw[];\n"
SHARED_ENABLED = SHARED_ANCHOR + """    constexpr bool UseCtaVShift =
        D == 256 && Warps == 8 && !PackedValues && Keys == KvKeyCoding::Int8 &&
        std::is_same_v<Geometry, CausalD256H24Kv4> &&
        std::is_same_v<Metadata, PagedKVBatchMetadata<false>>;
    int* cta_stage_shift = nullptr;
    if constexpr (UseCtaVShift) {
        __shared__ int stage_shift[2];
        cta_stage_shift = stage_shift;
    }
"""
NATIVE_SCALE_COPY = """            const int key = tid - Bc;
            if constexpr (PackedValues) {
                cp_async_zfill<16>(vs_s + key * VGroups,
                                   cache_v_scale + kv_cache_int4_value_scale_index<Geometry>(
                                                       page, kv_head, 0, key),
                                   key < valid ? 16 : 0);
            } else {
                cp_async_zfill<8>(vs_s + key * VGroups,
                                  cache_v_scale +
                                      kv_cache_int8_quant_scale_index<Geometry>(page, kv_head, 0, key),
                                  key < valid ? 8 : 0);
            }
"""
PAIRED_SCALE_COPY = """            if constexpr (UseCtaVShift) {
                if (tid < Bc + 32) {
                    const int key = 2 * (tid - Bc);
                    const int bytes = min(2, max(0, valid - key)) * 8;
                    cp_async_zfill<16>(
                        vs_s + key * VGroups,
                        cache_v_scale +
                            kv_cache_int8_quant_scale_index<Geometry>(page, kv_head, 0, key),
                        bytes);
                }
            } else {
""" + NATIVE_SCALE_COPY + "            }\n"
NATIVE_SHIFT = """        static_assert(Bc * VGroups % (32 * 8) == 0);
        __half2 vmax2 = __float2half2_rn(0.0f);
#pragma unroll
        for (int part = 0; part < Bc * VGroups / (32 * 8); ++part) {
            const uint4 v8 = load_vec<uint4>(&vs_s[8 * (lane + 32 * part)]);
            vmax2          = __hmax2(
                vmax2, __hmax2(__hmax2(__habs2(load_vec<__half2>(&v8.x)),
                                       __habs2(load_vec<__half2>(&v8.y))),
                               __hmax2(__habs2(load_vec<__half2>(&v8.z)),
                                       __habs2(load_vec<__half2>(&v8.w)))));
        }
        const float vmax = warp_max(fmaxf(__low2float(vmax2), __high2float(vmax2)), FullMask);
        constexpr float limit = PackedValues ? kCausalPromptI8FastPackedF16PartialScaleLimit
                                             : kCausalPromptI8FastF16PartialScaleLimit;
        tile_shift = 0;
        if (vmax > limit) {
            while (ldexpf(vmax, -tile_shift) > limit) { ++tile_shift; }
        }
"""
SHIFT_CONSUMER = """        if constexpr (UseCtaVShift) {
            tile_shift = cta_stage_shift[kb & 1];
        } else {
""" + NATIVE_SHIFT + "        }\n"
NATIVE_WAIT_BARRIER = "        cp_wait<0>();\n        __syncthreads();\n"
SHIFT_PRODUCER = """        if constexpr (UseCtaVShift) {
            asm volatile("cp.async.wait_group 0;\\n" ::: "memory");
            if (warp == 2) {
                const unsigned char* base = stage_base(kb & 1);
                const __half* vs_s =
                    reinterpret_cast<const __half*>(base + Stage::KScaleOffset) + Bc * Groups;
                int producer_shift;
""" + re.sub(r"\btile_shift\b", "producer_shift", NATIVE_SHIFT) + """                if (lane == 0) { cta_stage_shift[kb & 1] = producer_shift; }
            }
        } else {
            cp_wait<0>();
        }
        __syncthreads();
"""


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(data, old, new):
    require(data.count(old) == 1, "expected one exact source fragment")
    return data.replace(old, new, 1)


def guarded(old, enabled):
    return "#if " + SWITCH + "\n" + enabled + "#else\n" + old + "#endif\n"


def project(source, enabled):
    output = []
    active = [True]
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
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
            require(depth >= 0, "replacement closes an outer scope")
    require(depth == 0, "replacement leaks scope")


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

    changes = (
        ("#pragma once\n", "#pragma once\n" + PROVENANCE),
        ("#include <cstdint>\n", "#include <cstdint>\n#include <type_traits>\n"),
        (SHARED_ANCHOR, guarded(SHARED_ANCHOR, SHARED_ENABLED)),
        (NATIVE_SCALE_COPY, guarded(NATIVE_SCALE_COPY, PAIRED_SCALE_COPY)),
        (NATIVE_SHIFT, guarded(NATIVE_SHIFT, SHIFT_CONSUMER)),
        (NATIVE_WAIT_BARRIER, guarded(NATIVE_WAIT_BARRIER, SHIFT_PRODUCER)),
    )
    generated = original
    for old, new in changes:
        generated = replace_once(generated, encode(old), encode(new))
    restored = generated
    for old, new in reversed(changes):
        restored = replace_once(restored, encode(new), encode(old))
    require(restored == original, "inverse transform is not byte exact")
    body = generated.replace(encode(PROVENANCE), b"", 1).replace(
        encode("#include <type_traits>\n"), b"", 1).decode("utf-8")
    require(project(body, False).encode("utf-8") == original, "OFF projection changed native bytes")
    enabled = project(body, True).encode("utf-8")
    for block in (SHARED_ENABLED, PAIRED_SCALE_COPY, SHIFT_CONSUMER, SHIFT_PRODUCER):
        scope_proof(block)
    producer_math = re.sub(r"\bproducer_shift\b", "tile_shift",
                          SHIFT_PRODUCER[SHIFT_PRODUCER.index("        static_assert"):
                                         SHIFT_PRODUCER.index("                if (lane == 0)")])
    require(producer_math == NATIVE_SHIFT, "producer reduction differs from native arithmetic")
    require(enabled.count(b"__syncthreads();") == original.count(b"__syncthreads();") == 1,
            "CTA barrier count changed")
    for token in (b"mma_s8(", b"causal_prompt_i8_fast_mma_f16_acc(", b"__fmaf_rn(",
                  b"cp_commit();", b"issue_tile(kb + 1)", b"commit_keys(kb + 1)"):
        require(enabled.count(token) == original.count(token), "native math/schedule sites changed")
    original_text = original.decode("utf-8")
    pv = original_text[original_text.index("    const auto pv_packed"):
                       original_text.index("    // Every warp publishes")]
    require(enabled.count(pv.encode("utf-8")) == 1, "PV body changed")
    qk = original_text[original_text.index("    const auto qk_softmax"):
                       original_text.index("        static_assert(Bc * VGroups")]
    require(enabled.count(qk.encode("utf-8")) == 1, "QK/softmax body changed")
    require(enabled.count(encode(SHIFT_PRODUCER +
                                 "        if (kb + 1 < key_blocks) { issue_tile(kb + 1); }\n"
                                 "        qk_softmax(kb);\n"
                                 "        pv(kb);\n"
                                 "        if (kb + 1 < key_blocks) { commit_keys(kb + 1); }\n")) == 1,
            "producer/wait/barrier/issue/math ordering changed")
    require('asm volatile("cp.async.wait_group 0;\\n" ::: "memory");' in SHIFT_PRODUCER,
            "ON wait must prevent compiler memory motion")
    require("warp_active" not in SHIFT_PRODUCER and "return" not in SHIFT_PRODUCER and
            SHIFT_PRODUCER.count("if (warp == 2)") == 1,
            "producer participation depends on query-row activity")
    return generated, {
        "source_pins": PINS,
        "generated_sha256": sha256(generated),
        "inverse_byte_exact": True,
        "off_projection_byte_exact": True,
        "replacement_scopes_balanced": True,
        "producer_math_exact_after_local_variable_rename": True,
        "qk_softmax_and_pv_byte_identical": True,
        "cta_barriers_per_page_unchanged": 1,
        "exact_replacement_sites": len(changes),
        "generated_include": GENERATED,
    }


def address_proof():
    byte_checks = 0
    paired_copies = 0
    partial_pairs = 0
    old_ownership_failures = 0
    for stage in range(2):
        shared_base = 32768 + stage * 33792 + 32768 + 512
        require(shared_base % 16 == 0 and shared_base + 512 <= 100352, "V-scale arena moved")
        for page in (0, 3, 31, 1023):
            for head in range(4):
                global_base = 512 * (head + 4 * page)
                for valid in range(65):
                    native = {}
                    for key in range(64):
                        for byte in range(8):
                            dst = shared_base + key * 8 + byte
                            native[dst] = global_base + key * 8 + byte if key < valid else None
                    candidate = {}
                    for lane in range(32):
                        key = 2 * lane
                        count = min(2, max(0, valid - key)) * 8
                        src = global_base + key * 8
                        dst = shared_base + key * 8
                        require(src % 16 == dst % 16 == 0, "paired copy alignment")
                        require(count in (0, 8, 16), "unsupported partial source size")
                        require(src + count <= global_base + max(key, valid) * 8,
                                "partial copy reads beyond valid keys")
                        paired_copies += 1
                        partial_pairs += count == 8
                        for byte in range(16):
                            candidate[dst + byte] = src + byte if byte < count else None
                            half_index = 8 * lane + byte // 2
                            require(shared_base + 2 * half_index + byte % 2 == dst + byte,
                                    "producer reads a byte outside its own async copy")
                            expanded_index = 2 * (4 * 64 * (head + 4 * page) +
                                                  4 * (key + byte // 8) + (byte % 8) // 2)
                            require(expanded_index + byte % 2 == src + byte,
                                    "pair differs from actual paged scale index")
                            old_ownership_failures += 64 + (key + byte // 8) != 64 + lane
                            byte_checks += 1
                    require(candidate == native, "paired copy differs from native zero fill")
    require(old_ownership_failures > 0, "old copy ownership negative control escaped")
    return {
        "status": "PASS_ADDRESS_TAIL_AND_OWN_COPY_PROOF",
        "valid_keys_exhaustive": [0, 64],
        "physical_pages": [0, 3, 31, 1023],
        "kv_heads": 4,
        "stages": 2,
        "paired_copies": paired_copies,
        "partial_8byte_pairs": partial_pairs,
        "symbolic_byte_checks": byte_checks,
        "old_ownership_negative_control_failures": old_ownership_failures,
        "producer_threads": [64, 95],
        "source_alignment_precondition": "actual FP16 V-scale plane base aligned to 16 bytes",
    }


def lifetime_proof():
    edges = collections.defaultdict(set)
    pages = 6

    def edge(a, b):
        edges[a].add(b)

    def before(a, b):
        pending = [a]
        seen = set()
        while pending:
            node = pending.pop()
            if node == b:
                return True
            if node not in seen:
                seen.add(node)
                pending.extend(edges[node])
        return False

    for page in range(pages):
        barrier = ("barrier", page)
        publish = ("publish", page)
        for lane in range(32):
            wait = ("producer_wait", page, lane)
            own_read = ("own_read", page, lane)
            edge(("issue", page, 2), wait)
            edge(wait, own_read)
            edge(own_read, publish)
        edge(publish, barrier)
        for warp in range(8):
            issue = ("issue", page, warp)
            wait = ("wait", page, warp)
            read = ("shift_read", page, warp)
            done = ("pv_done", page, warp)
            edge(issue, wait)
            edge(wait, barrier)
            edge(barrier, read)
            if page + 1 < pages:
                edge(barrier, ("issue", page + 1, warp))
                edge(("issue", page + 1, warp), read)
                edge(done, ("wait", page + 1, warp))
                if warp == 2:
                    for lane in range(32):
                        edge(done, ("producer_wait", page + 1, lane))
            edge(read, done)
    publication_checks = 0
    reuse_checks = 0
    for page in range(pages):
        for warp in range(8):
            require(before(("publish", page), ("shift_read", page, warp)),
                    "shift consumed before publication barrier")
            publication_checks += 1
            if page + 2 < pages:
                require(before(("shift_read", page, warp), ("publish", page + 2)),
                        "shift slot overwritten before all consumers finish")
                for producer in range(8):
                    require(before(("pv_done", page, warp), ("issue", page + 2, producer)),
                            "K/V stage overwritten before all consumers finish")
                    reuse_checks += 1
    inactive_producer_cases = 0
    for row_count in range(1, 129):
        qk_threads = {tid for tid in range(256) if (tid // 32) * 16 < row_count}
        producer_threads = {tid for tid in range(256) if tid // 32 == 2}
        require(producer_threads == set(range(64, 96)), "producer warp is incomplete")
        inactive_producer_cases += producer_threads.isdisjoint(qk_threads)
        require(len(producer_threads) == 32 and
                {tid % 32 for tid in producer_threads} == set(range(32)),
                "full-mask reduction has missing lanes")
    require(inactive_producer_cases == 32, "inactive producer tail cases not exercised")
    return {
        "status": "PASS_PUBLICATION_AND_STAGE_REUSE_PARTIAL_ORDER",
        "pipeline_pages": pages,
        "publication_checks": publication_checks,
        "stage_reuse_checks": reuse_checks,
        "query_rows_exhaustive": [1, 128],
        "producer_inactive_for_qk_tail_cases": inactive_producer_cases,
        "inactive_query_warps_participate": True,
        "reason": "producer is outside qk_softmax early returns; q0>=tokens return is CTA-uniform",
        "own_wait_read": "each producer lane reads only its completed 16-byte cp.async destination",
        "cross_lane": "unchanged full-mask warp_max after all lane-local reads",
        "cross_warp": "publish before existing CTA barrier; consume after it",
        "reuse": "B(k+1) orders all k consumers before issue(k+2) and shift publish(k+2)",
        "compiler_order": "same ON cp.async.wait_group 0 with explicit memory clobber",
    }


def numeric_proof():
    checked = 0
    largest_shift = 0
    boundaries = []
    for bits in range(65536):
        if bits & 0x7C00 == 0x7C00:
            continue
        value = abs(struct.unpack("<e", struct.pack("<H", bits))[0])
        shift = 0
        if value > 8.0:
            while math.ldexp(value, -shift) > 8.0:
                shift += 1
        independent = next(candidate for candidate in range(14)
                           if value <= math.ldexp(8.0, candidate))
        require(shift == independent, "native ldexp shift differs from minimal exact bound")
        require(struct.unpack("<f", struct.pack("<f", math.ldexp(value, -shift)))[0] ==
                math.ldexp(value, -shift), "tested ldexp value is not exact FP32")
        largest_shift = max(largest_shift, shift)
        checked += 1
        if bits in (0, 1, 0x3FF, 0x400, 0x47FF, 0x4800, 0x4801, 0x4BFF,
                    0x4C00, 0x4C01, 0x77FF, 0x7800, 0x7801, 0x7BFF, 0x8000):
            boundaries.append({"half_bits": f"{bits:04x}", "value": value, "shift": shift})
    require(checked == 63488 and largest_shift == 13, "finite FP16 domain incomplete")
    return {
        "status": "PASS_FINITE_FP16_NUMERIC_BOUNDARIES",
        "finite_half_patterns": checked,
        "max_shift": largest_shift,
        "boundaries": boundaries,
        "reduction_equivalence": "identical per-lane eight half operands and exact donor reduction",
        "pv_rounding": "unchanged consumer code and identical integer tile_shift",
        "nonfinite_limit": "no new finite-value repair; NaN/Inf behavior is not GPU-qualified",
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
        "geometry": "CausalD256H24Kv4",
        "metadata": "PagedKVBatchMetadata<false>",
        "cache_storage": "Int8Group64",
        "warps": 8,
        "packed_values": False,
        "key_coding": "KvKeyCoding::Int8",
        "dynamic_shared_bytes": 100352,
        "additional_static_shared_bytes": 8,
        "total_shared_bytes": 100360,
        "global_workspace_bytes": 0,
        "compiler_gate": {
            "toolchain": "CUDA 13.1.115",
            "architecture": "sm_86",
            "baseline_stack_spill_store_spill_load_bytes": [8, 4, 4],
            "candidate_must_not_exceed_baseline": True,
            "unchanged": ["QK", "PV", "accumulator math", "one CTA barrier per page"],
        },
        "codegen_qualified": False,
        "gpu_correctness_qualified": False,
        "performance_claim": False,
        "ptx_reference": "https://docs.nvidia.com/cuda/parallel-thread-execution/"
                         "index.html#data-movement-and-conversion-instructions-cp-async",
        "compiler_reference": "https://docs.nvidia.com/cuda/inline-ptx-assembly/"
                              "index.html#incorrect-optimization",
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
