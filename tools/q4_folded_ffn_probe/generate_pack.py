import argparse
import difflib
import hashlib
import json
import math
import pathlib
import random
import struct
import sys


SWITCH = "NINFER_GGUF_FFN_FOLDED16_PACK"
PHASE1_SHA256 = {
    "generate.py": "c886450bbbed0de470375049bafddda2f80b9ca67174f9a099551aa127581936",
    "variant.cu": "ee3b3e6e7e1eb17d2590ee2d76192d072d51a665b6867fb340b60c3617c98cd0",
}
PHASE1_GENERATED_SHA256 = {
    "mmq.cuh": "801b6f5844cc823203eff32bf246ee28d5d766e7d619111eaf1dfa441918455e",
    "mmq-load-tiles.cuh": "4edc1272d97aa933e906c4feb5d3fc36fb7c69c1468f8a45c20d4c3ca77fa68a",
    "mmq-vec-dot.cuh": "1c527b7a0f488524b34c6a7d9ea9e91bdd744f973e93844e81fa0c8c161d76cc",
}
PACK_SOURCE = pathlib.Path("src/ops/linear/gguf/ggml_bridge.cu")
PACK_SOURCE_SHA256 = "670d0910869d8b1183b02d37f7ab2f554def75b9ff7ba7152562ce65327608e6"
KERNEL_HEADER = "q4_folded_ffn_pack_kernel.cuh"
KERNEL_ANCHOR = (
    b"template <bool Diagnostic>\n"
    b"__global__ __launch_bounds__(kThreads, 1) void q4_folded_ffn_kernel("
)
PACK_ANCHOR = (
    b"template <mmq_q8_1_ds_layout ds_layout>\n"
    b"__global__ void quantize_matrix_kernel("
)
STAGE_STORE = b"""            const __nv_bfloat16 rounded = __float2bfloat16(value);
            if constexpr (Diagnostic) {
                diagnostic_bf16[at] = rounded;
            }
            const int shared_row = row - row0;
            const int shared_column = column - column0;
            swiglu_shared[shared_column * 64 + (shared_row ^ ((shared_column & 6) << 2))] =
                __bfloat162float(rounded);
"""
PACK_TAIL = b"""
    __syncthreads();
#pragma unroll
    for (int quad0 = 0; quad0 < kJ * kOutputRowsPerTile / 4; quad0 += kThreads) {
        const int quad = quad0 + threadIdx.y * 32 + threadIdx.x;
        const int shared_column = quad / 16;
        const int shared_row = 4 * (quad % 16);
        const int at = shared_column * 64 + (shared_row ^ ((shared_column & 6) << 2));
        const float4 xi = *reinterpret_cast<const float4*>(swiglu_shared + at);
        q4_folded_native_pack_quad<Layout>(
            xi, packed_output, row0 + shared_row, column0 + shared_column, ncols_y);
    }
"""


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(data, old, new):
    require(data.count(old) == 1, "expected one exact source fragment")
    return data.replace(old, new, 1)


def load_phase1(probe_root):
    source_files = {}
    for name, expected in PHASE1_SHA256.items():
        source = (probe_root / name).read_bytes()
        require(sha256(source) == expected, "phase1 source pin mismatch: " + name)
        source_files[name] = source
    module = {"__name__": "ninfer_folded16_pinned_phase1",
              "__file__": str(probe_root / "generate.py")}
    exec(compile(source_files["generate.py"], str(probe_root / "generate.py"), "exec"), module)
    return module, source_files


def native_pack_helper(phase1, source):
    native = phase1["braced_statement"](source, PACK_ANCHOR)
    constants_start = native.index(b"    constexpr int vals_per_scale")
    constants_stop = native.index(b"    const std::int64_t i0", constants_start)
    constants = native[constants_start:constants_stop]
    arithmetic_start = native.index(b"    const std::int64_t k_block")
    arithmetic = native[arithmetic_start:-1]
    require(arithmetic.endswith(b"\n"), "native pack function ending changed")
    require(arithmetic.count(b"roundf(") == 4, "native pack rounding sites changed")
    require(arithmetic.count(b"__shfl_xor_sync(") == 2, "native reduction sites changed")
    helper = (
        b"template <mmq_q8_1_ds_layout ds_layout>\n"
        b"static __device__ __forceinline__ void q4_folded_native_pack_quad(\n"
        b"        const float4 xi, block_q8_1_mmq* __restrict__ y,\n"
        b"        const std::int64_t i0, const int column, const int columns) {\n"
        b"    static_assert(ds_layout == MMQ_Q8_1_DS_LAYOUT_D4 ||\n"
        b"                  ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4);\n" +
        constants + arithmetic + b"}\n"
    )
    require(helper.count(constants + arithmetic) == 1,
            "native pack arithmetic must remain byte exact")
    return helper, {
        "native_pack_function_sha256": sha256(native),
        "native_pack_constants_sha256": sha256(constants),
        "native_pack_arithmetic_and_stores_sha256": sha256(arithmetic),
        "native_pack_arithmetic_and_stores_byte_identical": True,
        "native_pack_zero_and_nonfinite_behavior_unchanged_in_source": True,
    }


def derive_kernel(phase1, variant):
    original = phase1["braced_statement"](variant, KERNEL_ANCHOR)
    epilogue_start = original.index(b"    using tile_C = ggml_cuda_mma::tile<16, 8, int>;")
    old_epilogue = original[epilogue_start:-1]
    new_epilogue = b"    float* swiglu_shared = reinterpret_cast<float*>(folded_shared);\n" + old_epilogue
    new_epilogue = replace_once(
        new_epilogue, b"            output[at] = __float2bfloat16(value);\n", STAGE_STORE
    )
    new_epilogue += PACK_TAIL
    changes = (
        (b"template <bool Diagnostic>", b"template <mmq_q8_1_ds_layout Layout, bool Diagnostic>"),
        (b"void q4_folded_ffn_kernel(", b"void q4_folded_ffn_pack_kernel("),
        (b"__nv_bfloat16* __restrict__ output,",
         b"block_q8_1_mmq* __restrict__ packed_output,"),
        (b"float* __restrict__ diagnostic_gate, float* __restrict__ diagnostic_up,\n",
         b"float* __restrict__ diagnostic_gate, float* __restrict__ diagnostic_up,\n"
         b"        __nv_bfloat16* __restrict__ diagnostic_bf16,\n"),
        (old_epilogue, new_epilogue),
    )
    kernel = original
    for old, new in changes:
        kernel = replace_once(kernel, old, new)
    restored = kernel
    for old, new in reversed(changes):
        restored = replace_once(restored, new, old)
    require(restored == original, "kernel reverse transform changed phase1 bytes")
    body_start = b"    constexpr int warp_size = 32;"
    original_prefix = original[original.index(body_start):epilogue_start]
    actual_prefix = kernel[
        kernel.index(body_start):kernel.index(b"    float* swiglu_shared =")
    ]
    require(actual_prefix == original_prefix, "phase1 K body or initialization changed")
    require(kernel.count(b"float sum[kI * kJ / kThreads]") == 1,
            "native single accumulator array changed")
    require(kernel.count(b"__syncthreads();") == original.count(b"__syncthreads();") + 1,
            "shared packing requires exactly one additional barrier")
    require(b"output[at]" not in kernel, "phase1 BF16 materialization survived")
    require(kernel.count(b"diagnostic_bf16[at] = rounded;") == 1,
            "diagnostic BF16 output missing")
    return kernel, {
        "phase1_kernel_sha256": sha256(original),
        "phase1_initialization_and_k_body_sha256": sha256(original_prefix),
        "phase2_initialization_and_k_body_sha256": sha256(actual_prefix),
        "initialization_and_k_body_byte_identical": True,
        "kernel_reverse_transform_byte_identical": True,
        "additional_post_k_barriers": 1,
        "shared_stage_bytes": 32768,
        "extra_dynamic_shared_bytes": 0,
    }


def prepare(source_root, probe_root):
    phase1, sources = load_phase1(probe_root)
    originals, folded_headers, phase1_metadata = phase1["prepare"](source_root)
    require({name: sha256(data) for name, data in folded_headers.items()} ==
            PHASE1_GENERATED_SHA256, "phase1 generated headers differ from qualified source")
    native_pack_source = (source_root / PACK_SOURCE).read_bytes()
    require(sha256(native_pack_source) == PACK_SOURCE_SHA256, "native pack source pin mismatch")
    files = {}
    substitutions = (
        (phase1["SWITCH"].encode("ascii"), SWITCH.encode("ascii")),
        (b"NINFER_Q4_FOLDED16_GENERATED_", b"NINFER_Q4_FOLDED16_PACK_GENERATED_"),
    )
    for name, data in folded_headers.items():
        generated = data
        for old, new in substitutions:
            require(old in generated, "phase1 generated guard or provenance missing")
            generated = generated.replace(old, new)
        restored = generated
        for old, new in reversed(substitutions):
            restored = restored.replace(new, old)
        require(restored == data, "pack guard inverse changed phase1 loader/dot header")
        files["ggml-cuda/" + name] = generated
    kernel, kernel_proof = derive_kernel(phase1, sources["variant.cu"])
    helper, pack_proof = native_pack_helper(phase1, native_pack_source)
    constant_start = sources["variant.cu"].index(b"constexpr int kRows = 17408;")
    constant_stop = sources["variant.cu"].index(KERNEL_ANCHOR)
    constants = sources["variant.cu"][constant_start:constant_stop]
    constants += (
        b"constexpr std::size_t kPackedBytes =\n"
        b"    std::size_t(kRows / QK8_1_MMQ) * kColumns * sizeof(block_q8_1_mmq);\n"
        b"static_assert(kOutputRowsPerTile * kJ * sizeof(float) <= kSharedBytes);\n"
        b"static_assert(kPackedBytes == 25067520);\n\n"
    )
    files[KERNEL_HEADER] = (
        b"#pragma once\n\n#if " + SWITCH.encode("ascii") + b"\n"
        b"#define NINFER_Q4_FOLDED16_PACK_GENERATED_KERNEL 1\n\nnamespace {\n\n" +
        constants + helper + b"\n" + kernel + b"\n\n}\n\n#endif\n"
    )
    metadata = {
        "status": "PASS_BOUNDED_FOLDED16_PACK_TRANSFORM",
        "switch": SWITCH,
        "default": 0,
        "phase1_source_files": dict(PHASE1_SHA256),
        "phase1_generated_headers": dict(PHASE1_GENERATED_SHA256),
        "source_headers": phase1_metadata["source_headers"],
        "source_dependencies": phase1_metadata["source_dependencies"],
        "native_pack_source": PACK_SOURCE.as_posix(),
        "native_pack_source_sha256": PACK_SOURCE_SHA256,
        "phase1_disabled_headers": phase1_metadata["disabled_headers"],
        "disabled_original_headers_byte_identical":
            phase1_metadata["disabled_headers_byte_identical"],
        "loader_and_dot_byte_identical_except_flag_and_provenance": True,
        "shape": phase1_metadata["shape"],
        "tile": phase1_metadata["tile"],
        "output_layouts": {"D4": 0, "DS4": 1},
        "packed_output_bytes": 25067520,
        "native_activation_slack_written": False,
        "diagnostic_output": "all three gate/up FP32 and BF16 planes, or none",
        "timed_global_stores": "packed Q8 codes and D4 or DS4 metadata only",
        "kernel_proof": kernel_proof,
        "pack_proof": pack_proof,
        "native_k_schedule": phase1["schedule_proof"](originals, kernel),
        "generated_files": {name: sha256(data) for name, data in files.items()},
        "codegen_qualified": False,
        "gpu_correctness_qualified": False,
        "performance_claim": False,
    }
    return phase1, files, metadata, kernel, sources["variant.cu"]


def shared_address(row, column):
    return column * 64 + (row ^ ((column & 6) << 2))


def writer_mapping_proof():
    shared = {}
    store_transactions = 0
    for warp in range(8):
        for j0 in range(0, 128, 16):
            for l in range(4):
                banks = []
                for lane in range(32):
                    row = 16 * (warp // 2) + 8 * (l // 2) + lane // 4
                    column = j0 + 8 * (warp % 2) + 2 * (lane % 4) + l % 2
                    address = shared_address(row, column)
                    require(address not in shared, "duplicate BF16-rounded shared writer")
                    shared[address] = (row, column)
                    banks.append(address % 32)
                require(len(set(banks)) == 32, "shared producer bank collision in model")
                store_transactions += 1
    require(set(shared) == set(range(8192)), "shared stage coverage or bound changed")
    reads = set()
    load_transactions = 0
    reduction_groups = 0
    for quad0 in range(0, 2048, 256):
        for warp in range(8):
            for subgroup in range(4):
                banks = []
                group_rows = []
                group_columns = []
                for local_lane in range(8):
                    lane = 8 * subgroup + local_lane
                    quad = quad0 + 32 * warp + lane
                    column, row_quad = divmod(quad, 16)
                    row = 4 * row_quad
                    address = shared_address(row, column)
                    require(address % 4 == 0, "unaligned shared float4 reader")
                    for item in range(4):
                        require(shared[address + item] == (row + item, column),
                                "shared permutation does not deliver native float4")
                        require((row + item, column) not in reads, "duplicate pack input")
                        reads.add((row + item, column))
                        banks.append((address + item) % 32)
                        group_rows.append(row + item)
                        group_columns.append(column)
                require(len(set(banks)) == 32, "shared reader bank collision in model")
                require(len(set(group_columns)) == 1, "reduction mixes columns")
                require(group_rows == list(range(group_rows[0], group_rows[0] + 32)) and
                        group_rows[0] % 32 == 0, "reduction does not own native 32-row group")
                load_transactions += 1
                reduction_groups += 1
    require(len(reads) == 8192, "pack input coverage changed")
    packed_bytes = set()
    codeword_writers = 0
    metadata_writers = 0
    for row0 in (0, 64):
        for quad in range(2048):
            column, row_quad = divmod(quad, 16)
            row = row0 + 4 * row_quad
            ib = (row // 128) * 128 + column
            code = 144 * ib + 16 + row % 128
            require(code % 4 == 0, "unaligned packed codeword")
            for address in range(code, code + 4):
                require(address not in packed_bytes, "overlapping CTA codeword stores")
                packed_bytes.add(address)
            codeword_writers += 1
            if row % 32 == 0:
                meta = 144 * ib + 4 * ((row % 128) // 32)
                for address in range(meta, meta + 4):
                    require(address not in packed_bytes, "overlapping CTA metadata stores")
                    packed_bytes.add(address)
                metadata_writers += 1
    require(packed_bytes == set(range(128 * 144)), "two CTAs do not cover full packed blocks")
    full_column_codes = set()
    full_column_meta = set()
    for row0 in range(0, 17408, 64):
        for local_row in range(0, 64, 4):
            row = row0 + local_row
            full_column_codes.add((row // 128, 16 + row % 128))
            if row % 32 == 0:
                full_column_meta.add((row // 128, 4 * ((row % 128) // 32)))
    require(len(full_column_codes) == 4352 and len(full_column_meta) == 544,
            "actual intermediate width writer coverage")
    require({block for block, _ in full_column_codes} == set(range(136)),
            "actual intermediate width block bound")
    block_ids = [row_block * 1280 + column
                 for row_block in range(136) for column in range(1280)]
    require(block_ids == list(range(136 * 1280)), "global packed block indexing changed")
    require((max(block_ids) + 1) * 144 == 25067520, "global packed allocation bound")
    return {
        "status": "PASS_CPU_PACK_WRITER_AND_SHARED_MAPPING",
        "shared_words": len(shared),
        "shared_bytes": 4 * len(shared),
        "modeled_conflict_free_store_transactions": store_transactions,
        "modeled_conflict_free_float4_load_transactions": load_transactions,
        "native_32_value_reduction_groups_per_cta": reduction_groups,
        "paired_cta_codeword_writers": codeword_writers,
        "paired_cta_metadata_writers": metadata_writers,
        "paired_cta_packed_bytes": len(packed_bytes),
        "actual_column_codeword_writers": len(full_column_codes),
        "actual_column_metadata_writers": len(full_column_meta),
        "actual_packed_blocks": len(block_ids),
        "actual_packed_bytes": 25067520,
        "payload_initialization_required": False,
        "native_activation_slack_outside_writer_proof": True,
        "gpu_bank_conflicts_measured": False,
    }


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def rounded_bf16(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def half_bytes(value):
    try:
        return struct.pack("<e", value)
    except OverflowError:
        return struct.pack("<e", math.copysign(math.inf, value))


def round_away(value):
    return math.floor(value + 0.5) if value >= 0.0 else math.ceil(value - 0.5)


def reference_group(values, layout):
    require(len(values) == 32 and all(math.isfinite(x) for x in values),
            "CPU pack model accepts finite groups of 32")
    maximum = max(abs(x) for x in values)
    require(maximum > 0.0, "CPU numerical model does not qualify zero/NaN char conversion")
    inverse = f32(127.0 / maximum)
    scale = f32(1.0 / inverse)
    codes = bytes(round_away(f32(value * inverse)) & 255 for value in values)
    partial = [f32(f32(f32(values[i] + values[i + 1]) + values[i + 2]) + values[i + 3])
               for i in range(0, 32, 4)]
    left = f32(f32(partial[0] + partial[4]) + f32(partial[2] + partial[6]))
    right = f32(f32(partial[1] + partial[5]) + f32(partial[3] + partial[7]))
    total = f32(left + right)
    metadata = struct.pack("<f", scale) if layout == 0 else half_bytes(scale) + half_bytes(total)
    return codes, metadata


def warp_pack(quads, layout):
    require(len(quads) == 32 and all(len(quad) == 4 for quad in quads), "warp quad shape")
    maxima = [max(abs(value) for value in quad) for quad in quads]
    sums = [f32(f32(f32(quad[0] + quad[1]) + quad[2]) + quad[3]) for quad in quads]
    for offset in (4, 2, 1):
        old_maxima = maxima
        maxima = [max(old_maxima[lane], old_maxima[lane ^ offset]) for lane in range(32)]
        if layout == 1:
            old_sums = sums
            sums = [f32(old_sums[lane] + old_sums[lane ^ offset]) for lane in range(32)]
    codes = []
    metadata = []
    for lane in range(32):
        require(maxima[lane] > 0.0 and math.isfinite(maxima[lane]), "finite CPU warp maximum")
        inverse = f32(127.0 / maxima[lane])
        scale = f32(1.0 / inverse)
        codes.append(bytes(round_away(f32(value * inverse)) & 255 for value in quads[lane]))
        metadata.append(struct.pack("<f", scale) if layout == 0
                        else half_bytes(scale) + half_bytes(sums[lane]))
    return codes, metadata


def packed_bytes_proof():
    cases = []
    rng = random.Random(0x309086)
    cases.append([[rounded_bf16(f32(rng.uniform(-64.0, 64.0))) for row in range(128)]
                  for column in range(128)])
    sequence = [16777216.0, 1.0, -16777216.0, 1.0, -0.0, 0.5, -0.5, 2.0,
                3.0, -2.0, 0.125, -0.125, 256.0, -256.0, 1.5, -1.5]
    cases.append([[rounded_bf16(sequence[(row + 3 * column) % len(sequence)])
                   for row in range(128)] for column in range(128)])
    cases.append([[rounded_bf16(f32(math.ldexp((row % 31) - 15.5,
                                              (column % 17) - 8)))
                   for row in range(128)] for column in range(128)])
    comparisons = 0
    for values in cases:
        for layout in (0, 1):
            expected = bytearray(128 * 144)
            for column in range(128):
                for row in range(0, 128, 32):
                    codes, meta = reference_group(values[column][row:row + 32], layout)
                    expected[column * 144 + 16 + row:column * 144 + 48 + row] = codes
                    expected[column * 144 + 4 * (row // 32):
                             column * 144 + 4 * (row // 32) + 4] = meta
            actual = bytearray([0xA5]) * (128 * 144)
            for row0 in (0, 64):
                staged = [None] * 8192
                for row in range(64):
                    for column in range(128):
                        staged[shared_address(row, column)] = values[column][row0 + row]
                for quad0 in range(0, 2048, 256):
                    for warp in range(8):
                        quads = []
                        coordinates = []
                        for lane in range(32):
                            quad = quad0 + 32 * warp + lane
                            column, row_quad = divmod(quad, 16)
                            row = 4 * row_quad
                            at = shared_address(row, column)
                            quads.append(staged[at:at + 4])
                            coordinates.append((row0 + row, column))
                        codes, metadata = warp_pack(quads, layout)
                        for lane, (row, column) in enumerate(coordinates):
                            code_at = column * 144 + 16 + row
                            actual[code_at:code_at + 4] = codes[lane]
                            if row % 32 == 0:
                                meta_at = column * 144 + 4 * (row // 32)
                                actual[meta_at:meta_at + 4] = metadata[lane]
            require(actual == expected, "CPU packed bytes differ from column/group oracle")
            comparisons += len(actual)
    return {
        "status": "PASS_CPU_FINITE_PACKED_BYTES",
        "layouts": ["D4", "DS4"],
        "input_cases": len(cases),
        "compared_bytes": comparisons,
        "BF16_boundary_explicit": True,
        "DS4_local4_then_xor_4_2_1": True,
        "oracle": "independent column/group bytes with explicit FP32 rounding",
        "zero_nonfinite_char_conversion_gpu_qualified": False,
        "gpu_correctness_qualified": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, default=pathlib.Path("/src"))
    parser.add_argument("--output-root", type=pathlib.Path,
                        default=pathlib.Path("/build/q4-folded-ffn/pack/generated"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--diff", action="store_true")
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    probe_root = pathlib.Path(__file__).resolve().parent
    phase1, files, metadata, kernel, original_variant = prepare(source_root, probe_root)
    if args.self_test:
        metadata["phase1_mapping"] = phase1["mapping_proof"]()
        metadata["writer_mapping"] = writer_mapping_proof()
        metadata["packed_bytes"] = packed_bytes_proof()
    metadata["probe_files"] = {
        name: sha256((probe_root / name).read_bytes())
        for name in ("generate_pack.py", "variant_pack.cu", "pack_api.h", "build_pack.sh")
    }
    if not args.check_only:
        destination = args.output_root.resolve()
        require(not destination.is_relative_to(source_root),
                "generated files must be outside the source tree")
        for name, data in files.items():
            path = destination / name
            require(not path.exists() or path.read_bytes() == data,
                    "refusing to overwrite a different generated file: " + name)
        for name, data in files.items():
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                with path.open("xb") as handle:
                    handle.write(data)
            require(sha256(path.read_bytes()) == metadata["generated_files"][name],
                    "generated file verification failed: " + name)
        metadata["output_root"] = str(destination)
    metadata["check_only"] = args.check_only
    if args.diff:
        original_kernel = phase1["braced_statement"](original_variant, KERNEL_ANCHOR)
        sys.stdout.writelines(difflib.unified_diff(
            original_kernel.decode("utf-8").splitlines(keepends=True),
            kernel.decode("utf-8").splitlines(keepends=True),
            fromfile="phase1/q4_folded_ffn_kernel",
            tofile="phase2/q4_folded_ffn_pack_kernel",
        ))
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
