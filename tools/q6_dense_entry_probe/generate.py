import argparse
import difflib
import hashlib
import json
import pathlib
import sys


CUDA_DIRECTORY = "third_party/ggml-quants/src/ggml-cuda/"
EXPECTED_SHA256 = {
    CUDA_DIRECTORY + "mmq.cuh": "493ce59f8be4ff269fd0a2a8386f9d1e9a80bd2ee9c7478083d36bc2ac9ccf45",
    CUDA_DIRECTORY + "mmq-load-tiles.cuh": "80a7e02c1b2019db4ebdc1ef398a405c0249174dcf13a3d74b5d29af0347e3cf",
    CUDA_DIRECTORY + "mmq-vec-dot.cuh": "e47f3be4edcf96341b9ad48df24bd87480051c50bf7cabeef251abcb6ae39c66",
    CUDA_DIRECTORY + "mma.cuh": "08a18ccf4400990e9b2a651fdf702f49f0a521e427f151443e4359c912ac4fab",
    "src/ops/linear/gguf/ggml_bridge.cu": "670d0910869d8b1183b02d37f7ab2f554def75b9ff7ba7152562ce65327608e6",
    "src/ops/linear/gguf/ggml_bridge_mmq.cuh": "2513cb559034db1422841508c1c251e060ec087fe7b92643b2b539f687a7b15b",
    "tests/ops/linear/test_gguf_mmq_store.cu": "c88bcb861c57546cfe83cec1135019dafce50b165f814389921bc3f915d137b5",
}
SWITCH = "NINFER_GGUF_Q6_MMQ_DENSE_ENTRY"
MMA_START = """template<ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_write_back_mma("""
BF16_START = """template<ggml_type type, int J, bool fallback, ggml_prec prec_src1, bool bf16_accumulate = false,
         bool bf16_gate_product = false>
static __device__ __forceinline__ void ggml_cuda_mmq_write_back_bf16("""
BF16_END = "// -------------------------------------------------------------------------------------------------------------------------------------"
PROCESS_START = """template <ggml_type type, int J, bool fallback, bool fixup, ggml_prec prec_src1 = GGML_PREC_Q8,
          bool bf16_store = false, bool bf16_accumulate = false, bool bf16_gate_product = false, bool child_cta = false>
static __device__ __forceinline__ void mul_mat_q_process_tile("""
PROCESS_END = '// The mul_mat_q kernel implements "stream-k" work partitioning'
DENSE_ENTRY = """#ifndef NINFER_GGUF_Q6_MMQ_DENSE_ENTRY
#define NINFER_GGUF_Q6_MMQ_DENSE_ENTRY 0
#endif

#if NINFER_GGUF_Q6_MMQ_DENSE_ENTRY
template <bool bf16_store, bool bf16_accumulate = false>
__launch_bounds__(256, 1)
static __global__ void q6_dense_entry_kernel(
        const char * __restrict__ x, const int * __restrict__ y,
        float * __restrict__ dst, __nv_bfloat16 * __restrict__ dst_bf16,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int blocks_per_row, const uint3 ntx) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 860)
    static_assert(!bf16_accumulate || bf16_store);
    static_assert(ggml_cuda_mmq_get_I(GGML_TYPE_Q6_K, 128, false, GGML_PREC_Q8) == 128);
    static_assert(ggml_cuda_mmq_get_nthreads(GGML_TYPE_Q6_K, 128, false, GGML_PREC_Q8) == 256);
    static_assert(ggml_cuda_mmq_get_occupancy(GGML_TYPE_Q6_K, 128, false, GGML_PREC_Q8) == 1);
    const uint2 ij = fast_div_modulo(blockIdx.x, ntx);
    const int it = ij.x;
    const int jt = ij.y;
    const int offset_x = it * 128 * stride_row_x;
    const int offset_y = jt * 128 * (sizeof(block_q8_1_mmq) / sizeof(int));
    const int offset_dst = jt * 128 * stride_col_dst + it * 128;
    float * tile_dst = nullptr;
    __nv_bfloat16 * tile_dst_bf16 = nullptr;
    if constexpr (bf16_store) { tile_dst_bf16 = dst_bf16 + offset_dst; }
    else { tile_dst = dst + offset_dst; }
    mul_mat_q_process_tile<GGML_TYPE_Q6_K, 128, false, false, GGML_PREC_Q8,
                          bf16_store, bf16_accumulate, false, false, true>(
        x, offset_x, y + offset_y, nullptr, tile_dst, tile_dst_bf16, nullptr, nullptr,
        stride_row_x, ncols_y, stride_col_dst, 127, 127, 0, blocks_per_row, 0);
#elif defined(__CUDA_ARCH__)
    NO_DEVICE_CODE;
#endif
}
#endif

"""
SHAPES = ((5120, 17408, 1280), (10240, 5120, 1280), (1024, 5120, 1280))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def encoded(text, newline):
    return text.replace("\n", newline.decode()).encode()


def span(data, start, end, newline):
    first = encoded(start, newline)
    last = encoded(end, newline)
    if data.count(first) != 1:
        raise ValueError("source start marker is not unique: " + start)
    begin = data.index(first)
    stop = data.find(last, begin + len(first))
    if stop < 0:
        raise ValueError("source end marker is missing: " + end)
    return begin, stop, data[begin:stop]


def replace_checked(data, old, new, count, newline):
    before, after = encoded(old, newline), encoded(new, newline)
    if data.count(before) != count:
        raise ValueError("unexpected bounded fragment count: " + old)
    changed = data.replace(before, after)
    if changed.count(after) != count or changed.replace(after, before) != data:
        raise ValueError("bounded replacement did not round-trip")
    return changed


def prepare(source_root):
    originals = {}
    for name, expected in EXPECTED_SHA256.items():
        data = (source_root / name).read_bytes()
        if sha256(data) != expected:
            raise ValueError("source SHA256 mismatch: " + name)
        originals[name] = data
    source = originals[CUDA_DIRECTORY + "mmq.cuh"]
    newline = b"\r\n" if b"\r\n" in source else b"\n"
    if SWITCH.encode() in source or b"dense_ids" in source or b"q6_dense_entry_kernel" in source:
        raise ValueError("source already contains dense-entry changes")
    replacements = []

    begin, stop, old = span(source, MMA_START, BF16_START, newline)
    new = replace_checked(old, "template<ggml_type type, int J, bool fallback>",
                          "template<ggml_type type, int J, bool fallback, bool dense_ids = false>",
                          1, newline)
    new = replace_checked(new, "ids_dst[j]", "(dense_ids ? j : ids_dst[j])", 3, newline)
    replacements.append(("mma_writer", begin, stop, old, new))

    begin, stop, old = span(source, BF16_START, BF16_END, newline)
    new = replace_checked(old, "bool bf16_gate_product = false>",
                          "bool bf16_gate_product = false, bool dense_ids = false>", 1, newline)
    new = replace_checked(new, "ids_dst[j]", "(dense_ids ? j : ids_dst[j])", 4, newline)
    replacements.append(("bf16_writer", begin, stop, old, new))

    begin, stop, old = span(source, PROCESS_START, PROCESS_END, newline)
    new = replace_checked(old, "bool child_cta = false>",
                          "bool child_cta = false, bool dense_ids = false>", 1, newline)
    new = replace_checked(new, "    extern __shared__ int data_mul_mat_q[];",
                          """    static_assert(!dense_ids || (type == GGML_TYPE_Q6_K && J == 128 &&
        !fallback && !fixup && !child_cta && !bf16_gate_product && prec_src1 == GGML_PREC_Q8));
    extern __shared__ int data_mul_mat_q[];""", 1, newline)
    new = replace_checked(new, "                                          bf16_gate_product>(",
                          "                                          bf16_gate_product, dense_ids>(", 1, newline)
    new = replace_checked(new,
                          "            write_back(sum, ids_dst, dst, y_scale, stride_col_dst, tile_x_max_i, tile_y_max_j);",
                          """            if constexpr (dense_ids) {
                ggml_cuda_mmq_write_back_mma<type, J, fallback, true>(
                    sum, ids_dst, dst, y_scale, stride_col_dst, tile_x_max_i, tile_y_max_j);
            } else {
                write_back(sum, ids_dst, dst, y_scale, stride_col_dst, tile_x_max_i, tile_y_max_j);
            }""", 1, newline)
    body_start = "    extern __shared__ int data_mul_mat_q[];"
    body_end = "    if constexpr (child_cta) {\n        constexpr int column_banks_per_warp = 4;"
    _, _, native_body = span(old, body_start, body_end, newline)
    _, _, candidate_body = span(new, body_start, body_end, newline)
    if candidate_body != native_body:
        raise ValueError("native shared layout or K-body changed")
    new += encoded(DENSE_ENTRY, newline)
    replacements.append(("process_epilogue_and_entry", begin, stop, old, new))

    generated = source
    for _, begin, stop, old, new in reversed(replacements):
        if generated[begin:stop] != old:
            raise ValueError("bounded regions overlap")
        generated = generated[:begin] + new + generated[stop:]
    reverse = generated
    for _, _, _, old, new in reversed(replacements):
        if reverse.count(new) != 1:
            raise ValueError("generated region is not unique")
        reverse = reverse.replace(new, old, 1)
    if reverse != source:
        raise ValueError("complete mmq.cuh transform did not round-trip")
    files = {"ggml-cuda/mmq.cuh": generated}
    for name in ("mmq-load-tiles.cuh", "mmq-vec-dot.cuh"):
        files["ggml-cuda/" + name] = originals[CUDA_DIRECTORY + name]
    bridge = originals["src/ops/linear/gguf/ggml_bridge.cu"]
    bridge_newline = b"\r\n" if b"\r\n" in bridge else b"\n"
    _, _, producer = span(bridge, "constexpr int kMatrixQuantizeThreads = 128;",
                         "template <mmq_q8_1_ds_layout layout_a, mmq_q8_1_ds_layout layout_b>",
                         bridge_newline)
    oracle = originals["tests/ops/linear/test_gguf_mmq_store.cu"]
    oracle_newline = b"\r\n" if b"\r\n" in oracle else b"\n"
    _, _, decoder = span(oracle, "double q6_runtime_weight_value(",
                         "int check_q6_runtime_oracle(", oracle_newline)
    _, _, input_maker = span(oracle, "std::vector<__nv_bfloat16> make_input(",
                             "bool compare_rounding(", oracle_newline)
    files["support/q6_native_producer.cuh"] = producer
    files["support/q6_independent_oracle.cuh"] = input_maker + decoder
    metadata = {
        "status": "PASS_BOUNDED_Q6_DENSE_ENTRY_TRANSFORM",
        "switch": SWITCH,
        "default": 0,
        "source_sha256": sha256(source),
        "generated_header_sha256": sha256(generated),
        "source_files": dict(EXPECTED_SHA256),
        "generated_headers": {name: sha256(data) for name, data in files.items()},
        "changed_regions": [
            {"name": name, "original_lines": [source[:begin].count(b"\n") + 1,
                                             source[:stop].count(b"\n")],
             "original_sha256": sha256(old), "generated_sha256": sha256(new)}
            for name, begin, stop, old, new in replacements
        ],
        "native_k_body_sha256": sha256(native_body),
        "native_k_body_byte_identical": candidate_body == native_body,
        "siblings_byte_identical": all(files["ggml-cuda/" + name] == originals[CUDA_DIRECTORY + name]
                                      for name in ("mmq-load-tiles.cuh", "mmq-vec-dot.cuh")),
        "native_producer_sha256": sha256(producer),
        "independent_q6_decoder_sha256": sha256(decoder),
        "input_maker_sha256": sha256(input_maker),
        "native_producer_entry": "quantize_matrix_kernel<MMQ_Q8_1_DS_LAYOUT_D4>",
        "dense_kernel": "q6_dense_entry_kernel",
        "shapes_n_k_t": SHAPES,
        "performance_claim": False,
    }
    return source, generated, files, metadata


def eligible(rows, k, columns, cc=860, i=128, j=128, threads=256, stream_k=True,
             fixup=False, blocks=None, row_tiles=None, column_tiles=None):
    if (rows, k, columns) not in SHAPES or cc != 860 or k % 512 or rows % 128 or columns % 128:
        return False
    ri, cj = rows // 128, columns // 128
    return (i, j, threads, stream_k, fixup, blocks, row_tiles, column_tiles) == (
        128, 128, 256, True, False, ri * cj, ri, cj)


def self_test():
    local_outputs = set()
    for warp in range(8):
        for lane in range(32):
            slots = set()
            for j0 in range(0, 128, 16):
                for n in range(2):
                    for l in range(4):
                        row = (warp // 2) * 32 + n * 16 + (l // 2) * 8 + lane // 4
                        column = j0 + (warp % 2) * 8 + (lane % 4) * 2 + l % 2
                        slot = (j0 // 8 + n) * 4 + l
                        if not (0 <= row < 128 and 0 <= column < 128 and 0 <= slot < 64):
                            raise ValueError("writer coordinate or accumulator slot is out of bounds")
                        if (row, column) in local_outputs or slot in slots:
                            raise ValueError("writer duplicates an output or accumulator slot")
                        local_outputs.add((row, column))
                        slots.add(slot)
            if len(slots) != 64:
                raise ValueError("writer lost an accumulator")
    if len(local_outputs) != 128 * 128:
        raise ValueError("writer does not cover the full tile")
    tiles = 0
    k_iterations = 0
    addresses = 0
    guard_rejections = 0
    for rows, k, columns in SHAPES:
        ni, nj, kb = rows // 128, columns // 128, k // 256
        count = ni * nj
        efficiency = 100 * count // (82 * ((count + 81) // 82))
        blocks = count if efficiency >= 90 else 82
        options = {"blocks": blocks, "row_tiles": ni, "column_tiles": nj}
        if not eligible(rows, k, columns, **options):
            raise ValueError("specified native shape does not use one full tile per CTA")
        for changes in ({"cc": 890}, {"i": 64}, {"j": 64}, {"threads": 128},
                        {"stream_k": False}, {"fixup": True}, {"blocks": count - 1},
                        {"row_tiles": ni + 1}, {"column_tiles": nj - 1}):
            if eligible(rows, k, columns, **(options | changes)):
                raise ValueError("dense guard admitted an invalid launch")
            guard_rejections += 1
        for invalid in ((rows - 1, k, columns), (rows, k - 256, columns),
                        (rows, k, columns - 1), (rows, k, 768), (rows, k + 512, columns)):
            if eligible(*invalid, **options):
                raise ValueError("dense guard admitted an unqualified shape")
            guard_rejections += 1
        for pid in range(count):
            first = pid * (ni * nj * kb) // blocks
            stop = (pid + 1) * (ni * nj * kb) // blocks
            it, jt = divmod(pid, nj)
            if first % kb or stop - first != kb or divmod(first // kb, nj) != (it, jt):
                raise ValueError("generic full-K partition differs from dense entry")
            old_k = list(range(first % kb, min(kb, first % kb + stop - first)))
            if old_k != list(range(kb)):
                raise ValueError("K iteration order changed")
            for step in (0, kb // 2, kb - 1):
                old_x = ((first // kb) // nj * 128 * kb + step) * 210
                new_x = (it * 128 * kb + step) * 210
                old_y = ((first // kb) % nj) * 128 * 36 + columns * step * 2 * 36
                new_y = jt * 128 * 36 + columns * step * 2 * 36
                if (old_x, old_y) != (new_x, new_y):
                    raise ValueError("native operand address changed")
                addresses += 2
            for stride in (rows, rows + 19):
                for row, column in ((0, 0), (0, 127), (127, 0), (127, 127), (63, 65)):
                    identity = tuple(range(128))
                    old_at = jt * 128 * stride + it * 128 + identity[column] * stride + row
                    new_at = (jt * 128 + column) * stride + it * 128 + row
                    if old_at != new_at or not (0 <= new_at < stride * columns):
                        raise ValueError("identity writer address changed")
                    addresses += 1
            tiles += 1
            k_iterations += kb
    return {"status": "PASS_CPU_PARTITION_WRITER_EQUIVALENCE", "shapes": len(SHAPES),
            "tiles": tiles, "ordered_k_iterations": k_iterations, "addresses": addresses,
            "writer_outputs_per_tile": len(local_outputs), "guard_rejections": guard_rejections}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, default=pathlib.Path("/src"))
    parser.add_argument("--output-root", type=pathlib.Path,
                        default=pathlib.Path("/build/q6-dense-entry/generated"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--diff", action="store_true")
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    original, generated, files, metadata = prepare(source_root)
    if args.self_test:
        metadata["mapping_check"] = self_test()
    if not args.check_only:
        destination = args.output_root.resolve()
        allowed = pathlib.Path("/build/q6-dense-entry").resolve()
        if not destination.is_relative_to(allowed) or destination.is_relative_to(source_root):
            raise ValueError("generated output must stay under /build/q6-dense-entry outside source")
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
            fromfile="original/ggml-cuda/mmq.cuh", tofile="generated/ggml-cuda/mmq.cuh"))
    metadata["check_only"] = args.check_only
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
