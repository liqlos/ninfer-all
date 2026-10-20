import argparse
import difflib
import hashlib
import json
import pathlib
import random
import sys


SOURCE_DIRECTORY = pathlib.Path("third_party/ggml-quants/src/ggml-cuda")
SWITCH = "NINFER_GGUF_Q4_MMQ_COMPACT_A"
DEVICE_GUARD = SWITCH + " && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 860)"
EXPECTED_SHA256 = {
    "mmq.cuh": "493ce59f8be4ff269fd0a2a8386f9d1e9a80bd2ee9c7478083d36bc2ac9ccf45",
    "mmq-load-tiles.cuh": "80a7e02c1b2019db4ebdc1ef398a405c0249174dcf13a3d74b5d29af0347e3cf",
    "mmq-vec-dot.cuh": "e47f3be4edcf96341b9ad48df24bd87480051c50bf7cabeef251abcb6ae39c66",
}
MMA_SHA256 = "08a18ccf4400990e9b2a651fdf702f49f0a521e427f151443e4359c912ac4fab"
BASELINE_OBJECT_SHA256 = "a32d451568dde06b834b8b34dfadbb2434ea0ee53c6a68b590cd06d1b946ddc5"
MARKERS = {
    "mmq.cuh": "NINFER_Q4_COMPACT_A_GENERATED_MMQ",
    "mmq-load-tiles.cuh": "NINFER_Q4_COMPACT_A_GENERATED_LOAD",
    "mmq-vec-dot.cuh": "NINFER_Q4_COMPACT_A_GENERATED_DOT",
}
LOAD_SCOPE = "static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q4_K("
DOT_SCOPE = "static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma("
HOST_STRIDE = """static __host__ int ggml_cuda_mmq_get_sram_stride(const ggml_type type, const int J, const bool fallback, const int cc) {
    return ggml_cuda_mmq_get_sram_stride(ggml_cuda_mmq_get_sram_layout(type, J, fallback, cc));
}
"""
DEVICE_STRIDE = """static constexpr __device__ int ggml_cuda_mmq_get_sram_stride(ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8) {
    return ggml_cuda_mmq_get_sram_stride(ggml_cuda_mmq_get_sram_layout(type, J, fallback, prec_src1));
}
"""
SHARED_X = """static __host__ int ggml_cuda_mmq_get_nbytes_shared_x(const ggml_cuda_mmq_config & config, const int cc) {
    if (config.use_mma_data_layout(cc)) {
        return config.I * ggml_cuda_mmq_get_sram_stride(config.sram_layout) * 4;
    }
    const tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(config.type, config.I);
    return (txs.qs + txs.dm + txs.sc) * 4;
}
"""
LOAD_DM = """    half2 * x_dm = (half2 *) (x_qs + 2*MMQ_TILE_NE_K);
"""
LOAD_QS = """        x_qs[i*sram_stride + 16*(txi/8) + txi % 8 + 0] = (qs0 >> 0) & 0x0F0F0F0F;
        x_qs[i*sram_stride + 16*(txi/8) + txi % 8 + 8] = (qs0 >> 4) & 0x0F0F0F0F;
"""
DOT_DM = """    const half2 * x_dm = (const half2 *) x_qs + 2*MMQ_TILE_NE_K;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_dm = (const half2 *) y;

    const int i0 = (threadIdx.y/ntx)*rows_per_warp;
"""
PRELOAD = """#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            const int k0 = k00 + k01;

            load_ldmatrix(A[n][k01/QI8_1], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }
"""
COMPACT_PRELOAD = """        if constexpr (type == GGML_TYPE_Q4_K && J == 128) {
#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 2*QI8_1) {
                tile_A packed;
                load_ldmatrix(packed, x_qs + (i0 + n*tile_A::I)*sram_stride + (k00 + k01)/2, sram_stride);
#pragma unroll
                for (int l = 0; l < tile_A::ne; ++l) {
                    const uint32_t qs = static_cast<uint32_t>(packed.x[l]);
                    A[n][k01/QI8_1 + 0].x[l] = qs & 0x0F0F0F0F;
                    A[n][k01/QI8_1 + 1].x[l] = (qs >> 4) & 0x0F0F0F0F;
                }
            }
        } else {
""" + PRELOAD + """        }
"""
FRAGMENTS = tuple(
    (header, "header_provenance", "", "#pragma once\n",
     "#pragma once\n#define " + marker + " 1\n", SWITCH)
    for header, marker in MARKERS.items()
) + (
    ("mmq.cuh", "host_stride", "", HOST_STRIDE,
     HOST_STRIDE.replace("    return ", """    if (type == GGML_TYPE_Q4_K && J == 128 && cc == 860) {
        return 44;
    }
    return """, 1), SWITCH),
    ("mmq.cuh", "device_stride", "", DEVICE_STRIDE,
     DEVICE_STRIDE.replace("    return ", """    if (type == GGML_TYPE_Q4_K && J == 128 && prec_src1 == GGML_PREC_Q8) {
        return 44;
    }
    return """, 1), DEVICE_GUARD),
    ("mmq.cuh", "shared_allocation", "", SHARED_X,
     SHARED_X.replace("    if (config.use_mma_data_layout(cc)) {", """    if (config.type == GGML_TYPE_Q4_K && config.J == 128 && cc == 860) {
        return config.I * 44 * 4;
    }
    if (config.use_mma_data_layout(cc)) {""", 1), SWITCH),
    ("mmq-load-tiles.cuh", "producer_metadata_offset", LOAD_SCOPE, LOAD_DM,
     """    constexpr int dm_offset = type == GGML_TYPE_Q4_K && J == 128 ? MMQ_TILE_NE_K : 2*MMQ_TILE_NE_K;
    half2 * x_dm = (half2 *) (x_qs + dm_offset);
""", DEVICE_GUARD),
    ("mmq-load-tiles.cuh", "producer_raw_nibbles", LOAD_SCOPE, LOAD_QS,
     """        if constexpr (type == GGML_TYPE_Q4_K && J == 128) {
            x_qs[i*sram_stride + txi] = qs0;
        } else {
""" + LOAD_QS + """        }
""", DEVICE_GUARD),
    ("mmq-vec-dot.cuh", "consumer_metadata_offset", DOT_SCOPE, DOT_DM,
     DOT_DM.replace("    const half2 * x_dm = (const half2 *) x_qs + 2*MMQ_TILE_NE_K;",
                    """    constexpr int dm_offset = type == GGML_TYPE_Q4_K && J == 128 ? MMQ_TILE_NE_K : 2*MMQ_TILE_NE_K;
    const half2 * x_dm = (const half2 *) x_qs + dm_offset;""", 1), DEVICE_GUARD),
    ("mmq-vec-dot.cuh", "expanded_a_preload", DOT_SCOPE, PRELOAD,
     COMPACT_PRELOAD, DEVICE_GUARD),
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def guarded(original, enabled, guard, define_default):
    prefix = ""
    if define_default:
        prefix = "#ifndef " + SWITCH + "\n#define " + SWITCH + " 0\n#endif\n"
    return prefix + "#if " + guard + "\n" + enabled + "#else\n" + original + "#endif\n"


def scope_bounds(data, anchor):
    if not anchor:
        return 0, len(data)
    start = data.index(anchor.encode("utf-8"))
    end = data.index(b"\ntemplate ", start)
    return start, end


def prepare(source_root):
    originals = {}
    for name, expected in EXPECTED_SHA256.items():
        data = (source_root / SOURCE_DIRECTORY / name).read_bytes()
        require(sha256(data) == expected, "source SHA256 mismatch: " + name)
        require(SWITCH.encode("ascii") not in data, "source already contains compact A")
        originals[name] = data
    mma = (source_root / SOURCE_DIRECTORY / "mma.cuh").read_bytes()
    require(sha256(mma) == MMA_SHA256, "source SHA256 mismatch: mma.cuh")
    files = dict(originals)
    replacements = []
    details = []
    marked = set()
    for header, name, scope, original, enabled, guard in FRAGMENTS:
        source = originals[header]
        newline = "\r\n" if b"\r\n" in source else "\n"
        old = original.replace("\n", newline).encode("utf-8")
        new = guarded(original, enabled, guard, header not in marked).replace("\n", newline).encode("utf-8")
        source_start, source_end = scope_bounds(source, scope)
        require(source[source_start:source_end].count(old) == 1,
                "expected exactly one source fragment in scope: " + name)
        current = files[header]
        start, end = scope_bounds(current, scope)
        require(current[start:end].count(old) == 1,
                "expected exactly one current fragment in scope: " + name)
        offset = current.index(old, start, end)
        files[header] = current[:offset] + new + current[offset + len(old):]
        replacements.append((header, old, new))
        line = source[:source.index(old, source_start, source_end)].count(b"\n") + 1
        details.append({
            "header": header,
            "name": name,
            "original_lines": [line, line + old.count(b"\n") - 1],
            "original_fragment_sha256": sha256(old),
            "generated_fragment_sha256": sha256(new),
        })
        marked.add(header)
    restored = dict(files)
    disabled = dict(files)
    for header, old, new in reversed(replacements):
        require(restored[header].count(new) == 1, "generated fragment is not unique")
        restored[header] = restored[header].replace(new, old, 1)
        newline = b"\r\n" if b"\r\n" in new else b"\n"
        disabled_fragment = new.rsplit(b"#else" + newline, 1)[1]
        suffix = b"#endif" + newline
        require(disabled_fragment.endswith(suffix), "missing closing switch guard")
        disabled_fragment = disabled_fragment[:-len(suffix)]
        require(disabled_fragment == old, "disabled branch differs from original")
        disabled[header] = disabled[header].replace(new, disabled_fragment, 1)
    require(restored == originals and disabled == originals, "header round-trip failed")
    metadata = {
        "status": "PASS_BOUNDED_COMPACT_A_TRANSFORM",
        "switch": SWITCH,
        "default": 0,
        "architecture": 860,
        "type": "GGML_TYPE_Q4_K",
        "I": 128,
        "J": 128,
        "warps": 8,
        "sram_stride_words": 44,
        "dynamic_shared_bytes": 41472,
        "full_expanded_a_preload": True,
        "source_headers": dict(EXPECTED_SHA256),
        "mma_header_sha256": MMA_SHA256,
        "generated_headers": {name: sha256(data) for name, data in files.items()},
        "disabled_headers": {name: sha256(data) for name, data in disabled.items()},
        "disabled_headers_byte_identical": True,
        "changed_fragments": len(replacements),
        "dataflow_fragments": 7,
        "provenance_fragments": 3,
        "fragments": details,
        "generated_sibling_markers": MARKERS,
        "baseline_object": "/build/q4-k-stream/off.o",
        "baseline_object_sha256": BASELINE_OBJECT_SHA256,
        "baseline_object_verified_here": False,
        "codegen_qualified": False,
        "performance_claim": False,
    }
    return originals, files, metadata, replacements


def ldmatrix_word_addresses(base, stride):
    providers = [base + (lane % 16) * stride + (lane // 16) * 4
                 for lane in range(32)]
    require(all(address % 4 == 0 for address in providers), "unaligned ldmatrix row")
    return [[providers[8 * matrix + lane // 4] + lane % 4
             for matrix in range(4)] for lane in range(32)]


def source_isolation_proof(originals, files, replacements):
    for header, old, new in replacements:
        require(new.count(b"#if " + (SWITCH + " && defined(__CUDA_ARCH__)").encode("ascii"))
                == (0 if old == b"#pragma once\n" or old == b"#pragma once\r\n"
                    or header == "mmq.cuh" and b"static __host__" in old else 1),
                "unexpected device guard")
    for header, marker in MARKERS.items():
        require(files[header].count(("#define " + marker + " 1").encode("ascii")) == 1,
                "missing generated sibling marker")
    for name in ("mmq-load-tiles.cuh", "mmq-vec-dot.cuh"):
        include = ('#include "' + name + '"').encode("ascii")
        require(originals["mmq.cuh"].count(include) == 1
                and files["mmq.cuh"].count(include) == 1, "sibling include changed")
    require(len(files) == 3, "all three scoped siblings are required")
    start = b"template <ggml_type type, int J, bool fallback, bool fixup,"
    require(originals["mmq.cuh"].split(start, 1)[1]
            == files["mmq.cuh"].split(start, 1)[1], "tile schedule or kernel dispatch changed")
    start = b"static __device__ __forceinline__ void ggml_cuda_mmq_write_back_dp4a("
    end = b"template <ggml_type type, int J, bool fallback, bool fixup,"
    for header in ("mmq.cuh",):
        a, b = originals[header], files[header]
        require(a[a.index(start):a.index(end)] == b[b.index(start):b.index(end)],
                "output writers or kernel utility changed")
    require(b"for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J)" not in
            COMPACT_PRELOAD.encode("ascii"), "decode moved into N loop")
    for token in ("mma(", "sum[", "__half22float2(", "__syncthreads(", "cp.async", "__stcs("):
        added = sum(new.count(token.encode()) - old.count(token.encode())
                    for _, old, new in replacements)
        require(added == 0, "arithmetic, barriers, pipeline or output store changed: " + token)
    return {
        "status": "PASS_SOURCE_ISOLATION",
        "ordered_compute_and_pipeline_byte_identical": True,
        "output_writers_byte_identical": True,
        "all_three_siblings_required_by_main": True,
        "resident_gguf_load_and_scale_arithmetic_unchanged": True,
        "new_mma_or_fp32_arithmetic_or_barriers": 0,
    }


def fragment_mapping_proof():
    native = {}
    compact = {}
    for warp in range(8):
        for i0 in range(0, 128, 8):
            row = i0 + warp
            for lane in range(32):
                symbols = tuple((row, 4 * lane + byte) for byte in range(4))
                low = tuple((r, index, 0) for r, index in symbols)
                high = tuple((r, index, 4) for r, index in symbols)
                native[row * 76 + 16 * (lane // 8) + lane % 8] = low
                native[row * 76 + 16 * (lane // 8) + lane % 8 + 8] = high
                compact[row * 44 + lane] = symbols
    require(len(native) == 128 * 64 and len(compact) == 128 * 32, "producer coverage")
    symbolic_words = 0
    mapping = []
    bank_transactions = 0
    for half in range(2):
        k00 = half * 32
        for warp in range(8):
            i0 = (warp // 2) * 32
            for n in range(2):
                row = i0 + n * 16
                for pair in range(2):
                    packed_addresses = ldmatrix_word_addresses(row * 44 + k00 // 2 + 8 * pair, 44)
                    for matrix in range(4):
                        addresses = [packed_addresses[lane][matrix] for lane in range(32)]
                        require(len({address % 32 for address in addresses}) == 32,
                                "compact ldmatrix bank conflict in modeled 8x8 transaction")
                        bank_transactions += 1
                    for high in range(2):
                        segment = 4 * half + 2 * pair + high
                        native_addresses = ldmatrix_word_addresses(row * 76 + 8 * segment, 76)
                        for lane in range(32):
                            for reg in range(4):
                                old = native[native_addresses[lane][reg]]
                                new = tuple((r, index, 4 * high)
                                            for r, index in compact[packed_addresses[lane][reg]])
                                require(old == new, "native A fragment symbol mismatch")
                                source_row = row + (reg % 2) * 8 + lane // 4
                                word_col = (reg // 2) * 4 + lane % 4
                                expected = tuple(
                                    (source_row, (32 * segment + 4 * word_col + byte) // 64 * 32
                                     + (32 * segment + 4 * word_col + byte) % 32, 4 * high)
                                    for byte in range(4)
                                )
                                require(new == expected, "independent GGUF coordinate mismatch")
                                mapping.append((native_addresses[lane][reg],
                                                packed_addresses[lane][reg], high, expected))
                                symbolic_words += 1
    byte_cases = 0
    for byte in range(4):
        for value in range(256):
            raw_word = (0xA55A3CC3 & ~(255 << (8 * byte))) | (value << (8 * byte))
            signed_word = raw_word if raw_word < 2**31 else raw_word - 2**32
            for shift in (0, 4):
                old = (signed_word >> shift) & 0x0F0F0F0F
                new = (raw_word >> shift) & 0x0F0F0F0F
                independent = sum((((raw_word >> (8 * b)) & 255) // 2**shift % 16) << (8 * b)
                                  for b in range(4))
                require(old == new == independent, "signed/raw word nibble mismatch")
                byte_cases += 1
    random_words = 0
    for seed in (0, 1, 0x3090, 0x860):
        rng = random.Random(seed)
        raw = [bytes(rng.randrange(256) for _ in range(128)) for _ in range(128)]
        expanded_memory = {}
        packed_memory = {}
        for row in range(128):
            for txi in range(32):
                value = int.from_bytes(raw[row][4 * txi:4 * txi + 4], "little", signed=True)
                expanded_memory[row * 76 + 16 * (txi // 8) + txi % 8] = value & 0x0F0F0F0F
                expanded_memory[row * 76 + 16 * (txi // 8) + txi % 8 + 8] = (value >> 4) & 0x0F0F0F0F
                packed_memory[row * 44 + txi] = value & 0xFFFFFFFF
        for old_address, new_address, high, coordinates in mapping:
            expected = expanded_memory[old_address]
            actual = (packed_memory[new_address] >> (4 * high)) & 0x0F0F0F0F
            independent = sum(((raw[row][index] // 2**shift) % 16) << (8 * b)
                              for b, (row, index, shift) in enumerate(coordinates))
            require(expected == actual == independent, "random fragment mismatch")
            random_words += 1
    old_dm = {}
    new_dm = {}
    for warp in range(8):
        for lane in range(32):
            row = warp * 16 + lane // 2
            for l in range(4):
                group = 4 * (lane % 2) + l
                old_address = row * 76 + 64 + group
                new_address = row * 44 + 32 + group
                require(old_address not in old_dm and new_address not in new_dm,
                        "metadata has multiple producers")
                old_dm[old_address] = (row, group)
                new_dm[new_address] = (row, group)
                require(old_address % 32 == new_address % 32, "metadata store bank differs")
    metadata_reads = 0
    for half in range(2):
        for warp in range(8):
            for n in range(2):
                for l in range(2):
                    for lane in range(32):
                        row = (warp // 2) * 32 + n * 16 + l * 8 + lane // 4
                        for ki in range(4):
                            group = 4 * half + ki
                            a, b = row * 76 + 64 + group, row * 44 + 32 + group
                            require(old_dm[a] == new_dm[b] == (row, group), "metadata mapping mismatch")
                            require(a % 32 == b % 32, "metadata load bank differs")
                            metadata_reads += 1
    require(set(compact).isdisjoint(new_dm), "raw nibbles overlap metadata")
    require(max(set(compact) | set(new_dm)) < 128 * 44, "compact tile out of bounds")
    require(128 * 4 + 128 * 144 + 128 * 44 * 4 == 41472, "shared byte accounting")
    return {
        "status": "PASS_CPU_NATIVE_FRAGMENT_MAPPING",
        "model": "PTX m8n8.x4.b16 non-transposed; matrix register order 0,1,2,3",
        "ptx_reference": "https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#warp-level-matrix-load-instruction-ldmatrix",
        "symbolic_fragment_words": symbolic_words,
        "symbolic_fragment_bytes": 4 * symbolic_words,
        "independent_gguf_coordinates_checked": True,
        "exhaustive_byte_and_shift_cases": byte_cases,
        "random_fragment_words": random_words,
        "metadata_producer_words": len(new_dm),
        "metadata_consumer_reads": metadata_reads,
        "metadata_banks_identical": True,
        "modeled_conflict_free_ldmatrix_transactions": bank_transactions,
        "raw_and_metadata_disjoint": True,
        "all_ldmatrix_rows_16_byte_aligned": True,
        "native_x_useful_store_bytes_per_k256": 128 * 72 * 4,
        "compact_x_useful_store_bytes_per_k256": 128 * 40 * 4,
        "native_a_fragment_load_bytes_per_k256": 128 * 64 * 4 * 2,
        "compact_a_fragment_load_bytes_per_k256": 128 * 32 * 4 * 2,
        "shared_accounting_is_measured_traffic": False,
        "gpu_correctness_qualified": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, default=pathlib.Path("/src"))
    parser.add_argument("--output-root", type=pathlib.Path,
                        default=pathlib.Path("/build/q4-compact-a/generated"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--diff", action="store_true")
    args = parser.parse_args()
    originals, files, metadata, replacements = prepare(args.source_root.resolve())
    if args.self_test:
        metadata["source_isolation"] = source_isolation_proof(originals, files, replacements)
        metadata["fragment_mapping"] = fragment_mapping_proof()
    if not args.check_only:
        destination = args.output_root.resolve() / "ggml-cuda"
        require(not destination.is_relative_to(args.source_root.resolve()),
                "generated files must be outside the source tree")
        for name, data in files.items():
            path = destination / name
            require(not path.exists() or path.read_bytes() == data,
                    "refusing to overwrite a different generated header: " + name)
        destination.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            path = destination / name
            if not path.exists():
                with path.open("xb") as handle:
                    handle.write(data)
            require(sha256(path.read_bytes()) == metadata["generated_headers"][name],
                    "generated header verification failed: " + name)
        metadata["output_root"] = str(destination.parent)
    metadata["check_only"] = args.check_only
    if args.diff:
        for name in EXPECTED_SHA256:
            sys.stdout.writelines(difflib.unified_diff(
                originals[name].decode("utf-8").splitlines(keepends=True),
                files[name].decode("utf-8").splitlines(keepends=True),
                fromfile="original/ggml-cuda/" + name,
                tofile="generated/ggml-cuda/" + name,
            ))
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
