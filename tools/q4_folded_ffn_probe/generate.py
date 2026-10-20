import argparse
import difflib
import hashlib
import json
import pathlib
import random
import re
import sys


SOURCE_DIRECTORY = pathlib.Path("third_party/ggml-quants/src/ggml-cuda")
SWITCH = "NINFER_GGUF_FFN_FOLDED16_MMQ"
EXPECTED_SHA256 = {
    "mmq.cuh": "493ce59f8be4ff269fd0a2a8386f9d1e9a80bd2ee9c7478083d36bc2ac9ccf45",
    "mmq-load-tiles.cuh": "80a7e02c1b2019db4ebdc1ef398a405c0249174dcf13a3d74b5d29af0347e3cf",
    "mmq-vec-dot.cuh": "e47f3be4edcf96341b9ad48df24bd87480051c50bf7cabeef251abcb6ae39c66",
}
DEPENDENCY_SHA256 = {
    "third_party/ggml-quants/src/ggml-cuda/mma.cuh":
        "08a18ccf4400990e9b2a651fdf702f49f0a521e427f151443e4359c912ac4fab",
    "third_party/ggml-quants/src/ggml-cuda/mmq-config-ampere.cuh":
        "6de9336a6c2f65c0ad83cbe9303dcd03e8cbb6185d15346133a31421000f6408",
    "third_party/ggml-quants/src/ggml-common.h":
        "0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c",
    "src/ops/linear/gguf/ggml_bridge_internal.cuh":
        "8b78c42a2cb01b10e1227987cae22b3d0a030684070d6f35a213b033fb3c9d90",
}
BASELINE_OBJECT_SHA256 = "a32d451568dde06b834b8b34dfadbb2434ea0ee53c6a68b590cd06d1b946ddc5"
MARKERS = {
    "mmq.cuh": "NINFER_Q4_FOLDED16_GENERATED_MMQ",
    "mmq-load-tiles.cuh": "NINFER_Q4_FOLDED16_GENERATED_LOAD",
    "mmq-vec-dot.cuh": "NINFER_Q4_FOLDED16_GENERATED_DOT",
}
LOAD_START = (
    b"template <ggml_type type, int J, bool fallback, int I_OVERRIDE = -1> "
    b"static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q4_K("
)
LOAD_END = b"#if defined(NINFER_GGUF_Q4_MMQ_ASYNC_X)"
POINTER_OLD = b"const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride"
POINTER_NEW = (
    b"const block_q4_K * bxi = (const block_q4_K *) ((i & 16) != 0 ? up : x)"
    b" + kbx0 + (intermediate_row0 + 16*(i/32) + i%16)*stride"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(data, old, new):
    require(data.count(old) == 1, "expected one exact source fragment")
    return data.replace(old, new, 1)


def guarded(old, enabled, newline):
    flag = SWITCH.encode("ascii")
    return (
        b"#ifndef " + flag + newline + b"#define " + flag + b" 0" + newline +
        b"#endif" + newline + b"#if " + flag + newline + enabled +
        b"#else" + newline + old + b"#endif" + newline
    )


def clone_loader(native):
    transforms = (
        (b"ggml_cuda_mmq_load_tiles_q4_K(", b"ggml_cuda_mmq_load_tiles_q4_K_folded16("),
        (b"const int stride) {",
         b"const int stride, const char * __restrict__ up, const int intermediate_row0) {"),
        (POINTER_OLD, POINTER_NEW),
    )
    clone = native
    for old, new in transforms:
        expected = 4 if old == POINTER_OLD else 1
        require(clone.count(old) == expected, "native loader fragment count changed")
        clone = clone.replace(old, new)
    restored = clone
    for old, new in reversed(transforms):
        restored = restored.replace(new, old)
    require(restored == native, "loader inverse transform is not byte exact")
    require(clone.count(b"__syncthreads") == native.count(b"__syncthreads") == 0,
            "loader synchronization changed")
    for token in (b"get_int_b4(", b"unpack_scales_q45_K(", b"make_half2(",
                  b"x_qs[", b"x_dm[", b"x_sc["):
        require(clone.count(token) == native.count(token), "loader arithmetic changed")
    return clone


def prepare(source_root):
    originals = {}
    for name, expected in EXPECTED_SHA256.items():
        data = (source_root / SOURCE_DIRECTORY / name).read_bytes()
        require(sha256(data) == expected, "source pin mismatch: " + name)
        originals[name] = data
    for name, expected in DEPENDENCY_SHA256.items():
        require(sha256((source_root / name).read_bytes()) == expected,
                "dependency pin mismatch: " + name)
    files = dict(originals)
    replacements = []
    for name, marker in MARKERS.items():
        newline = b"\r\n" if b"\r\n" in files[name] else b"\n"
        old = b"#pragma once" + newline
        enabled = old + b"#define " + marker.encode("ascii") + b" 1" + newline
        new = guarded(old, enabled, newline)
        files[name] = replace_once(files[name], old, new)
        replacements.append((name, old, new))
    name = "mmq-load-tiles.cuh"
    source = originals[name]
    require(source.count(LOAD_START) == 1, "Q4 native loader is not unique")
    start = source.index(LOAD_START)
    stop = source.index(LOAD_END, start)
    native = source[start:stop]
    require(native.rstrip().endswith(b"}"), "native loader end moved")
    clone = clone_loader(native)
    newline = b"\r\n" if b"\r\n" in native else b"\n"
    new = guarded(native, native + clone, newline)
    files[name] = replace_once(files[name], native, new)
    replacements.append((name, native, new))
    disabled = dict(files)
    for name, old, new in reversed(replacements):
        disabled[name] = replace_once(disabled[name], new, old)
    require(disabled == originals, "OFF projection/inverse transform changed original bytes")
    require(files["mmq-vec-dot.cuh"].replace(replacements[2][2], replacements[2][1], 1)
            == originals["mmq-vec-dot.cuh"], "dot header was changed")
    require(files["mmq.cuh"].replace(replacements[0][2], replacements[0][1], 1)
            == originals["mmq.cuh"], "native MMQ schedule was changed")
    metadata = {
        "status": "PASS_BOUNDED_FOLDED16_TRANSFORM",
        "switch": SWITCH,
        "default": 0,
        "architecture": 860,
        "shape": {"k": 5120, "rows": 17408, "columns": 1280, "row_bytes": 2880},
        "tile": {"I": 128, "J": 128, "warps": 8, "final_rows": 64,
                 "accumulators_per_thread": 64, "shared_bytes": 57856, "ctas": 2720},
        "source_headers": dict(EXPECTED_SHA256),
        "source_dependencies": dict(DEPENDENCY_SHA256),
        "generated_headers": {name: sha256(data) for name, data in files.items()},
        "disabled_headers": {name: sha256(data) for name, data in disabled.items()},
        "disabled_headers_byte_identical": True,
        "native_loader_inverse_byte_identical": True,
        "native_dot_header_byte_identical_except_provenance": True,
        "native_mmq_header_byte_identical_except_provenance": True,
        "native_loader_source_pointer_sites": 4,
        "generated_sibling_markers": MARKERS,
        "baseline_object": "/build/q4-k-stream/off.o",
        "baseline_object_sha256": BASELINE_OBJECT_SHA256,
        "baseline_object_verified_here": False,
        "diagnostic_fp32_stores": "template Diagnostic=true only",
        "activation_pack_included": False,
        "gpu_correctness_qualified": False,
        "codegen_qualified": False,
        "performance_claim": False,
    }
    return originals, files, metadata


def folded_row(logical_row):
    return ((logical_row // 16) % 2, 16 * (logical_row // 32) + logical_row % 16)


def braced_statement(source, anchor):
    require(source.count(anchor) == 1, "source schedule anchor is not unique")
    start = source.index(anchor)
    opening = source.index(b"{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == ord("{"):
            depth += 1
        elif source[index] == ord("}"):
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise RuntimeError("source schedule has no closing brace")


def schedule_proof(originals, variant):
    native_anchor = (
        b"for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {"
        b"\n        if constexpr (child_cta)"
    )
    native = braced_statement(originals["mmq.cuh"], native_anchor)
    new = native
    transforms = (
        (b"for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter)",
         b"for (int kb0 = 0; kb0 < blocks_per_k; ++kb0)"),
        (b"if constexpr (child_cta) { ggml_cuda_mmq_load_tiles_q4_K<type, J, fallback, 64>"
         b"(x, tile_x, offset_x + kb0 + child_row_offset*stride_row_x, "
         b"tile_x_max_i-child_row_offset, stride_row_x); }\n"
         b"        else { load_tiles(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x); }",
         b"ggml_cuda_mmq_load_tiles_q4_K_folded16<GGML_TYPE_Q4_K, kJ, false>"
         b"(gate, tile_x, kb0, kI - 1, stride_row, up, row0);"),
        (b"if constexpr (child_cta) { ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma"
         b"<type, J, fallback, true>(tile_x, tile_y, sum, 0); }\n"
         b"        else { vec_dot(tile_x, tile_y, sum, 0); }",
         b"vec_dot(tile_x, tile_y, sum, 0);"),
        (b"if constexpr (child_cta) { ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma"
         b"<type, J, fallback, true>(tile_x, tile_y, sum, MMQ_TILE_NE_K); }\n"
         b"        else { vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K); }",
         b"vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K);"),
    )
    for old, replacement in transforms:
        new = replace_once(new, old, replacement)
    new = re.sub(rb"\bJ\b", b"kJ", new)
    new = re.sub(rb"\bnwarps\b", b"kWarps", new)
    actual = braced_statement(variant, b"for (int kb0 = 0; kb0 < blocks_per_k; ++kb0)")
    require(re.sub(rb"\s+", b"", actual) == re.sub(rb"\s+", b"", new),
            "folded runtime K schedule differs from the native full-K loop")
    require(actual.count(b"__syncthreads();") == 4, "barrier count changed")
    require(variant.count(b"#pragma unroll 1\n    for (int kb0") == 1,
            "runtime K loop must retain explicit no-unroll")
    return {
        "status": "PASS_NATIVE_K_SCHEDULE",
        "native_full_k_source_sha256": sha256(native),
        "folded_runtime_k_source_sha256": sha256(actual),
        "native_loop_equal_after_explicit_pointer_and_bound_substitution": True,
        "y_loads_dot_calls_barriers_order_preserved": True,
        "barriers_per_k256": 4,
        "runtime_k_bound": True,
    }


def ldmatrix_addresses(base, stride):
    providers = [base + (lane % 16) * stride + (lane // 16) * 4 for lane in range(32)]
    require(all(address % 4 == 0 for address in providers), "unaligned ldmatrix row")
    return [[providers[8 * matrix + lane // 4] + lane % 4
             for matrix in range(4)] for lane in range(32)]


def mapping_proof():
    expected_rows = {(operand, row) for operand in range(2) for row in range(64)}
    require({folded_row(i) for i in range(128)} == expected_rows, "row map is not bijective")
    native_qs = {}
    metadata = {}
    for warp in range(8):
        for i0 in range(0, 128, 8):
            row = i0 + warp
            operand, intermediate = folded_row(row)
            for lane in range(32):
                for high in range(2):
                    address = row * 76 + 16 * (lane // 8) + lane % 8 + 8 * high
                    require(address not in native_qs, "duplicate q4 producer")
                    native_qs[address] = tuple(
                        (operand, intermediate, 4 * lane + byte, 4 * high) for byte in range(4)
                    )
        for lane in range(32):
            row = warp * 16 + lane // 2
            operand, intermediate = folded_row(row)
            for l in range(4):
                group = 4 * (lane % 2) + l
                address = row * 76 + 64 + group
                require(address not in metadata, "duplicate metadata producer")
                metadata[address] = (operand, intermediate, group)
    require(len(native_qs) == 8192 and len(metadata) == 1024, "producer coverage")
    require(set(native_qs).isdisjoint(metadata), "q4 data overlaps metadata")
    require(max(set(native_qs) | set(metadata)) < 128 * 76, "shared tile out of bounds")
    fragment_cases = []
    metadata_reads = 0
    for half in range(2):
        for warp in range(8):
            i0 = (warp // 2) * 32
            for n in range(2):
                for ki in range(4):
                    segment = half * 4 + ki
                    addresses = ldmatrix_addresses((i0 + n * 16) * 76 + 8 * segment, 76)
                    for lane in range(32):
                        for reg in range(4):
                            logical_row = i0 + n * 16 + (reg % 2) * 8 + lane // 4
                            operand, row = folded_row(logical_row)
                            word_column = (reg // 2) * 4 + lane % 4
                            k0 = 32 * segment + 4 * word_column
                            expected = tuple(
                                (operand, row, ((k0 + byte) // 64) * 32 + (k0 + byte) % 32,
                                 4 * (((k0 + byte) // 32) % 2))
                                for byte in range(4)
                            )
                            actual = native_qs[addresses[lane][reg]]
                            require(actual == expected, "independent GGUF fragment mismatch")
                            require(operand == n, "gate/up A fragment ownership mismatch")
                            fragment_cases.append(actual)
                    for lane in range(32):
                        for l in range(2):
                            logical_row = i0 + n * 16 + l * 8 + lane // 4
                            operand, row = folded_row(logical_row)
                            actual = metadata[logical_row * 76 + 64 + segment]
                            require(actual == (operand, row, segment),
                                    "metadata does not follow the q4 source row")
                            metadata_reads += 1
    random_words = 0
    for seed in (0, 1, 0x3090, 0x860):
        rng = random.Random(seed)
        raw = [[bytes(rng.randrange(256) for _ in range(128)) for row in range(64)]
               for operand in range(2)]
        for coordinates in fragment_cases:
            operand, row, index, shift = coordinates[0]
            require(all((a, b, c, d) == (operand, row, index + byte, shift)
                        for byte, (a, b, c, d) in enumerate(coordinates)),
                    "fragment byte continuity changed")
            word = int.from_bytes(raw[operand][row][index:index + 4], "little", signed=True)
            actual = (word >> shift) & 0x0F0F0F0F
            expected = sum(
                ((raw[a][b][c] // (2 ** d)) % 16) << (8 * byte)
                for byte, (a, b, c, d) in enumerate(coordinates)
            )
            require(actual == expected, "random signed nibble decode differs")
            random_words += 1
    owners = {}
    pairs = {}
    for warp in range(8):
        for lane in range(32):
            slots = set()
            for j0 in range(0, 128, 16):
                for n in range(2):
                    for l in range(4):
                        logical_row = (warp // 2) * 32 + n * 16 + (l // 2) * 8 + lane // 4
                        column = j0 + (warp % 2) * 8 + 2 * (lane % 4) + l % 2
                        slot = (j0 // 8 + n) * 4 + l
                        operand, row = folded_row(logical_row)
                        key = (operand, row, column)
                        require(key not in owners, "duplicate MMA output owner")
                        owners[key] = (warp, lane, slot)
                        slots.add(slot)
                        if n == 0:
                            pairs[(row, column)] = (warp, lane, slot, slot + 4)
            require(slots == set(range(64)), "each thread must own exactly 64 native accumulators")
    require(len(owners) == 16384 and len(pairs) == 8192, "MMA output coverage")
    for (row, column), (warp, lane, gate_slot, up_slot) in pairs.items():
        require(owners[(0, row, column)] == (warp, lane, gate_slot), "gate slot mismatch")
        require(owners[(1, row, column)] == (warp, lane, up_slot), "up slot mismatch")
        require(row == 16 * (warp // 2) + 8 * ((gate_slot % 4) // 2) + lane // 4,
                "epilogue row mapping differs")
    require({row for row, column in pairs} == set(range(64)), "missing final row")
    require({column for row, column in pairs} == set(range(128)), "missing final column")
    global_rows = [64 * block + row for block in range(272) for row in range(64)]
    global_columns = [128 * block + column for block in range(10) for column in range(128)]
    require(global_rows == list(range(17408)), "actual-shape row boundary")
    require(global_columns == list(range(1280)), "actual-shape column boundary")
    require(len(pairs) * 272 * 10 == 17408 * 1280, "actual-shape output count")
    y_words = set()
    for kb in range(20):
        for half in range(2):
            for column_block in range(10):
                for linear in range(128 * 36):
                    address = (column_block * 128 * 36 +
                               1280 * (2 * kb + half) * 36 + linear)
                    require(address not in y_words, "activation block loaded into two tiles")
                    y_words.add(address)
    require(len(y_words) == 40 * 1280 * 36 and min(y_words) == 0 and
            max(y_words) == 40 * 1280 * 36 - 1, "activation global bounds")
    k_order = [kb * 256 + half * 128 + ki * 32
               for kb in range(20) for half in range(2) for ki in range(4)]
    require(k_order == list(range(0, 5120, 32)), "native K update order changed")
    return {
        "status": "PASS_CPU_FOLDED16_MAPPING",
        "logical_rows": 128,
        "q4_producer_words": len(native_qs),
        "metadata_producer_words": len(metadata),
        "symbolic_fragment_words": len(fragment_cases),
        "random_fragment_words": random_words,
        "metadata_consumer_reads": metadata_reads,
        "native_accumulator_owners": len(owners),
        "same_thread_gate_up_pairs": len(pairs),
        "actual_shape_output_elements": 17408 * 1280,
        "actual_shape_activation_words": len(y_words),
        "native_k32_update_offsets": k_order,
        "shared_sram_stride_words": 76,
        "shared_bytes": (128 + 128 * 36 + 128 * 76) * 4,
        "gpu_correctness_qualified": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, default=pathlib.Path("/src"))
    parser.add_argument("--output-root", type=pathlib.Path,
                        default=pathlib.Path("/build/q4-folded-ffn/generated"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--diff", action="store_true")
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    originals, files, metadata = prepare(source_root)
    probe_root = pathlib.Path(__file__).resolve().parent
    if args.self_test:
        metadata["mapping"] = mapping_proof()
        metadata["schedule"] = schedule_proof(originals, (probe_root / "variant.cu").read_bytes())
    metadata["probe_files"] = {
        name: sha256((probe_root / name).read_bytes())
        for name in ("generate.py", "variant.cu", "probe_api.h", "build.sh")
    }
    if not args.check_only:
        destination = args.output_root.resolve() / "ggml-cuda"
        require(not destination.is_relative_to(source_root),
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
