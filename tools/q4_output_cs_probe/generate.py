import argparse
import difflib
import hashlib
import json
import pathlib
import sys


SOURCE_DIRECTORY = pathlib.Path("third_party/ggml-quants/src/ggml-cuda")
CHANGED_HEADER = "mmq.cuh"
SWITCH = "NINFER_GGUF_Q4_MMQ_OUTPUT_CS"
GUARD = SWITCH + " && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 860)"
EXPECTED_SHA256 = {
    "mmq.cuh": "493ce59f8be4ff269fd0a2a8386f9d1e9a80bd2ee9c7478083d36bc2ac9ccf45",
    "mmq-load-tiles.cuh": "80a7e02c1b2019db4ebdc1ef398a405c0249174dcf13a3d74b5d29af0347e3cf",
    "mmq-vec-dot.cuh": "e47f3be4edcf96341b9ad48df24bd87480051c50bf7cabeef251abcb6ae39c66",
}
ORIGINAL_TEMPLATE = """template<ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_write_back_mma(
"""
ENABLED_TEMPLATE = """template<ggml_type type, int J, bool fallback, bool output_cs = false>
static __device__ __forceinline__ void ggml_cuda_mmq_write_back_mma(
"""
ORIGINAL_STORE = """                } else {
                    dst[ids_dst[j]*stride + i] = sum[(j0/tile_C::J + n)*tile_C::ne + l];
                    GGML_UNUSED(y_scale_used);
                }
"""
ENABLED_STORE = """                } else {
                    if constexpr (output_cs && type == GGML_TYPE_Q4_K && J == 128) {
                        __stcs(dst + ids_dst[j]*stride + i, sum[(j0/tile_C::J + n)*tile_C::ne + l]);
                    } else {
                        dst[ids_dst[j]*stride + i] = sum[(j0/tile_C::J + n)*tile_C::ne + l];
                    }
                    GGML_UNUSED(y_scale_used);
                }
"""
ORIGINAL_CALL = """            write_back(sum, ids_dst, dst, y_scale, stride_col_dst, tile_x_max_i, tile_y_max_j);
"""
ENABLED_CALL = """            if constexpr (type == GGML_TYPE_Q4_K && J == 128) {
                ggml_cuda_mmq_write_back_mma<type, J, fallback, true>(
                    sum, ids_dst, dst, y_scale, stride_col_dst, tile_x_max_i, tile_y_max_j);
            } else {
                write_back(sum, ids_dst, dst, y_scale, stride_col_dst, tile_x_max_i, tile_y_max_j);
            }
"""
FRAGMENTS = (
    ("writer_template", ORIGINAL_TEMPLATE, ENABLED_TEMPLATE),
    ("writer_store", ORIGINAL_STORE, ENABLED_STORE),
    ("final_fp32_call", ORIGINAL_CALL, ENABLED_CALL),
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def guarded(original, enabled, define_default):
    prefix = ""
    if define_default:
        prefix = "#ifndef " + SWITCH + "\n#define " + SWITCH + " 0\n#endif\n"
    return prefix + "#if " + GUARD + "\n" + enabled + "#else\n" + original + "#endif\n"


def prepare(source_root):
    originals = {}
    for name, expected in EXPECTED_SHA256.items():
        data = (source_root / SOURCE_DIRECTORY / name).read_bytes()
        if sha256(data) != expected:
            raise ValueError("source SHA256 mismatch: " + name)
        originals[name] = data
    source = originals[CHANGED_HEADER]
    if SWITCH.encode("ascii") in source:
        raise ValueError("source already contains output streaming")
    newline = "\r\n" if b"\r\n" in source else "\n"
    replacements = []
    generated = source
    details = []
    for index, (name, original, enabled) in enumerate(FRAGMENTS):
        old = original.replace("\n", newline).encode("utf-8")
        new = guarded(original, enabled, index == 0).replace("\n", newline).encode("utf-8")
        if source.count(old) != 1 or generated.count(old) != 1:
            raise ValueError("expected exactly one fragment: " + name)
        start = source.index(old)
        line = source[:start].count(b"\n") + 1
        generated = generated.replace(old, new, 1)
        replacements.append((old, new))
        details.append({
            "name": name,
            "original_lines": [line, line + old.count(b"\n") - 1],
            "original_fragment_sha256": sha256(old),
            "generated_fragment_sha256": sha256(new),
        })
    restored = generated
    disabled = generated
    for old, new in reversed(replacements):
        if restored.count(new) != 1:
            raise ValueError("generated fragment is not unique")
        restored = restored.replace(new, old, 1)
        suffix = ("#endif" + newline).encode("ascii")
        disabled_fragment = new.split(("#else" + newline).encode("ascii"), 1)[1]
        if not disabled_fragment.endswith(suffix):
            raise ValueError("generated fragment has no closing switch guard")
        disabled_fragment = disabled_fragment[:-len(suffix)]
        if disabled_fragment != old:
            raise ValueError("disabled branch differs from original")
        disabled = disabled.replace(new, disabled_fragment, 1)
    if restored != source or disabled != source:
        raise ValueError("generated header does not round-trip to original bytes")
    files = dict(originals)
    files[CHANGED_HEADER] = generated
    metadata = {
        "status": "PASS_BOUNDED_WRITER_TRANSFORM",
        "switch": SWITCH,
        "default": 0,
        "architecture": 860,
        "type": "GGML_TYPE_Q4_K",
        "J": 128,
        "final_fp32_only": True,
        "store_intrinsic": "__stcs(float*, float)",
        "source_sha256": EXPECTED_SHA256[CHANGED_HEADER],
        "generated_header_sha256": sha256(generated),
        "source_headers": dict(EXPECTED_SHA256),
        "generated_headers": {name: sha256(data) for name, data in files.items()},
        "changed_header": CHANGED_HEADER,
        "changed_fragments": len(replacements),
        "fragments": details,
        "disabled_header_sha256": sha256(disabled),
        "disabled_header_byte_identical": True,
        "siblings_byte_identical": all(
            files[name] == originals[name] for name in EXPECTED_SHA256 if name != CHANGED_HEADER
        ),
        "codegen_qualified": False,
        "performance_claim": False,
    }
    return originals, files, metadata, replacements


def self_test(originals, files, replacements):
    source = originals[CHANGED_HEADER]
    generated = files[CHANGED_HEADER]
    bf16_start = b"static __device__ __forceinline__ void ggml_cuda_mmq_write_back_bf16("
    bf16_end = b"struct ggml_cuda_mmq_util_funcs {"
    if source[source.index(bf16_start):source.index(bf16_end)] != \
            generated[generated.index(bf16_start):generated.index(bf16_end)]:
        raise ValueError("BF16 writer changed")
    compute_start = b"template <ggml_type type, int J, bool fallback, bool fixup,"
    old_call, new_call = replacements[-1]
    if source[source.index(compute_start):source.index(old_call)] != \
            generated[generated.index(compute_start):generated.index(new_call)]:
        raise ValueError("tile computation, child CTA, BF16 route, or fixup write changed")
    if source[source.index(old_call) + len(old_call):] != \
            generated[generated.index(new_call) + len(new_call):]:
        raise ValueError("kernel routing or fixup reduction changed")
    if generated.count(b"__stcs(") != source.count(b"__stcs(") + 1:
        raise ValueError("expected exactly one new streaming store call")
    return {
        "status": "PASS_SOURCE_ISOLATION",
        "bf16_writer_byte_identical": True,
        "tile_compute_and_prior_routes_byte_identical": True,
        "kernel_routing_and_fixup_reduction_byte_identical": True,
        "additional_streaming_store_calls": 1,
        "inline_asm_added": False,
        "compiler_memory_clobber_added": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, default=pathlib.Path("/src"))
    parser.add_argument("--output-root", type=pathlib.Path,
                        default=pathlib.Path("/build/q4-output-cs/generated"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--diff", action="store_true")
    args = parser.parse_args()
    originals, files, metadata, replacements = prepare(args.source_root.resolve())
    if args.self_test:
        metadata["source_isolation_check"] = self_test(originals, files, replacements)
    if not args.check_only:
        destination = args.output_root.resolve() / "ggml-cuda"
        source_directory = (args.source_root.resolve() / SOURCE_DIRECTORY).resolve()
        if destination == source_directory or destination.is_relative_to(args.source_root.resolve()):
            raise ValueError("generated files must be outside the source tree")
        for name, data in files.items():
            path = destination / name
            if path.exists() and path.read_bytes() != data:
                raise ValueError("refusing to overwrite a different generated header: " + name)
        destination.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            path = destination / name
            if not path.exists():
                with path.open("xb") as handle:
                    handle.write(data)
            if sha256(path.read_bytes()) != metadata["generated_headers"][name]:
                raise ValueError("generated header verification failed: " + name)
        metadata["output_root"] = str(destination.parent)
    metadata["check_only"] = args.check_only
    if args.diff:
        sys.stdout.writelines(difflib.unified_diff(
            originals[CHANGED_HEADER].decode("utf-8").splitlines(keepends=True),
            files[CHANGED_HEADER].decode("utf-8").splitlines(keepends=True),
            fromfile="original/ggml-cuda/" + CHANGED_HEADER,
            tofile="generated/ggml-cuda/" + CHANGED_HEADER,
        ))
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
