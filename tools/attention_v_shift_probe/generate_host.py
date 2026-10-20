import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys

sys.dont_write_bytecode = True

PINS = {
    "tests/ops/softmax_attention/causal_cache.cpp":
        "0ee83c9cb0d01f92512b696017ab2b1e906fa4ecea9ae45e8b4349eadf36ca89",
    "tests/ops/softmax_attention/oracle.h":
        "a53af99811e3f00ce31baeb05b0a5bba4bd21bbecea60bf9f200cba7447db27d",
    "tests/ops/op_check.h":
        "da41427a183a367805ce713a7697f21747e3699c9f402ad87c36094973c24cd6",
    "tests/ops/op_tester.h":
        "1a48c936715c6a0afa2ef1ee6389af45e6afe3674777e7c908f9f72793947b1f",
    "include/ninfer/ops/attention_geometry.h":
        "32cc81e045773e2b3119f2971030ad4878a8090a1b6556cb6fb45ad2230249f4",
}
FROZEN = {
    "generate.py": "cb252ee03bffbf414ea2dc2ec96adf93b30ecd59aa6f953fdbcf77275ce1c8c8",
    "variant.cu": "63d701e5c6680dfa645080ae6121552e92caea9108ef1e6742adeeefd7ca3a80",
    "probe_api.h": "8502a3ae87dd1ce19828615440407dde92c6609b93ced5546c58f362a97f6d83",
    "build.sh": "f4c9d4d35dd471c34b7a109322f72d93b420177b0652ec8205a1940f99babb6e",
}
ARTIFACTS = {
    "off.o": "0b3dfa49b6017dc59f9545a12d94adb205b95eb755401269faad70b671db2f69",
    "on.o": "22214d9e34b536756d11dc107b00eadf2f390f7ebd736d1f3549364ff04b0ae7",
    "off.d": "c070da9d8e824c80b4da5002f136c12ecf27cc378fb0184796d4934ae4dbb259",
    "on.d": "0ba33aafab9777c772e4fee29633b5611c004951d3afcd85d294d6e67a6b961e",
    "toolchain.txt": "d51da6fa301a0c63b030c6a9e45b942c5fe2228e63c0ec276a445943cf0bc6f6",
    "common-flags.txt": "831eb4fa3025a889fcc900b8bc309f821cb72a6929029abb993dce11211c2d6f",
    "generation.json": "e21448b13b15792d20dd25313d1a969ffb94af86419f4304d410bc9b1d59f5bf",
    "generated/attention_v_shift_generated.cuh":
        "948973b0c236143d2c14586276c8a98f768ba07e0cc8a6b64e43299cdb5379ca",
}
DEPENDENCIES = "724b1137d46e612b9ed12260f2ccd77996681e0c9bba52e9053f0baa5e036471"
OWNED = ("main.cu", "generate_host.py", "build_main.sh")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def span(raw, first, after):
    newline = b"\r\n" if b"\r\n" in raw else b"\n"
    first = first.encode().replace(b"\n", newline)
    after = after.encode().replace(b"\n", newline)
    require(raw.count(first) == 1 and raw.count(after) == 1,
            "helper extraction anchors must both be unique: " + repr((first, after)))
    begin, end = raw.index(first), raw.index(after)
    require(begin < end, "helper extraction anchors are reversed")
    return raw[begin:end], [raw[:begin].count(b"\n") + 1, raw[:end].count(b"\n")]


def dependency_list(path):
    return path.read_text().replace("\\\n", " ").split(": ", 1)[1].split()


def prepare(source, probe, objects, source_only=False):
    for name, expected in FROZEN.items():
        require(sha((probe / name).read_bytes()) == expected, "frozen source changed: " + name)
    raw = {}
    for name, expected in PINS.items():
        raw[name] = (source / name).read_bytes()
        require(sha(raw[name]) == expected, "raw helper source changed: " + name)
    spec = importlib.util.spec_from_file_location("vshift_frozen_generator", probe / "generate.py")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    generated, native_receipt = native.prepare(source)
    require(sha(generated) == ARTIFACTS["generated/attention_v_shift_generated.cuh"],
            "frozen generated kernel no longer reproduces")
    dependency_count = None
    if not source_only:
        for name, expected in ARTIFACTS.items():
            require(sha((objects / name).read_bytes()) == expected, "compiled artifact changed: " + name)
        names = dependency_list(objects / "off.d")
        require(names == dependency_list(objects / "on.d"), "OFF/ON dependency sets differ")
        dependencies = []
        for name in names:
            path = pathlib.Path(name) if name.startswith("/") else source / name
            dependencies.append((name, sha(path.read_bytes())))
        require(sha(json.dumps(dependencies, separators=(",", ":")).encode()) == DEPENDENCIES,
                "reviewed 428-file compiled dependency tree changed")
        dependency_count = len(dependencies)
        require(dependency_count == 428, "compiled dependency count changed")
    causal = "tests/ops/softmax_attention/causal_cache.cpp"
    tester = "tests/ops/op_tester.h"
    specs = (
        ("conversions_and_random", tester, "inline float bf16_to_f32(",
         "// --- host <-> device"),
        ("constants", causal, "constexpr std::int32_t kHeadDim", "constexpr std::int32_t kFp8QuantGroup "),
        ("scale_and_canary", causal, "constexpr float kAttentionScale", "// The Op contract"),
        ("criterion", causal, "constexpr ReductionCriterion kAttentionInt8Criterion{",
         "constexpr ReductionCriterion kAttentionRk8v4Criterion{"),
        ("geometry", causal, "struct Geometry {", "struct AttentionCase {"),
        ("mapping_and_logical_address", causal, "enum class MappingPattern",
         "// Packed rk8v4 value plane:"),
        ("physical_address_and_scatter", causal, "std::size_t paged_index(",
         "template <typename T>\nvoid scatter_paged_into("),
        ("fixture_conversions", causal, "std::vector<float> make_bf16_values(",
         "float decode_e4m3fn_positive("),
        ("int8_encoder", causal, "void encode_group(", "// rk8v4 value group encoding:"),
        ("hadamard", causal, "void normalized_hadamard_d256(std::array<float, kHeadDim>&",
         "void encode_nvfp4_rotated_row("),
        ("rotated_key_encoder", causal, "void encode_rotated_key_row(", "// rk4v4 key coding:"),
    )
    parts, regions = [], {}
    for name, source_name, first, after in specs:
        content, lines = span(raw[source_name], first, after)
        parts.append(content)
        regions[name] = {"source": source_name, "source_lines": lines,
                         "sha256": sha(content), "raw_without_substitutions": True}
    files = {
        "attention_v_shift_host_helpers.cuh": b"\n".join(parts),
        "attention_v_shift_oracle.h": raw["tests/ops/softmax_attention/oracle.h"],
        "attention_v_shift_check.h": raw["tests/ops/op_check.h"],
    }
    require(b"/*relative_l2*/ 3.15e-3" in files["attention_v_shift_host_helpers.cuh"],
            "native INT8 criterion missing")
    require(b"__global__" not in files["attention_v_shift_host_helpers.cuh"],
            "host helper extraction captured CUDA kernels")
    metadata = {
        "schema": 1, "scope": "V_shift_paired_correctness_and_exploratory_attention_screen",
        "cuda_version": "13.1.115", "architecture": "sm_86",
        "compiled_artifact_pins": ARTIFACTS, "compiled_dependency_sha256": DEPENDENCIES,
        "compiled_dependency_count": dependency_count, "compiled_dependencies_checked": not source_only,
        "frozen_source_sha256": FROZEN, "host_source_sha256": {
            name: sha((probe / name).read_bytes()) for name in OWNED},
        "raw_helper_pins": PINS, "native_source_pins": native_receipt["source_pins"],
        "raw_regions": regions, "kernel_recompiled": False, "gpu_run": False,
        "launch_wrapper": "__cudaLaunchKernel", "source_only": source_only,
        "short_cases_tokens_base": [[1, 192], [33, 158], [127, 130], [129, 126]],
        "long_case": {"tokens": 1280, "base": 31488, "context_capacity": 65536,
                      "FP64_token_head_rows": [[0, 0], [127, 7], [128, 12], [1279, 23]],
                      "synthetic_represented_inputs": True},
        "relative_l2": 3.15e-3, "gross_absolute": 1.1e-3,
        "gross_relative_to_max_reference": 2.0 / 256.0,
        "screen": {"warmup_order": "ABAB", "block_orders": ["ABBA", "BAAB", "ABBA"],
                   "calls_per_sample": 32, "long_case_only": True,
                   "scope": "full_attention", "statistical_confirmation": False},
    }
    receipt = json.dumps(metadata, separators=(",", ":"), sort_keys=True)
    files["attention_v_shift_host_receipt.h"] = (
        "#pragma once\ninline constexpr const char* kAttentionVShiftHostReceiptJson = "
        'R"avshost(' + receipt + ')avshost";\n').encode()
    metadata["generated_sha256"] = {name: sha(data) for name, data in files.items()}
    return files, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, default=pathlib.Path("/src"))
    parser.add_argument("--probe-root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parent)
    parser.add_argument("--objects-root", type=pathlib.Path, default=pathlib.Path("/build/attention-v-shift"))
    parser.add_argument("--output-root", type=pathlib.Path,
                        default=pathlib.Path("/build/attention-v-shift/host/generated"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    require(not args.source_only or args.check_only, "source-only cannot generate build inputs")
    source, probe, objects = (p.resolve() for p in (args.source_root, args.probe_root, args.objects_root))
    files, metadata = prepare(source, probe, objects, args.source_only)
    if not args.check_only:
        out = args.output_root.resolve()
        require(out.is_relative_to(objects / "host") and not out.is_relative_to(source)
                and not out.is_relative_to(probe), "host generation must stay in existing build/host")
        out.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (out / name).write_bytes(content)
    metadata["check_only"] = args.check_only
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
