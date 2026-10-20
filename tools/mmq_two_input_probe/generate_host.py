import argparse
import hashlib
import json
import pathlib
import sys

sys.dont_write_bytecode = True

import generate


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FROZEN = {
    "build.py": "1b7dfaceabecb9c8f2f178af7f1012ddff3734158ff4d345c70e73fb52f7780f",
    "generate.py": "9ae873e9bfb617e605015bfda11cba11a0008adf0754e2b37f25f8f09ac1a3ba",
    "proof.py": "160956d2a3aacb0fac55bf90468ffa335e988c8e4f1b7b9d41f14f3acf71bde3",
    "variant.cu": "fa6834fbe0e1d947421970f8227a882820c6ebc762667774a06fb2c7b6cc8cee",
    "probe_api.h": "71359b24f97ed31e0a04d9fbdf9c815c3f2bcec6347152987bb93f3c6b8f39df",
}
PINS = {
    "src/ops/linear/gguf/ggml_bridge.cu":
        "670d0910869d8b1183b02d37f7ab2f554def75b9ff7ba7152562ce65327608e6",
    "src/ops/linear/gguf/ggml_bridge_internal.cuh":
        "8b78c42a2cb01b10e1227987cae22b3d0a030684070d6f35a213b033fb3c9d90",
    "tests/ops/linear/test_gguf_mmq_store.cu":
        "c88bcb861c57546cfe83cec1135019dafce50b165f814389921bc3f915d137b5",
    "tools/q6_word_group_probe/main.cu":
        "697f2470a795cc2a1f976f330150d7715dd9811c030c7ff239dff8dc3b0c8e6d",
    "tools/q4_folded_ffn_probe/whole_main.cu":
        "fc08be3e76b413eee22fcdac59b684fd8b700047ec73ec9e6dacbd2e8c23211d",
    "tools/lab/mmq_swiglu_probe.cuh":
        "97045b5b86c5849896d4bbc311228d2686f31f84f96935ae13a41279d2d725f1",
}
PROTOCOL = {
    "cases": ["Q6Down", "Q4Gate", "Q4Up"],
    "fixtures": {
        "q6": {"path": "operators/q6-full-k17408.bin", "bytes": 73113600,
               "sha256": "a381c6be36f572170ccc9c7e4a3f8b25e5061e4eb633649399f3ec9759cb43e2"},
        "q4": {"path": "operators/q4-full-k5120.bin", "bytes": 100270080,
               "sha256": "be10db58b96b5614830c01afe1e28f9d55c4581a6c481827fa7b9d885a6e2790"},
    },
    "input": "two distinct BF16 panels from pinned make_input; real weights, synthetic represented inputs",
    "seeds": {"Q6Down": [303385, 303386], "Q4Gate": [303401, 303402],
              "Q4Up": [303417, 303418]},
    "columns": 1280, "samples": 2,
    "output_column_padding_elements": 16, "output_sample_padding_elements": 64,
    "packed_sample_padding_bytes": 256,
    "correctness_modes": ["PlainFp32", "StoreBf16", "AccumulateBf16"],
    "full_output_exact_comparison": True,
    "independent_FP64_samples_per_panel_per_arm_per_case": 64,
    "independent_full_single_call_BF16_semantics": True,
    "carry32_calls": 32, "carry32_full_exact_comparison": True,
    "carry32_independent_semantic_points_per_panel": 64,
    "API_negative_calls": "actual compiled calls for both arms, no kernel launches or output changes",
    "timing_mode": "StoreBf16", "timing_scope": "two_native_packs_plus_projection",
    "warmup_order": "ABAB", "calls_per_warmup_arm": 8,
    "calls_per_sample": 8, "block_orders": ["ABBA", "BAAB", "ABBA"],
    "screen_reduction": "100 * (1 - sum of six B samples / sum of six A samples)",
    "all_qualification_before_timing": True,
    "primary": "Q6Down", "primary_minimum_mean_time_reduction_percent": 10.0,
    "primary_required_positive_blocks": 3,
    "next_gate": "whole paired FFN against strongest folded16, without recrediting its fusion gain",
    "runtime_qualified": False, "full_ffn_qualified": False,
    "statistical_confirmation": False, "performance_claim": False,
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def prepare(root):
    native_guard, _ = generate.prepare(root)
    for name, expected in FROZEN.items():
        require(sha((HERE / name).read_bytes()) == expected, "frozen source changed: " + name)
    raw = {}
    for name, expected in PINS.items():
        path = (ROOT if name.startswith("tools/") else root) / name
        raw[name] = path.read_bytes()
        require(sha(raw[name]) == expected, "raw helper changed: " + name)
    host = "tools/q6_word_group_probe/main.cu"
    whole = "tools/q4_folded_ffn_probe/whole_main.cu"
    test = "tests/ops/linear/test_gguf_mmq_store.cu"
    specs = [
        ("host_support.cuh", host, 'extern "C" void ggml_abort(', "namespace q6wg_host_native {"),
        ("host_common.cuh", host, "void require(bool condition,", '#include "q6wg_original_oracle.cuh"'),
        ("host_guarded.cuh", host, "struct Guarded {", "struct Fixture {"),
        ("host_fixture_type.cuh", host, "struct Fixture {", "constexpr Fixture kHead"),
        ("host_read_fixture.cuh", host, "std::vector<std::uint8_t> read_fixture(", "struct Resident {"),
        ("host_samples.cuh", host, "std::vector<int> sample_indices(", "std::vector<std::size_t> oracle_positions("),
        ("host_launch_wrapper.cuh", host, 'extern "C" cudaError_t CUDARTAPI __real___cudaLaunchKernel(',
         "namespace {\n\nstd::string rejection_checks("),
        ("host_events.cuh", host, "struct Events {", "double timing_sample("),
        ("host_input.cuh", test, "std::vector<__nv_bfloat16> make_input(", "bool compare_rounding("),
        ("host_q6_decoder.cuh", test, "double q6_runtime_weight_value(", "int check_q6_runtime_oracle("),
        ("host_sha256.cuh", whole, "std::string sha256(", "std::vector<std::uint8_t> pinned_fixture("),
        ("host_half_bound.cuh", whole, "double half_round_bound(", "struct OracleStats {"),
        ("host_oracle_stats.cuh", whole, "struct OracleStats {", "OracleStats gate_oracle("),
        ("host_producer_oracle.cuh", whole, "int producer_oracle(", "std::pair<double, double> q4_down_oracle("),
        ("host_down_oracle.cuh", whole, "std::pair<double, double> q4_down_oracle(", "void run_case("),
        ("host_q4_oracle.cuh", "tools/lab/mmq_swiglu_probe.cuh",
         "float half_value(", "std::vector<__nv_bfloat16> make_input("),
        ("host_native_producer.cuh", "src/ops/linear/gguf/ggml_bridge.cu",
         "constexpr int kMatrixQuantizeThreads = 128;",
         "template <mmq_q8_1_ds_layout layout_a, mmq_q8_1_ds_layout layout_b>"),
    ]
    files, regions = {}, {}
    for name, source, first, after in specs:
        data = raw[source].replace(b"\r\n", b"\n")
        begin, end = first.encode(), after.encode()
        require(data.count(begin) == 1 and data.count(end) == 1,
                "nonunique raw helper boundary: " + name)
        start, stop = data.index(begin), data.index(end)
        require(start < stop, "reversed helper boundaries: " + name)
        files[name] = data[start:stop]
        regions[name] = {"source": source, "source_lines": [data[:start].count(b"\n") + 1,
                         data[:stop].count(b"\n")], "sha256": sha(files[name]),
                         "substitutions": [], "newline_normalization": "CRLF to LF only"}
    files["mmq_two_input_native.cuh"] = native_guard.encode()
    metadata = {
        "scope": "two independent native-packed T1280 panels; isolated operator region qualification",
        "kernel_recompiled": False, "frozen_source_sha256": FROZEN,
        "raw_helper_pins": PINS, "raw_regions": regions, "protocol": PROTOCOL,
        "host_source_sha256": {name: sha((HERE / name).read_bytes())
                               for name in ("main.cu", "build_host.py", "generate_host.py")},
        "GPU_run": False, "performance_claim": False,
    }
    receipt = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    files["host_receipt.h"] = (
        '#pragma once\ninline constexpr const char* kMmqTwoInputHostReceipt = R"mmqhost('
        + receipt + ')mmqhost";\n').encode()
    return files, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, default=ROOT)
    args = parser.parse_args()
    files, metadata = prepare(args.source_root.resolve())
    print(json.dumps({"source_check_only": True, "metadata": metadata,
                      "generated_sha256": {name: sha(data) for name, data in files.items()}}, indent=2))


if __name__ == "__main__":
    main()
