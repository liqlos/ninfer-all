import argparse
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import time


sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OBJECT_SHA = "e63473eb2bc492318605e878bf8d6ce22fb9e884c1d74dcbae44b035cd5d1b9b"
MANIFEST_CANONICAL_SHA = "c46dc49022faa80134030b1f6034a20b48498547a9887f60cf07a4aff3c8a0f4"
COMPILED_BUILD_SHA = "086f4ea3c3c81dc37f3316c4653b20f02ae27c68cdba7718a02c9cd75beeb657"
FROZEN = {
    "build.py": "fbcb328c224cf509df56c71e865a1c251b204b202ebbf19690152f61729a50b0",
    "chain_probe.cu": "d52838f3d918b3fbd02832b8b2c4082c5565cde33e8c1dc4161d46a4406c466a",
    "generate.py": "57e7683a4e8f6d551d54b5c1cf0d6dd0e6bb4d8a7ef2e9bdc8ce7d2e88151437",
    "probe_api.h": "1e97746ad5f99f98156d66b562494579672b6e118a228673541ca0ee752cbdf2",
    "symbolic.py": "141da134c82a9caa40b3e86283d4f6a76c95777e23842b825d2f5fac11ae0997",
}
PINS = {
    "src/ops/linear/gguf/ggml_bridge.cu":
        "670d0910869d8b1183b02d37f7ab2f554def75b9ff7ba7152562ce65327608e6",
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
    "fixture": "operators/q4-full-k5120.bin",
    "fixture_bytes": 100270080,
    "fixture_sha256": "be10db58b96b5614830c01afe1e28f9d55c4581a6c481827fa7b9d885a6e2790",
    "rows": [6144, 17408, 34816], "K": 5120, "T": 4,
    "correctness_modes": ["PlainFP32", "StoreBF16", "AccumulateBF16"],
    "correctness_shape_count": 3, "correctness_mode_count": 9,
    "guard_N1024": "NativeW execution plus Chain rejection",
    "oracle_samples_per_shape_per_arm": 64,
    "oracle_bound": "0.02 + 2e-4 * abs(FP64_reference)",
    "accumulate_direct_calls": 32, "graph_nodes": 4, "graph_replays": 8,
    "timing_endpoints": [
        {"N": n, "mode": "StoreBF16", "scope": scope,
         "primary": n == 34816 and scope == "prepared"}
        for n in (6144, 17408, 34816) for scope in ("projection_only", "prepared")],
    "prepared_scope": "current native BF16-to-planar-Q8 helperpack + projection + StoreBF16",
    "projection_only_scope": "common Q8 already packed; projection + StoreBF16",
    "warmup_order": "ABAB", "calls_per_sample": 32,
    "block_orders": ["ABBA", "BAAB", "ABBA"], "samples_per_block": 4,
    "primary_minimum_mean_time_gain_percent": 1.0,
    "primary_minimum_positive_blocks": 2,
    "guard_maximum_mean_time_regression_percent": 1.0,
    "screen_requires_all_correctness_before_timing": True,
    "statistical_confirmation": False, "end_to_end_claim": False,
    "fused_gate_up_qualified": False, "runtime_integration": False,
    "N6144": "prefix of actual FFN fixture; geometry/weights proxy",
    "N17408": "first gate half of actual FFN fixture",
    "N34816": "full actual concatenated gate/up weight matrix; single projection Store",
    "input": "pinned make_input BF16 seed 0x4a119+N, four declared 32-value edge groups",
    "input_first_stripe_overrides": [
        "T0: all zero", "T1: alternating +0.25,-0.25",
        "T2: constant 0.03125", "T3: (i-16)/128 for i=0..31"],
    "StoreBF16_epilogue": "same projection launch; included in both timing scopes",
    "existing_object_resource_status": "PASS",
    "existing_object_mechanism_review": "PARTIAL; numerical qualification remains pending",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def prepare(root):
    raw = {}
    for name, expected in PINS.items():
        path = (ROOT if name.startswith("tools/") else root) / name
        raw[name] = path.read_bytes()
        require(sha(raw[name]) == expected, "raw helper changed: " + name)
    for name, expected in FROZEN.items():
        require(sha((HERE / name).read_bytes()) == expected, "frozen probe file changed: " + name)
    host = "tools/q6_word_group_probe/main.cu"
    specs = [
        ("host_support.cuh", host, 'extern "C" void ggml_abort(', "namespace q6wg_host_native {"),
        ("host_common.cuh", host, "void require(bool condition,", '#include "q6wg_original_oracle.cuh"'),
        ("host_guarded.cuh", host, "struct Guarded {", "struct Fixture {"),
        ("host_samples.cuh", host, "std::vector<int> sample_indices(", "std::vector<std::size_t> oracle_positions("),
        ("host_output.cuh", host, "std::vector<std::uint8_t> initial_output(", "struct SemanticStats {"),
        ("host_launch_wrapper.cuh", host, 'extern "C" cudaError_t CUDARTAPI __real___cudaLaunchKernel(',
         "namespace {\n\nstd::string rejection_checks("),
        ("native_producer.cuh", "src/ops/linear/gguf/ggml_bridge.cu",
         "constexpr int kQuantizeThreads = 256;", "using VectorLaunch ="),
        ("host_input.cuh", "tests/ops/linear/test_gguf_mmq_store.cu",
         "std::vector<__nv_bfloat16> make_input(", "bool compare_rounding("),
        ("host_sha256.cuh", "tools/q4_folded_ffn_probe/whole_main.cu",
         "std::string sha256(", "std::vector<std::uint8_t> pinned_fixture("),
        ("host_q4_oracle.cuh", "tools/lab/mmq_swiglu_probe.cuh",
         "float half_value(", "std::vector<__nv_bfloat16> make_input("),
    ]
    files, regions = {}, {}
    for name, source, first, after in specs:
        data = raw[source].replace(b"\r\n", b"\n")
        begin, end = first.encode(), after.encode()
        require(data.count(begin) == 1 and data.count(end) == 1, "nonunique raw extraction: " + name)
        start, stop = data.index(begin), data.index(end)
        require(start < stop, "reversed raw extraction")
        files[name] = data[start:stop]
        regions[name] = {"source": source, "source_lines": [data[:start].count(b"\n") + 1,
                         data[:stop].count(b"\n")], "sha256": sha(files[name]),
                         "substitutions": [], "newline_normalization": "CRLF to LF only"}
    metadata = {
        "scope": "isolated Q4 T4 chain operator qualification and fixed exploratory screen",
        "kernel_object_sha256": OBJECT_SHA, "kernel_recompiled": False,
        "compiled_manifest_canonical_sha256": MANIFEST_CANONICAL_SHA,
        "compile_time_build_py_sha256": COMPILED_BUILD_SHA,
        "frozen_source_sha256": FROZEN, "raw_helper_pins": PINS, "raw_regions": regions,
        "host_source_sha256": {name: sha((HERE / name).read_bytes()) for name in ("main.cu", "build_main.py")},
        "protocol": PROTOCOL, "GPU_run": False,
    }
    text = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    files["host_receipt.h"] = (
        '#pragma once\ninline constexpr const char* kQ4TCHostReceipt = R"q4tc(' + text + ')q4tc";\n').encode()
    return files, metadata


def run(command, root, log=None):
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, **options)
    if log is not None:
        log.write_text(result.stdout + result.stderr, encoding="utf-8")
    require(result.returncode == 0, "command failed: " + " ".join(command) + "\n" + result.stdout + result.stderr)
    return result.stdout + result.stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--objects-root", type=pathlib.Path, default=pathlib.Path("/build/q4-t4-chain"))
    parser.add_argument("--output-root", type=pathlib.Path)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    root, objects = args.source_root.resolve(), args.objects_root.resolve()
    files, metadata = prepare(root)
    if not args.build:
        print(json.dumps({"source_check_only": True, "metadata": metadata,
                          "generated_sha256": {k: sha(v) for k, v in files.items()}}, sort_keys=True))
        return
    require(not os.environ.get("NVCC_PREPEND_FLAGS") and not os.environ.get("NVCC_APPEND_FLAGS"),
            "ambient NVCC flags must be empty")
    manifest_raw = (objects / "build_manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    require(sha(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()) ==
            MANIFEST_CANONICAL_SHA, "existing compile manifest changed")
    require(manifest["compile_returncode"] == 0 and manifest["expected_cuda_version"] == "13.1.115",
            "host requires the reviewed compile")
    require(sha((objects / "chain.o").read_bytes()) == OBJECT_SHA, "existing chain.o changed")
    for name, expected in manifest["compiled_dependency_sha256"].items():
        require(sha(pathlib.Path(name).read_bytes()) == expected, "compiled dependency changed: " + name)
    nvcc, dump, nm = (shutil.which(name) for name in ("nvcc", "cuobjdump", "nm"))
    require(nvcc and dump and nm, "existing CUDA 13.1 toolchain and nm required")
    version = run([nvcc, "--version"], root)
    require(re.search(r"\bV13\.1\.115\b", version), "different CUDA compiler")
    require(re.search(r"\bU __cudaLaunchKernel\b", run([nm, "-u", str(objects / "chain.o")], root)),
            "compiled launch ABI differs")
    out = (args.output_root or objects / "host").resolve()
    require(out.is_relative_to(objects / "host") and not out.is_relative_to(root) and
            not out.is_relative_to(HERE),
            "host output must remain in selected build/host")
    out.mkdir(parents=True, exist_ok=True)
    generated = out / "generated"
    generated.mkdir(exist_ok=True)
    for name, data in files.items():
        (generated / name).write_bytes(data)
    flags = ["-O3", "-std=c++17", "-arch=sm_86", "--extended-lambda", "-lineinfo", "-Xptxas=-v",
             "--frandom-seed=q4-t4-chain-host", "-DNINFER_Q4_T4_CHAIN_PROBE=1",
             "-DNINFER_GGUF_Q4_COOPERATIVE_DECODE=1", "-DNINFER_GGUF_Q6_COOPERATIVE_DECODE=1",
             "-DNINFER_GGUF_MMQ_STORE_FUSION=1"]
    for path in (HERE, generated, root / "src", root / "include",
                 root / "third_party/ggml-quants/include", root / "third_party/ggml-quants/src",
                 root / "third_party/ggml-quants/src/ggml-cuda"):
        flags.extend(["-I", str(path)])
    started = time.monotonic()
    compile_command = [nvcc, *flags, "--generate-dependencies-with-compile", "--dependency-output",
                       str(out / "main.d"), "-c", str(HERE / "main.cu"), "-o", str(out / "main.o")]
    run(compile_command, root, out / "compile-main.log")
    link_command = [nvcc, "-arch=sm_86", str(out / "main.o"), str(objects / "chain.o"),
                    "-Xlinker=--wrap=__cudaLaunchKernel", "-lcuda", "-o", str(out / "probe")]
    run(link_command, root, out / "link.log")
    dependencies = (out / "main.d").read_text().replace("\\\n", " ").split(":", 1)[1]
    paths = [(pathlib.Path(p) if pathlib.Path(p).is_absolute() else root / p).resolve()
             for p in shlex.split(dependencies)]
    metadata.update({"toolchain": version, "compile_command": compile_command,
                     "link_command": link_command, "compile_link_seconds": time.monotonic() - started,
                     "linked_existing_object": str(objects / "chain.o"),
                     "existing_manifest_raw_sha256": sha(manifest_raw),
                     "generated_sha256": {name: sha(data) for name, data in files.items()},
                     "host_dependency_sha256": {str(p): sha(p.read_bytes()) for p in sorted(set(paths))},
                     "host_object_sha256": sha((out / "main.o").read_bytes()),
                     "probe_sha256": sha((out / "probe").read_bytes())})
    (out / "resources-linked.txt").write_text(run([dump, "--dump-resource-usage", str(out / "probe")], root))
    (out / "host_build.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
