import argparse
import hashlib
import importlib.util
import json
import pathlib
import re
import sys

sys.dont_write_bytecode = True

PINS = {
    "src/ops/linear/gguf/ggml_bridge.cu":
        "670d0910869d8b1183b02d37f7ab2f554def75b9ff7ba7152562ce65327608e6",
    "tests/ops/linear/test_gguf_mmq_store.cu":
        "c88bcb861c57546cfe83cec1135019dafce50b165f814389921bc3f915d137b5",
    "tools/q6_raw32_probe/generate.py":
        "2d60fd3b60fe236de7773c16101f9a8fa4043465692dd150e5b5fe6b94f291a5",
    "tools/q4_folded_ffn_probe/whole_main.cu":
        "fc08be3e76b413eee22fcdac59b684fd8b700047ec73ec9e6dacbd2e8c23211d",
}
OBJECTS = {
    "vector.o": "3cec379e71f05a41ad006c17a63d3812064aa62b726f4449a85181945b99fd9b",
    "mmq.o": "5c899fcc75acffd45d503cf6442b24ecf1f9651916b46170bb6ad8ec7db4591e",
}
MANIFEST = "fd279cc932ab67de54bae63b5b00aab5480f76a81020fa785512bc01cabada21"
OWNED = ("main.cu", "codec.h", "generate_host.py", "build_main.sh")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def prepare(source, probe, objects):
    raw = {}
    for name, expected in PINS.items():
        path = probe.parent.parent / name if name.startswith("tools/") else source / name
        raw[name] = path.read_bytes()
        require(sha(raw[name]) == expected, "raw helper pin changed: " + str(path))
    extractor = module(probe.parent / "q6_raw32_probe/generate.py", "q6wg_host_raw_extract")
    frozen_generator = module(probe / "generate.py", "q6wg_frozen_generator")
    _, native_receipt = frozen_generator.prepare(source)
    manifest_raw = (objects / "build_manifest.json").read_bytes()
    require(sha(manifest_raw) == MANIFEST, "selected two-object build manifest changed")
    manifest = json.loads(manifest_raw)
    require(manifest["expected_cuda_version"] == "13.1.115" and manifest["candidate_enabled"]
            and manifest["compile_only"] and len(manifest["units"]) == 2,
            "host requires the reviewed CUDA 13.1.115 two-object compile")
    require(sha((objects / "toolchain.txt").read_bytes()) == manifest["toolchain_sha256"],
            "selected toolchain receipt changed")
    for name, expected in manifest["probe_sha256"].items():
        require(sha((probe / name).read_bytes()) == expected, "compiled probe source changed: " + name)
    for unit in manifest["units"]:
        name = unit["object"]
        require(name in OBJECTS and unit["returncode"] == 0
                and unit["artifact_sha256"][name] == OBJECTS[name]
                and sha((objects / name).read_bytes()) == OBJECTS[name],
                "existing compiled object does not match the reviewed identity")
        for name, expected in unit["dependency_sha256"].items():
            path = pathlib.Path(name)
            require(sha(path.read_bytes()) == expected, "compiled dependency changed: " + name)
    bridge = raw["src/ops/linear/gguf/ggml_bridge.cu"]
    oracle = raw["tests/ops/linear/test_gguf_mmq_store.cu"]
    common = raw["tools/q4_folded_ffn_probe/whole_main.cu"]
    files = {}
    regions = {}
    specs = (
        ("q6wg_native_vector_producer.cuh", bridge,
         "constexpr int kQuantizeThreads = 256;", "using VectorLaunch ="),
        ("q6wg_native_matrix_producer.cuh", bridge,
         "constexpr int kMatrixQuantizeThreads = 128;",
         "template <mmq_q8_1_ds_layout layout_a, mmq_q8_1_ds_layout layout_b>"),
        ("q6wg_original_oracle.cuh", oracle,
         "double q6_runtime_weight_value(", "int check_q6_runtime_oracle("),
        ("q6wg_make_input.cuh", oracle,
         "std::vector<__nv_bfloat16> make_input(", "bool compare_rounding("),
        ("q6wg_sha256.cuh", common, "std::string sha256(", "std::vector<std::uint8_t> pinned_fixture("),
    )
    for name, data, first, after in specs:
        begin, end, content = extractor.span(data, first, after)
        files[name] = content
        regions[name] = {"sha256": sha(content), "source_lines": [
            data[:begin].count(b"\n") + 1, data[:end].count(b"\n")],
            "raw_extraction_without_substitutions": True}
    metadata = {
        "schema": 1, "scope": "isolated_Q6_word_group_correctness_and_exploratory_operator_screen",
        "build_manifest_sha256": MANIFEST, "linked_objects": OBJECTS,
        "cuda_version": "13.1.115", "host_source_sha256": {
            name: sha((probe / name).read_bytes()) for name in OWNED},
        "raw_helper_pins": PINS, "native13_pins": native_receipt["source_pins"],
        "raw_regions": regions, "kernel_recompiled": False, "gpu_run": False,
        "strict_sass_gate_rerun": False, "static_admission": "parent_decision",
        "launch_wrapper": "__cudaLaunchKernel",
        "fixture_pins": {
            "weights/q6-head-full-k5120.bin": {
                "rows": 248320, "k": 5120, "bytes": 1042944000,
                "sha256": "2155d6e1f7c5549077438c4760e6e150945dd5e406484b29162dff7b03d05da0"},
            "operators/q6-full-k17408.bin": {
                "rows": 5120, "k": 17408, "bytes": 73113600,
                "sha256": "a381c6be36f572170ccc9c7e4a3f8b25e5061e4eb633649399f3ec9759cb43e2"},
        },
    }
    receipt = json.dumps(metadata, separators=(",", ":"), sort_keys=True)
    files["q6wg_host_receipt.h"] = (
        "#pragma once\ninline constexpr const char* kQ6WGHostReceiptJson = "
        'R"q6wgreceipt(' + receipt + ')q6wgreceipt";\n').encode()
    metadata["generated_sha256"] = {name: sha(data) for name, data in files.items()}
    return files, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, default=pathlib.Path("/src"))
    parser.add_argument("--probe-root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parent)
    parser.add_argument("--objects-root", type=pathlib.Path, default=pathlib.Path("/build/q6-word-group"))
    parser.add_argument("--output-root", type=pathlib.Path, default=pathlib.Path("/build/q6-word-group/host/generated"))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    source, probe, objects = (x.resolve() for x in (args.source_root, args.probe_root, args.objects_root))
    files, metadata = prepare(source, probe, objects)
    if not args.check_only:
        out = args.output_root.resolve()
        require(out.is_relative_to(objects / "host") and not out.is_relative_to(source)
                and not out.is_relative_to(probe), "host generation must stay in existing build/host")
        out.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            require(not name.startswith("q6_word_group_"), "candidate header replacement is forbidden")
            (out / name).write_bytes(data)
    metadata["check_only"] = args.check_only
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
