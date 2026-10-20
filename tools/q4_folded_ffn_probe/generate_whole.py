import argparse
import hashlib
import json
import pathlib


PINS = {
    "tools/lab/mmq_swiglu_probe.cuh":
        "97045b5b86c5849896d4bbc311228d2686f31f84f96935ae13a41279d2d725f1",
    "tools/ffn_pair_probe/probe_api.h":
        "969ab378b519cc3b365c188b323431ee0d360c16a8d3a512aa876bb245be0ec1",
    "src/ops/linear/gguf/ggml_bridge.cu":
        "670d0910869d8b1183b02d37f7ab2f554def75b9ff7ba7152562ce65327608e6",
    "src/ops/linear/gguf/ggml_bridge_mmq.cuh":
        "2513cb559034db1422841508c1c251e060ec087fe7b92643b2b539f687a7b15b",
    "src/ops/linear/gguf/ggml_bridge_internal.cuh":
        "8b78c42a2cb01b10e1227987cae22b3d0a030684070d6f35a213b033fb3c9d90",
    "tests/ops/linear/test_gguf_mmq_store.cu":
        "c88bcb861c57546cfe83cec1135019dafce50b165f814389921bc3f915d137b5",
    "third_party/ggml-quants/src/ggml-cuda/mmq.cuh":
        "493ce59f8be4ff269fd0a2a8386f9d1e9a80bd2ee9c7478083d36bc2ac9ccf45",
    "third_party/ggml-quants/src/ggml-cuda/mmq-load-tiles.cuh":
        "80a7e02c1b2019db4ebdc1ef398a405c0249174dcf13a3d74b5d29af0347e3cf",
    "third_party/ggml-quants/src/ggml-cuda/mmq-vec-dot.cuh":
        "e47f3be4edcf96341b9ad48df24bd87480051c50bf7cabeef251abcb6ae39c66",
}
FIXTURES = {
    "q4-full-k5120.bin": {
        "bytes": 100270080,
        "sha256": "be10db58b96b5614830c01afe1e28f9d55c4581a6c481827fa7b9d885a6e2790",
    },
    "q4-full-k17408.bin": {
        "bytes": 50135040,
        "sha256": "8a5bbd71b5016c0382895b156c554ff9909e7b7f1dccf1158a7de191b94ef3ce",
    },
    "q6-full-k17408.bin": {
        "bytes": 73113600,
        "sha256": "a381c6be36f572170ccc9c7e4a3f8b25e5061e4eb633649399f3ec9759cb43e2",
    },
}
OWNED = ("generate_whole.py", "down.cu", "whole_main.cu", "build_whole.sh", "whole_api.h")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def extract(data, first, after):
    require(data.count(first) == 1, "non-unique source start")
    start = data.index(first)
    require(data.count(after, start) == 1, "non-unique source end")
    stop = data.index(after, start)
    return data[start:stop]


def prepare(source_root):
    sources = {}
    for name, expected in PINS.items():
        data = (source_root / name).read_bytes()
        require(digest(data) == expected, "source pin mismatch: " + name)
        sources[name] = data
    harness = sources["tools/lab/mmq_swiglu_probe.cuh"]
    require(harness.count(b"int main(int argc, char** argv)") == 1, "main boundary moved")
    common = harness[:harness.index(b"int main(int argc, char** argv)")]
    timing = extract(common, b"double time_arm(", b"void verify_oracle(")
    common = common.replace(timing, b"", 1)
    require(b"run_candidate" not in common, "unresolved candidate function in common harness")
    producer = extract(
        sources["src/ops/linear/gguf/ggml_bridge.cu"],
        b"constexpr int kMatrixQuantizeThreads = 128;",
        b"template <mmq_q8_1_ds_layout layout_a, mmq_q8_1_ds_layout layout_b>",
    )
    decoder = extract(
        sources["tests/ops/linear/test_gguf_mmq_store.cu"],
        b"double q6_runtime_weight_value(", b"int check_q6_runtime_oracle(",
    )
    files = {
        "support/whole_common.cuh": common,
        "support/whole_native_producer.cuh": producer,
        "support/whole_q6_decode.cuh": decoder,
        "ffn_pair_probe/probe_api.h": sources["tools/ffn_pair_probe/probe_api.h"],
    }
    metadata = {
        "status": "PASS_WHOLE_FFN_SOURCE_EXTRACTION",
        "source_pins": PINS,
        "fixtures": FIXTURES,
        "generated_sha256": {name: digest(data) for name, data in files.items()},
        "removed_time_arm_sha256": digest(timing),
        "common_prefix_only_time_arm_removed": True,
        "native_matrix_producer_raw": True,
        "q6_decoder_raw": True,
        "gpu_qualified": False,
        "timing_qualified": False,
    }
    return files, metadata


def static_proof():
    require(17408 % 512 == 0, "native pack requires whole 512-value groups")
    require(136 * 1280 * 144 == 25067520, "packed payload size")
    require(68 * 144 == 9792 and 68 * 210 == 14280, "down row bytes")
    require(5120 * 9792 == FIXTURES["q4-full-k17408.bin"]["bytes"], "Q4 fixture extent")
    require(5120 * 14280 == FIXTURES["q6-full-k17408.bin"]["bytes"], "Q6 fixture extent")
    require(34816 * 20 * 144 == FIXTURES["q4-full-k5120.bin"]["bytes"], "gate fixture extent")
    for block in (0, 1, 67, 68, 135):
        for token in (0, 127, 128, 1279):
            base = (block * 1280 + token) * 144
            require(0 <= base and base + 144 <= 25067520, "packed byte extent")
            metadata = {base + 4 * group + byte for group in range(4) for byte in range(4)}
            quants = {base + 16 + index for index in range(128)}
            require(metadata.isdisjoint(quants) and len(metadata | quants) == 144, "pack ownership")
    orders = ((0, 1, 1, 0), (1, 0, 0, 1), (0, 1, 1, 0))
    require(all(order.count(0) == order.count(1) == 2 for order in orders), "balanced blocks")
    require(4 * 32 + sum(len(order) for order in orders) * 32 == 512, "calls per case")
    return {
        "status": "PASS_CPU_WHOLE_SHAPES",
        "native_pack_grid": [1280, 34, 1],
        "native_pack_threads": 128,
        "packed_payload_bytes": 25067520,
        "down_shape": [5120, 17408, 1280],
        "native_down_tiles": 400,
        "warmup_calls_per_case": 128,
        "measured_calls_per_case": 384,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, default=pathlib.Path("/src"))
    parser.add_argument("--output-root", type=pathlib.Path,
                        default=pathlib.Path("/build/q4-folded-ffn/whole/generated"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    root = args.source_root.resolve()
    files, metadata = prepare(root)
    if args.self_test:
        metadata["static_proof"] = static_proof()
    probe = pathlib.Path(__file__).resolve().parent
    metadata["owned_sha256"] = {name: digest((probe / name).read_bytes()) for name in OWNED}
    if not args.check_only:
        output = args.output_root.resolve()
        require(not output.is_relative_to(root), "generated output must be outside source")
        for name, data in files.items():
            path = output / name
            require(not path.exists() or path.read_bytes() == data,
                    "refusing to replace different generated bytes: " + name)
        for name, data in files.items():
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                with path.open("xb") as target:
                    target.write(data)
            require(path.read_bytes() == data, "generated file verification failed")
    metadata["check_only"] = args.check_only
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
