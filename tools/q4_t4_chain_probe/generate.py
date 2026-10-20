import argparse
import hashlib
import json
import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[2]
VECTOR = "src/ops/linear/gguf/ggml_bridge_vec.cuh"
W_FLAGS = {
    "NINFER_GGUF_DP4A_SHORT_CHAINS": False,
    "NINFER_GGUF_MMQ_STORE_FUSION": True,
    "NINFER_GGUF_Q4_COOPERATIVE_DECODE": True,
    "NINFER_GGUF_Q6_COOPERATIVE_DECODE": True,
    "NINFER_GGUF_Q6_MMQ_A_X4": False,
}
PINS = {
    VECTOR: "0a0c82d6cede1707f91044efa70aeb39a536eeecc53b479b5c976e92c82e6596",
    "src/ops/linear/gguf/ggml_bridge_internal.cuh":
        "8b78c42a2cb01b10e1227987cae22b3d0a030684070d6f35a213b033fb3c9d90",
    "src/ops/linear/gguf/ggml_bridge.h":
        "63dc7967143129a4e454824d0176cfbf9e6c8502b4b88423acdd291e74856b76",
    "src/ops/common/mma.cuh":
        "4f3f05cfb7ecce39a650cf53588587997a9e8bee4faf992e23564cf8c696d94d",
    "src/ops/common/memory.cuh":
        "68d4ce555d86761aa4c21d49726be2e48d82fcbdc4f6dbe4fc8b7356f945e205",
    "third_party/ggml-quants/src/ggml-cuda/common.cuh":
        "6010078d639db19b1e8a2f78c4661ad240659b7bfa5a22864e315a2cbbc8da9b",
    "third_party/ggml-quants/src/ggml-cuda/vecdotq.cuh":
        "a7fd4281ad9f0a6d9373c776325411d2522e0b62ffae2c16e91e667a1af251fe",
    "third_party/ggml-quants/src/ggml-cuda/mmq.cuh":
        "493ce59f8be4ff269fd0a2a8386f9d1e9a80bd2ee9c7478083d36bc2ac9ccf45",
}


def digest(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def prepare(root):
    sources = {}
    for name, expected in PINS.items():
        raw = (root / name).read_bytes()
        if digest(raw) != expected:
            raise ValueError("native source pin changed: " + name)
        sources[name] = raw.decode().replace("\r\n", "\n")
    native = sources[VECTOR]
    first = native.index("struct Decoder<GGML_TYPE_Q4_K> {")
    last = native.index("struct BaselineDecoder<GGML_TYPE_Q4_K> {", first)
    region = native[first:last]
    expected = ("return s.f[0] * d * float(dot4(s.w + 4, a + 4, "
                "dot4(s.w, a, 0))) - s.f[1] * sum;")
    if region.count(expected) != 1:
        raise ValueError("native Q4 dot expression changed")
    substitutions = [
        ("dot4(s.w + 4, a + 4, dot4(s.w, a, 0))", "integer_dot"),
        ("s.f[0]", "weight_scale"),
        ("s.f[1]", "weight_minimum"),
    ]
    transformed = expected
    for before, after in substitutions:
        if transformed.count(before) != 1:
            raise ValueError("nonunique dot substitution")
        transformed = transformed.replace(before, after)
    restored = transformed
    for before, after in reversed(substitutions):
        restored = restored.replace(after, before)
    if restored != expected:
        raise ValueError("unlisted FP32 expression change")
    flags = set()
    for text in sources.values():
        flags.update(re.findall(r"NINFER_GGUF_[A-Z0-9_]+", text))
    enabled = {name for name, value in W_FLAGS.items() if value}
    flags.difference_update(enabled)
    flags.update(name for name, value in W_FLAGS.items() if not value)
    guard = (
        "#pragma once\n#include \"probe_api.h\"\n"
        "#if NINFER_Q4_T4_CHAIN_PROBE != 0 && NINFER_Q4_T4_CHAIN_PROBE != 1\n"
        "#error Invalid_chain_switch\n#endif\n"
        "#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 860\n"
        "#error Closed_SM86_probe\n#endif\n")
    for flag in sorted(enabled):
        guard += f"#if !defined({flag}) || {flag} != 1\n#error Required_W_switch_{flag}\n#endif\n"
    for flag in sorted(flags):
        guard += f"#ifdef {flag}\n#error Unlisted_native_switch_{flag}\n#endif\n"
    header = (
        guard +
        "#include \"ops/linear/gguf/ggml_bridge_vec.cuh\"\n"
        "#include \"ops/common/mma.cuh\"\n"
        "namespace ninfer::ops::gguf::detail::q4tc {\n"
        "__device__ __forceinline__ float native_dot_integer(\n"
        "    float weight_scale, float weight_minimum, float d, float sum, int integer_dot) {\n"
        "    " + transformed + "\n}\n}\n")
    receipt = {
        "native_pins": PINS,
        "native_baseline": {
            "source": VECTOR,
            "edits": [],
            "instantiation": "kernel<GGML_TYPE_Q4_K,4,false,5120,true,false,2>",
            "launcher": "vec::launch<GGML_TYPE_Q4_K,4,false,5120>",
            "W_build_options": W_FLAGS,
            "other_vector_switches": False,
            "producer": "unchanged planar native Q8_1 qs and literal half2(d,sum)",
        },
        "float_expression": {
            "source_line": native[:native.index(expected)].count("\n") + 1,
            "native": expected,
            "candidate": transformed,
            "reversible_substitutions": substitutions,
            "FP32_reassociation": False,
        },
        "generated_sha256": digest(header),
        "scope": {
            "T": 4, "K": 5120, "candidate_rows": [6144, 17408, 34816],
            "native_rows": "positive; N1024 remains native",
            "fused_gate_up": False,
            "full_gemm_helperpack_qualified": False,
            "compiled": False, "gpu_run": False,
        },
    }
    return header, receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--output-dir", type=pathlib.Path)
    args = parser.parse_args()
    root = args.source_root.resolve()
    header, receipt = prepare(root)
    if args.output_dir:
        out = args.output_dir.resolve()
        if out == root or root in out.parents:
            parser.error("generated headers belong in the existing external build area")
        out.mkdir(parents=True, exist_ok=True)
        (out / "native_q4.cuh").write_bytes(header.encode())
        (out / "source_receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
