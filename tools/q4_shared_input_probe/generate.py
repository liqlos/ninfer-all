import argparse
import hashlib
import json
import pathlib
import re

from layout_proof import prove


ROOT = pathlib.Path(__file__).resolve().parents[2]
PROBE = pathlib.Path(__file__).resolve().parent
VECTOR = "src/ops/linear/gguf/ggml_bridge_vec.cuh"
PINS = {
    VECTOR: "0a0c82d6cede1707f91044efa70aeb39a536eeecc53b479b5c976e92c82e6596",
    "src/ops/linear/gguf/ggml_bridge_internal.cuh": "8b78c42a2cb01b10e1227987cae22b3d0a030684070d6f35a213b033fb3c9d90",
    "src/ops/linear/gguf/ggml_bridge.h": "63dc7967143129a4e454824d0176cfbf9e6c8502b4b88423acdd291e74856b76",
    "src/ops/common/memory.cuh": "68d4ce555d86761aa4c21d49726be2e48d82fcbdc4f6dbe4fc8b7356f945e205",
    "src/ops/common/mma.cuh": "4f3f05cfb7ecce39a650cf53588587997a9e8bee4faf992e23564cf8c696d94d",
    "third_party/ggml-quants/src/ggml-cuda/common.cuh": "6010078d639db19b1e8a2f78c4661ad240659b7bfa5a22864e315a2cbbc8da9b",
    "third_party/ggml-quants/src/ggml-cuda/vecdotq.cuh": "a7fd4281ad9f0a6d9373c776325411d2522e0b62ffae2c16e91e667a1af251fe",
    "third_party/ggml-quants/src/ggml-cuda/mmq.cuh": "493ce59f8be4ff269fd0a2a8386f9d1e9a80bd2ee9c7478083d36bc2ac9ccf45",
}
FILES = ("generate.py", "layout_proof.py", "probe_contract.cuh", "instantiate.cuh",
         "variant_native.cu", "variant_off.cu", "variant_on.cu", "build.sh", "sass_gate.py",
         "self_test.py")
W_FLAGS = ("NINFER_GGUF_Q4_COOPERATIVE_DECODE", "NINFER_GGUF_Q6_COOPERATIVE_DECODE",
           "NINFER_GGUF_MMQ_STORE_FUSION")
SWITCH = "NINFER_Q4_SHARED_INPUT_PROBE"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace_one(data, old, new):
    if old == new or data.count(old) != 1:
        raise ValueError("source transform anchor is not unique")
    at = data.index(old)
    result = data[:at] + new + data[at + len(old):]
    if result[:at] + old + result[at + len(new):] != data:
        raise ValueError("source transform did not reverse exactly")
    return result


def prepare(source_root):
    originals = {}
    for name, expected in PINS.items():
        raw = (source_root / name).read_bytes()
        if sha(raw) != expected:
            raise ValueError("pinned source changed: " + name)
        originals[name] = raw
    native = originals[VECTOR]
    newline = b"\r\n" if b"\r\n" in native else b"\n"
    enc = lambda text: text.replace("\n", newline.decode()).encode()
    first = enc("template <ggml_type type, int T, bool Fused, int KS, bool Prefetch2 = true,")
    last = enc("template <ggml_type type, int T, bool Fused, int KS>\nvoid launch_baseline(")
    if native.count(first) != 1 or native.count(last) != 1:
        raise ValueError("native kernel anchors changed")
    begin, end = native.index(first), native.index(last)
    original_kernel = native[begin:end]
    changes = []

    def change(label, old, new):
        nonlocal candidate_kernel
        before, after = enc(old), enc(new)
        candidate_kernel = replace_one(candidate_kernel, before, after)
        changes.append((label, before, after))

    candidate_kernel = original_kernel
    entry = "__launch_bounds__(kVecWarps * 32) __global__ void kernel(VecArgs p) {\n"
    change("closed_template_and_dynamic_shared", entry, entry +
           "    static_assert(type == GGML_TYPE_Q4_K && T == 4 && Fused && KS == 5120 &&\n"
           "                  Prefetch2 && !Baseline && Rows == 2 && kVecWarps == 4);\n"
           "    extern __shared__ __align__(16) std::uint8_t resident_q8[];\n")
    row = "    for (int row0 = (blockIdx.x * kVecWarps + warp) * step; row0 < p.rows;"
    preload = (
        "    auto* resident_codes = reinterpret_cast<int4*>(resident_q8);\n"
        "    auto* resident_metadata = reinterpret_cast<int4*>(resident_q8 + 20480);\n"
        "    const auto* source_codes = reinterpret_cast<const int4*>(p.qs);\n"
        "    const auto* source_metadata = reinterpret_cast<const int4*>(p.ds);\n"
        "    for (int i = threadIdx.x; i < 1280; i += kVecWarps * 32) {\n"
        "        resident_codes[i ^ ((i >> 3) & 1)] = source_codes[i];\n"
        "    }\n"
        "    for (int i = threadIdx.x; i < 160; i += kVecWarps * 32) {\n"
        "        resident_metadata[i] = source_metadata[i];\n"
        "    }\n"
        "    __syncthreads();\n"
        "    const auto* resident_ds = reinterpret_cast<const half2*>(resident_q8 + 20480);\n"
        "    const int resident_half = (lane >> 2) & 1;\n\n")
    change("one_preload_before_native_rows", row, preload + row)
    old_loads = (
        "                const float2 ds = __half22float2(p.ds[std::int64_t(j) * slices + s]);\n"
        "                d[j]            = ds.x;\n"
        "                sum[j]          = ds.y;\n"
        "                const auto* q   = reinterpret_cast<const int4*>(p.qs + std::int64_t(j) * k + 32 * s);\n"
        "                const int4 v0 = q[0], v1 = q[1];")
    new_loads = (
        "                const float2 ds = __half22float2(resident_ds[std::int64_t(j) * slices + s]);\n"
        "                d[j]            = ds.x;\n"
        "                sum[j]          = ds.y;\n"
        "                const auto* q   = reinterpret_cast<const int4*>(resident_q8 + std::int64_t(j) * k + 32 * s);\n"
        "                const int4 v0 = q[resident_half], v1 = q[resident_half ^ 1];")
    slice_start = candidate_kernel.index(enc("        const auto slice = [&](int it) {"))
    slice_end = candidate_kernel.index(enc("#if defined(NINFER_GGUF_VECTOR_PREFETCH2)"), slice_start)
    slice_body = candidate_kernel[slice_start:slice_end]
    updated_slice = replace_one(slice_body, enc(old_loads), enc(new_loads))
    candidate_kernel = candidate_kernel[:slice_start] + updated_slice + candidate_kernel[slice_end:]
    changes.append(("native_slice_loads_only", enc(old_loads), enc(new_loads)))
    reverse = candidate_kernel
    for _, before, after in reversed(changes):
        reverse = replace_one(reverse, after, before)
    if reverse != original_kernel:
        raise ValueError("unlisted kernel edit")
    candidate = native[:begin] + candidate_kernel + native[end:]
    includes = (
        ('#include "ggml_bridge_internal.cuh"', '#include "ops/linear/gguf/ggml_bridge_internal.cuh"'),
        ('#include "../../common/memory.cuh"', '#include "ops/common/memory.cuh"'),
        ('#include "../../common/mma.cuh"', '#include "ops/common/mma.cuh"'),
    )
    rebound_native, rebound_candidate = native, candidate
    for before, after in includes:
        rebound_native = replace_one(rebound_native, enc(before), enc(after))
        rebound_candidate = replace_one(rebound_candidate, enc(before), enc(after))
    restored_off = rebound_native
    for before, after in reversed(includes):
        restored_off = replace_one(restored_off, enc(after), enc(before))
    if restored_off != native:
        raise ValueError("OFF branch is not native byte-exact after include rebinding")
    flags = set()
    for raw in originals.values():
        flags.update(flag.decode() for flag in re.findall(rb"NINFER_GGUF_[A-Z0-9_]+", raw))
    guard = "#pragma once\n"
    for flag in W_FLAGS:
        guard += f"#if !defined({flag}) || {flag} != 1\n#error Required_W_switch_{flag}\n#endif\n"
    for flag in sorted(flags - set(W_FLAGS)):
        guard += f"#ifdef {flag}\n#error Unlisted_native_switch_{flag}\n#endif\n"
    header = (enc("#pragma once\n#ifndef " + SWITCH + "\n#error Include_probe_contract_first\n#endif\n"
                  "#if " + SWITCH + " == 0\n") + rebound_native + enc("\n#else\n") +
              rebound_candidate + enc("\n#endif\n"))
    outputs = {"q4_shared_input_generated.cuh": header,
               "q4_shared_input_flags_generated.cuh": guard.encode()}
    metadata = {
        "status": "PASS_PINNED_SOURCE_ONLY",
        "source_root": str(source_root),
        "probe_root": str(PROBE),
        "source_files": PINS,
        "probe_files": {name: sha((PROBE / name).read_bytes()) for name in FILES},
        "generated_files": {name: sha(data) for name, data in outputs.items()},
        "OFF_native_byte_identity_after_include_rebinding": True,
        "ON_reverse_transform_byte_exact": True,
        "native_kernel_sha256": sha(original_kernel),
        "candidate_kernel_sha256": sha(candidate_kernel),
        "changed_regions": [name for name, _, _ in changes],
        "protected": ["row ownership and grid stride", "static KS5120 unroll",
                      "weight addresses and cooperative decoder", "DP4A and FP32 expressions",
                      "butterfly reduction", "SwiGLU and BF16 store", "native launch templates"],
        "switch": SWITCH,
        "default": 0,
        "compile_entry": "kernel<GGML_TYPE_Q4_K,4,true,5120,true,false,2>",
        "runtime_route_or_host_harness_provided": False,
        "future_launch_contract": {"T": 4, "K": 5120, "rows": 17408,
                                   "weight_row_bytes": 2880, "grid": [328, 1, 1],
                                   "block": [128, 1, 1], "dynamic_shared_OFF": 0,
                                   "dynamic_shared_ON": 23040, "epilogue": "BF16 Store"},
        "layout_proof": prove(),
        "compile_run": False,
        "GPU_run": False,
        "performance_claim": False,
    }
    return outputs, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--output-root", type=pathlib.Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    outputs, metadata = prepare(source_root)
    if not args.check_only:
        if args.output_root is None:
            parser.error("--output-root is required unless --check-only is selected")
        destination = args.output_root.resolve()
        if destination == source_root or source_root in destination.parents:
            parser.error("generated compiler inputs must be outside the source tree")
        destination.mkdir(parents=True, exist_ok=True)
        for name, data in outputs.items():
            path = destination / name
            if path.exists() and path.read_bytes() != data:
                raise ValueError("refusing different existing generated input: " + str(path))
            if not path.exists():
                with path.open("xb") as handle:
                    handle.write(data)
        metadata["generated_root"] = str(destination)
        receipt = destination / "source_receipt.json"
        if receipt.exists():
            raise ValueError("use a fresh or deliberately reviewed compile directory")
        receipt.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
