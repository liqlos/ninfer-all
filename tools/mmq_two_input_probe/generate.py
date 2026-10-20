import argparse
import hashlib
import json
import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[2]
CUDA = "third_party/ggml-quants/src/ggml-cuda/"
MMQ = CUDA + "mmq.cuh"
PINS = {
    MMQ: "493ce59f8be4ff269fd0a2a8386f9d1e9a80bd2ee9c7478083d36bc2ac9ccf45",
    CUDA + "mmq-load-tiles.cuh":
        "80a7e02c1b2019db4ebdc1ef398a405c0249174dcf13a3d74b5d29af0347e3cf",
    CUDA + "mmq-vec-dot.cuh":
        "e47f3be4edcf96341b9ad48df24bd87480051c50bf7cabeef251abcb6ae39c66",
    CUDA + "mmq-config-ampere.cuh":
        "6de9336a6c2f65c0ad83cbe9303dcd03e8cbb6185d15346133a31421000f6408",
    CUDA + "mma.cuh":
        "08a18ccf4400990e9b2a651fdf702f49f0a521e427f151443e4359c912ac4fab",
    CUDA + "common.cuh":
        "6010078d639db19b1e8a2f78c4661ad240659b7bfa5a22864e315a2cbbc8da9b",
    CUDA + "vecdotq.cuh":
        "a7fd4281ad9f0a6d9373c776325411d2522e0b62ffae2c16e91e667a1af251fe",
    "third_party/ggml-quants/src/ggml-common.h":
        "0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c",
    "src/ops/linear/gguf/ggml_bridge.cu":
        "670d0910869d8b1183b02d37f7ab2f554def75b9ff7ba7152562ce65327608e6",
    "src/ops/linear/gguf/ggml_bridge_mmq.cuh":
        "2513cb559034db1422841508c1c251e060ec087fe7b92643b2b539f687a7b15b",
}
W_FLAGS = {
    "NINFER_GGUF_MMQ_STORE_FUSION": True,
    "NINFER_GGUF_Q4_COOPERATIVE_DECODE": True,
    "NINFER_GGUF_Q6_COOPERATIVE_DECODE": True,
}
PROFILES = {
    "Q4GateOrUp": {"N": 17408, "K": 5120, "T": 1280, "layout": 1,
                   "layout_name": "DS4", "weight_block_bytes": 144},
    "Q6Down": {"N": 5120, "K": 17408, "T": 1280, "layout": 0,
               "layout_name": "D4", "weight_block_bytes": 210},
}


def digest(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def require(text, literal):
    if literal not in text:
        raise ValueError("native source contract changed: " + literal)


def prepare(root):
    sources = {}
    for name, expected in PINS.items():
        raw = (root / name).read_bytes()
        if digest(raw) != expected:
            raise ValueError("native source pin changed: " + name)
        sources[name] = raw.decode().replace("\r\n", "\n")
    native = sources[MMQ]
    first = native.index("static __global__ void mul_mat_q(")
    stream = native.index("int kbc      =", first)
    body = native[stream:native.index("constexpr bool fixup = false;", stream)]
    for literal in (
        "nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z",
        "tmp2 = fast_div_modulo(tmp, nchannels_y);",
        "tmp2 = fast_div_modulo(tmp, nsamples_y);",
        "const int wt = tmp2.y;",
        "const int it = tmp2.x;",
        "wt*stride_sample_y   + zt*stride_channel_y",
        "wt*stride_sample_dst + zt*stride_channel_dst + jt*J*stride_col_dst",
        "offset_y += (col_low + jt * J) * (sizeof(block_q8_1_mmq) / sizeof(int));",
        "offset_dst += it*I;",
        "fastdiv(wt, sample_ratio)*stride_sample_x",
        "const int offset_x =",
        "it*I*stride_row_x",
        "tile_dst_bf16 = dst_bf16 + offset_dst;",
    ):
        require(body, literal)
    for literal in (
        "if constexpr (bf16_accumulate) { value += __bfloat162float(dst[at]); }",
        "dst[at] = __float2bfloat16(value);",
        "dst[ids_dst[j]*stride + i] = sum[(j0/tile_C::J + n)*tile_C::ne + l];",
        "const int * by0 = y + ncols_y * (kb0 * qk / ne_block) * sz;",
        "const int * by0 = y + ncols_y * ((kb0 * qk / ne_block) * sz + sz);",
        "#define MMQ_ITER_K             256",
    ):
        require(native, literal)
    config = sources[CUDA + "mmq-config-ampere.cuh"]
    for kind, layout in (("Q4_K", "Q8_1"), ("Q6_K", "Q6_K")):
        require(config, f"CASE(GGML_TYPE_{kind}, 256, 1, 128, 128, "
                f"GGML_CUDA_MMQ_SRAM_LAYOUT_{layout}, MMQ_ITER_K, true, false);")
    mma = sources[CUDA + "mma.cuh"]
    require(mma, "return ((l / 2) * 8) + (threadIdx.x / 4);")
    require(mma, "return ((threadIdx.x % 4) * 2) + (l % 2);")
    enabled = {name for name, value in W_FLAGS.items() if value}
    flags = set().union(*(set(re.findall(r"NINFER_GGUF_[A-Z0-9_]+", text))
                          for text in sources.values())) - enabled
    header = "#pragma once\n"
    header += "#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 860\n#error Closed_SM86_probe\n#endif\n"
    for flag in sorted(enabled):
        header += f"#if !defined({flag}) || {flag} != 1\n#error Required_W_switch_{flag}\n#endif\n"
    for flag in sorted(flags):
        header += f"#ifdef {flag}\n#error Unlisted_native_switch_{flag}\n#endif\n"
    header += '#include "ggml-cuda/mmq.cuh"\n'
    receipt = {
        "status": "SOURCE_PREPARED_ONLY",
        "native_pins": PINS,
        "generated_sha256": digest(header),
        "native_edits": [],
        "W_build_options": W_FLAGS,
        "disallowed_defined_options": sorted(flags),
        "profiles": PROFILES,
        "launch": {"control": "two separate nsamples_y=sample_ratio=1 launches",
                   "candidate": "one nsamples_y=sample_ratio=2 launch",
                   "weight_sample_stride_blocks": 0,
                   "activation_sample_stride_unit": "int32 words",
                   "output_sample_stride_unit": "destination elements",
                   "input_panels": "two independently native-packed T1280 panels",
                   "reshape_to_T2560": False, "full_K_per_CTA": True,
                   "fixup": False, "I": 128, "J": 128, "threads": 256,
                   "dynamic_shared_bytes": 57856},
        "epilogues": {"PlainFp32": "native FP32 sum store",
                      "StoreBf16": "native FP32 sum then BF16 round",
                      "AccumulateBf16": "native FP32 sum + FP32(existing BF16), then BF16 round"},
        "excluded": ["gate_product", "activation producer", "FFN integration", "host main"],
        "required_followups": [
            "compile both arms and compare all six native SASS/resource records",
            "GPU independent sample fixtures, exact outputs, canaries and carry qualification",
            "whole paired FFN against strongest folded16 before a performance claim",
        ],
        "compiled": False, "GPU_run": False, "performance_claim": False,
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
            parser.error("generated files belong in the existing external build area")
        out.mkdir(parents=True, exist_ok=True)
        (out / "mmq_two_input_native.cuh").write_bytes(header.encode())
        (out / "source_receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
