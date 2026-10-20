import argparse
import hashlib
import json
import pathlib

import proof


PINS = {
    "third_party/ggml-quants/src/ggml-cuda/mmq.cuh": "493ce59f8be4ff269fd0a2a8386f9d1e9a80bd2ee9c7478083d36bc2ac9ccf45",
    "third_party/ggml-quants/src/ggml-cuda/mmq-load-tiles.cuh": "80a7e02c1b2019db4ebdc1ef398a405c0249174dcf13a3d74b5d29af0347e3cf",
    "third_party/ggml-quants/src/ggml-cuda/mmq-vec-dot.cuh": "e47f3be4edcf96341b9ad48df24bd87480051c50bf7cabeef251abcb6ae39c66",
    "third_party/ggml-quants/src/ggml-cuda/mmq-config-ampere.cuh": "6de9336a6c2f65c0ad83cbe9303dcd03e8cbb6185d15346133a31421000f6408",
    "third_party/ggml-quants/src/ggml-cuda/mma.cuh": "08a18ccf4400990e9b2a651fdf702f49f0a521e427f151443e4359c912ac4fab",
    "third_party/ggml-quants/src/ggml-cuda/common.cuh": "6010078d639db19b1e8a2f78c4661ad240659b7bfa5a22864e315a2cbbc8da9b",
    "third_party/ggml-quants/src/ggml-common.h": "0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c",
    "src/ops/linear/gguf/ggml_bridge_internal.cuh": "8b78c42a2cb01b10e1227987cae22b3d0a030684070d6f35a213b033fb3c9d90",
    "src/ops/linear/gguf/ggml_bridge_mmq.cuh": "2513cb559034db1422841508c1c251e060ec087fe7b92643b2b539f687a7b15b",
    "src/ops/linear/gguf/ggml_bridge.h": "63dc7967143129a4e454824d0176cfbf9e6c8502b4b88423acdd291e74856b76",
}
PIN_HEAD = "460863299ef41e129059b4635c86ca0088fe3004"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def raw(text):
    return text.encode()


def function(data, name):
    prefix = b"template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void "
    begin = data.index(prefix + name.encode() + b"(")
    end = data.index(b"\n}\n", begin) + 3
    return data[begin:end]


def transform(data, changes):
    original = data
    encoded = [(raw(old), raw(new)) for old, new in changes]
    for old, new in encoded:
        proof.require(data.count(old) == 1, "bounded source anchor not unique: " + old.decode())
        data = data.replace(old, new, 1)
    inverse = data
    for old, new in reversed(encoded):
        proof.require(inverse.count(new) == 1, "inverse source anchor not unique")
        inverse = inverse.replace(new, old, 1)
    proof.require(inverse == original, "source inverse is not byte-exact")
    return data


def prepare(source):
    source = source.resolve()
    probe = pathlib.Path(__file__).resolve().parent
    loaded = {}
    for name, expected in PINS.items():
        loaded[name] = (source / name).read_bytes()
        proof.require(sha(loaded[name]) == expected, "native source pin changed: " + name)
    prefix = "third_party/ggml-quants/src/ggml-cuda/"
    native = loaded[prefix + "mmq.cuh"]
    loader0 = function(loaded[prefix + "mmq-load-tiles.cuh"], "ggml_cuda_mmq_load_tiles_q6_K")
    dot0 = function(loaded[prefix + "mmq-vec-dot.cuh"], "ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma")
    loader = transform(loader0, [
        ("void ggml_cuda_mmq_load_tiles_q6_K(", "void q6_j256_load_tiles("),
        ("constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;",
         "constexpr int nwarps      = 16;"),
        ("constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);", "constexpr int I           = 128;"),
        ("constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);",
         "constexpr int sram_stride = 76;\n    static_assert(type == GGML_TYPE_Q6_K && J == 256 && !fallback, \"closed loader\");"),
        ("#pragma unroll\n    for (int i0 = 0; i0 < I; i0 += nwarps*warp_size) {",
         "    if (threadIdx.y*warp_size + threadIdx.x < I) {\n#pragma unroll\n    for (int i0 = 0; i0 < I; i0 += nwarps*warp_size) {"),
        ("\n    constexpr int rows_per_warp = warp_size / 4;", "\n    }\n    constexpr int rows_per_warp = warp_size / 4;"),
    ])
    branch_start = dot0.index(raw("#elif defined(TURING_MMA_AVAILABLE)\n"))
    branch_end = dot0.index(raw("\n#else\n    GGML_UNUSED_VARS(x, y, sum, k00);"), branch_start)
    branch0 = dot0[branch_start:branch_end]
    branch = transform(branch0, [
        ("constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback);",
         "constexpr int sram_stride   = 76;\n    static_assert(type == GGML_TYPE_Q6_K && J == 256 && !fallback, \"closed dot\");"),
        ("constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);",
         "constexpr int rows_per_warp = 16;"),
        ("y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);",
         "y += (threadIdx.y % 2) * (tile_C::J*MMQ_TILE_Y_K);"),
        ("const int i0 = (threadIdx.y / ntx) * (ntx*tile_A::I);",
         "const int i0 = (threadIdx.y / 2) * tile_A::I;"),
        ("for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J)",
         "for (int j0 = 0; j0 < J; j0 += 2*tile_C::J)"),
        ("sum[(j0/tile_C::J + n)*tile_C::ne + l]", "sum[(j0/(2*tile_C::J) + n)*tile_C::ne + l]"),
    ])
    dot = transform(dot0, [(branch0.decode().replace("\r\n", "\n"), branch.decode().replace("\r\n", "\n")),
                          ("void ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma(", "void q6_j256_vec_dot(")])
    loop_start = native.index(raw("#endif\n        for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {")) + len(raw("#endif\n"))
    loop_end = native.index(raw("#if defined(NINFER_GGUF_Q4_MMQ_ASYNC_X)\n    }\n#endif"), loop_start)
    loop0 = native[loop_start:loop_end]
    loop = transform(loop0, [
        ("        if constexpr (child_cta) { ggml_cuda_mmq_load_tiles_q4_K<type, J, fallback, 64>(x, tile_x, offset_x + kb0 + child_row_offset*stride_row_x, tile_x_max_i-child_row_offset, stride_row_x); }\n        else { load_tiles(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x); }",
         "        q6_j256_load_tiles<GGML_TYPE_Q6_K, J, false>(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);"),
        ("        if constexpr (child_cta) { ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma<type, J, fallback, true>(tile_x, tile_y, sum, 0); }\n        else { vec_dot(tile_x, tile_y, sum, 0); }",
         "        q6_j256_vec_dot<GGML_TYPE_Q6_K, J, false>(tile_x, tile_y, sum, 0);"),
        ("        if constexpr (child_cta) { ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma<type, J, fallback, true>(tile_x, tile_y, sum, MMQ_TILE_NE_K); }\n        else { vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K); }",
         "        q6_j256_vec_dot<GGML_TYPE_Q6_K, J, false>(tile_x, tile_y, sum, MMQ_TILE_NE_K);"),
    ])
    proof.require(loop.count(b"__syncthreads();") == 4, "native K loop barrier count changed")
    proof.require(b"if (" not in loop and b"return" not in loop, "conditional barrier path")
    kernel0 = (probe / "kernel.cuh.in").read_bytes()
    proof.require(kernel0.count(b"@NATIVE_K_LOOP@") == 1, "kernel insertion anchor changed")
    kernel = kernel0.replace(b"@NATIVE_K_LOOP@", loop)
    guard = raw("\n#ifndef NINFER_Q6_J256_PROBE\n#define NINFER_Q6_J256_PROBE 0\n#endif\n#if NINFER_Q6_J256_PROBE\n")
    tail = guard + loader + dot + kernel + raw("\n#endif\n")
    overlay = native + tail
    proof.require(overlay[:-len(tail)] == native, "OFF extraction is not byte-exact")
    proof.require(b"tmp[n][l] += (C[0].x[l]*scA[n][l/2][k01/4 + 0] + C[1].x[l]*scA[n][l/2][k01/4 + 1])*dB[l%2];" in branch,
                  "native INT32 combine/FP32 update changed")
    files = {"ggml-cuda/mmq.cuh": overlay,
             "ggml-cuda/mmq-load-tiles.cuh": loaded[prefix + "mmq-load-tiles.cuh"],
             "ggml-cuda/mmq-vec-dot.cuh": loaded[prefix + "mmq-vec-dot.cuh"]}
    receipt = {"status": "PASS_BYTE_EXACT_OFF_AND_INVERSE", "source_root": str(source),
               "pin_head": PIN_HEAD, "source_sha256": PINS,
               "generated_sha256": {name: sha(data) for name, data in files.items()},
               "native_mmq_bytes": len(native), "appendix_bytes": len(tail),
               "native_regions_sha256": {"loader": sha(loader0), "dot": sha(dot0), "K_loop": sha(loop0)},
               "probe_sha256": {path.name: sha(path.read_bytes()) for path in sorted(probe.iterdir())
                                if path.is_file() and path.suffix in (".py", ".cu", ".cuh", ".h", ".in")},
               "unconditional_barriers_per_K256": 4,
               "proof": proof.run(), "default_enabled": False, "build_run": False,
               "off_compile_headers": "unmodified native paths; no generated include directory",
               "source_changes": ["16 row and 128 token warp ownership", "unique d writers",
                                  "512 threads and 76800 shared bytes", "closed full-K nofixup entry"]}
    return files, receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=pathlib.Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-dir", type=pathlib.Path)
    args = parser.parse_args()
    if args.check_only == bool(args.output_dir):
        parser.error("select exactly one of --check-only and --output-dir")
    files, receipt = prepare(args.source_root)
    if args.output_dir:
        source, out = args.source_root.resolve(), args.output_dir.resolve()
        if out == source or source in out.parents:
            parser.error("generated output must be outside source")
        for name, data in files.items():
            path = out / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (out / "source_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
