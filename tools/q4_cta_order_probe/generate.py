import argparse
import difflib
import hashlib
import json
import pathlib
import sys


SOURCE_DIRECTORY = pathlib.Path("third_party/ggml-quants/src/ggml-cuda")
EXPECTED_SHA256 = {
    "mmq.cuh": "493ce59f8be4ff269fd0a2a8386f9d1e9a80bd2ee9c7478083d36bc2ac9ccf45",
    "mmq-load-tiles.cuh": "80a7e02c1b2019db4ebdc1ef398a405c0249174dcf13a3d74b5d29af0347e3cf",
    "mmq-vec-dot.cuh": "e47f3be4edcf96341b9ad48df24bd87480051c50bf7cabeef251abcb6ae39c66",
}
ORIGINAL = """    int kbc      = int64_t(blockIdx.x)    *(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
    int kbc_stop = int64_t(blockIdx.x + 1)*(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
"""
REPLACEMENT = """#ifndef NINFER_GGUF_Q4_MMQ_CTA_GROUP_I4
#define NINFER_GGUF_Q4_MMQ_CTA_GROUP_I4 0
#endif
    uint32_t logicalpid = blockIdx.x;
#if NINFER_GGUF_Q4_MMQ_CTA_GROUP_I4 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 860)
    if constexpr (type == GGML_TYPE_Q4_K && J == 128 && !child_cta) {
        if (ids_dst == nullptr && expert_bounds == nullptr && tmp_fixup == nullptr &&
            nsamples_y.z == 1 && nchannels_y.z == 1 && gridDim.y == 1 && gridDim.z == 1 &&
            ntx.z > 0 && ntx.z <= UINT32_MAX / 4u &&
            uint64_t(gridDim.x) == uint64_t(ntx.z)*nty) {
            const uint32_t group = blockIdx.x / (4u*ntx.z);
            const uint32_t first = group*4u;
            const uint32_t width = min(4u, nty - first);
            const uint32_t local = blockIdx.x - group*4u*ntx.z;
            const uint32_t i = first + local % width;
            const uint32_t j = local / width;
            logicalpid = i*ntx.z + j;
        }
    }
#endif
    int kbc      = int64_t(logicalpid)    *(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
    int kbc_stop = int64_t(logicalpid + 1)*(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
"""


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def prepare(source_root):
    originals = {}
    for name, expected in EXPECTED_SHA256.items():
        data = (source_root / SOURCE_DIRECTORY / name).read_bytes()
        if sha256(data) != expected:
            raise ValueError("source SHA256 mismatch: " + name)
        originals[name] = data
    source = originals["mmq.cuh"]
    newline = "\r\n" if b"\r\n" in source else "\n"
    old = ORIGINAL.replace("\n", newline).encode("utf-8")
    new = REPLACEMENT.replace("\n", newline).encode("utf-8")
    if source.count(old) != 1:
        raise ValueError("expected exactly one initial kbc fragment")
    if b"NINFER_GGUF_Q4_MMQ_CTA_GROUP_I4" in source:
        raise ValueError("source already contains CTA grouping")
    start = source.index(old)
    end = start + len(old)
    generated = source[:start] + new + source[end:]
    if generated.count(new) != 1 or generated.replace(new, old, 1) != source:
        raise ValueError("generated source does not round-trip through the single fragment")
    files = dict(originals)
    files["mmq.cuh"] = generated
    line = source[:start].count(b"\n") + 1
    metadata = {
        "status": "PASS_SINGLE_FRAGMENT_TRANSFORM",
        "switch": "NINFER_GGUF_Q4_MMQ_CTA_GROUP_I4",
        "default": 0,
        "group_i": 4,
        "source_sha256": EXPECTED_SHA256["mmq.cuh"],
        "generated_header_sha256": sha256(generated),
        "source_headers": dict(EXPECTED_SHA256),
        "generated_headers": {name: sha256(data) for name, data in files.items()},
        "changed_fragments": 1,
        "original_lines": [line, line + old.count(b"\n") - 1],
        "generated_lines": [line, line + new.count(b"\n") - 1],
        "original_fragment_sha256": sha256(old),
        "generated_fragment_sha256": sha256(new),
        "unchanged_prefix_sha256": sha256(source[:start]),
        "unchanged_suffix_sha256": sha256(source[end:]),
        "siblings_byte_identical": all(
            files[name] == originals[name] for name in EXPECTED_SHA256 if name != "mmq.cuh"
        ),
    }
    return originals, files, metadata


def remap(pid, ni, nj):
    group = pid // (4 * nj)
    first = group * 4
    width = min(4, ni - first)
    local = pid - group * 4 * nj
    i = first + local % width
    j = local // width
    return i * nj + j


def self_test():
    grids = 0
    tiles = 0
    for ni in range(1, 138):
        for nj in range(1, 18):
            mapped = [remap(pid, ni, nj) for pid in range(ni * nj)]
            if len(set(mapped)) != ni * nj or min(mapped) != 0 or max(mapped) != ni * nj - 1:
                raise ValueError(f"mapping is not a bounded bijection: {ni}x{nj}")
            grids += 1
            tiles += ni * nj
    ni, nj, kb = 136, 10, 20
    for pid in range(ni * nj):
        logicalpid = remap(pid, ni, nj)
        kbc = logicalpid * (nj * ni * kb) // (ni * nj)
        kbc_stop = (logicalpid + 1) * (nj * ni * kb) // (ni * nj)
        if kbc != logicalpid * kb or kbc_stop - kbc != kb or kbc % kb:
            raise ValueError("full-K interval changed")
    return {"status": "PASS", "grids": grids, "tiles": tiles, "full_k_tiles": ni * nj}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, default=pathlib.Path("/src"))
    parser.add_argument("--output-root", type=pathlib.Path,
                        default=pathlib.Path("/build/q4-cta-order/generated"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--diff", action="store_true")
    args = parser.parse_args()
    originals, files, metadata = prepare(args.source_root.resolve())
    if args.self_test:
        metadata["mapping_check"] = self_test()
    if not args.check_only:
        destination = args.output_root.resolve() / "ggml-cuda"
        source_directory = (args.source_root.resolve() / SOURCE_DIRECTORY).resolve()
        if destination == source_directory or destination.is_relative_to(args.source_root.resolve()):
            raise ValueError("generated files must be outside the source tree")
        destination.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            path = destination / name
            if path.exists():
                if path.read_bytes() != data:
                    raise ValueError("refusing to overwrite a different generated header: " + name)
            else:
                with path.open("xb") as handle:
                    handle.write(data)
            if sha256(path.read_bytes()) != metadata["generated_headers"][name]:
                raise ValueError("generated header verification failed: " + name)
        metadata["output_root"] = str(destination.parent)
    metadata["check_only"] = args.check_only
    if args.diff:
        sys.stdout.writelines(difflib.unified_diff(
            originals["mmq.cuh"].decode("utf-8").splitlines(keepends=True),
            files["mmq.cuh"].decode("utf-8").splitlines(keepends=True),
            fromfile="original/ggml-cuda/mmq.cuh",
            tofile="generated/ggml-cuda/mmq.cuh",
        ))
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
