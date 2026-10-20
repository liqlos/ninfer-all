import argparse
import hashlib
import json
import pathlib
import re

from proof import prove


SOURCE = "src/ops/softmax_attention/dense/causal_cache/prompt_i8_fast.cuh"
HERE = pathlib.Path(__file__).resolve().parent
NATIVE = "attention_native_generated.cuh"
PAIRED = "attention_paired_generated.cuh"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(text, before, after):
    require(text.count(before) == 1, "source match count: " + before[:100])
    return text.replace(before, after, 1)


def generate(source_root):
    pins = json.loads((HERE / "source_pins.json").read_text(encoding="utf-8"))
    for relative, expected in pins.items():
        require(digest((source_root / relative).read_bytes()) == expected,
                "native source pin changed: " + relative)
    native_bytes = (source_root / SOURCE).read_bytes()
    native = native_bytes.decode("utf-8")
    lines = native.splitlines(keepends=True)
    extracted = {}

    def take(name, first, last):
        value = "".join(lines[first - 1:last])
        extracted[name] = {
            "first_line": first, "last_line": last,
            "native_sha256": digest(value.encode("utf-8")),
        }
        return value

    dispatch = take("dispatch_and_zero", 180, 203)
    prepare = take("q_prepare", 219, 240)
    prepare = replace_once(
        prepare,
        "            if (gid == (r & 7)) { q_scale_r[r >> 3][grp] = qs; }",
        "            if (gid == r && lid == grp) { prepare_scales[row * Groups + grp] = qs; }")
    qk = take("qk_group", 401, 434)
    require(qk.startswith("#pragma unroll\n"), "QK group pragma moved")
    qk = qk.replace("#pragma unroll\n", "#pragma unroll 1\n", 1)
    qk = replace_once(qk, "const int brow = nt * 8 + a_rin;",
                      "const int brow = (nt + side * 4) * 8 + a_rin;")
    qk = replace_once(qk, "const int keya  = nt * 8 + 2 * lid;",
                      "const int keya  = (nt + side * 4) * 8 + 2 * lid;")
    qk = replace_once(qk, "q_scale_r[0][grp]",
                      "__shfl_sync(FullMask, q_scale0, (lane & ~3) | grp)")
    qk = replace_once(qk, "q_scale_r[1][grp]",
                      "__shfl_sync(FullMask, q_scale1, (lane & ~3) | grp)")
    qk_scale_lines = (
        "                const float qs0 = __shfl_sync(FullMask, q_scale0, (lane & ~3) | grp);\n"
        "                const float qs1 = __shfl_sync(FullMask, q_scale1, (lane & ~3) | grp);\n")
    qk = replace_once(qk, qk_scale_lines, "")
    qk = replace_once(qk, "#pragma unroll\n            for (int nt = 0; nt < QKNt; ++nt)",
                      "".join(line[4:] for line in qk_scale_lines.splitlines(keepends=True)) +
                      "#pragma unroll\n            for (int nt = 0; nt < QKNt; ++nt)")
    maximum = take("qk_max", 436, 453)
    maximum = replace_once(maximum, "        float bm0            = -CUDART_INF_F;\n", "")
    maximum = replace_once(maximum, "        float bm1            = -CUDART_INF_F;\n", "")
    maximum = replace_once(maximum, "k0 + nt * 8 + 2 * lid",
                           "k0 + (nt + side * 4) * 8 + 2 * lid")
    online = take("online_max", 455, 466)
    probabilities = take("probabilities", 472, 482)
    probabilities = replace_once(
        probabilities,
        "            bl0 += p00 + p01;\n            bl1 += p10 + p11;\n",
        "            if (side == 0) {\n"
        "                bl0 += p00 + p01;\n"
        "                bl1 += p10 + p11;\n"
        "            } else {\n"
        "                high_pair_sum[nt][0] = p00 + p01;\n"
        "                high_pair_sum[nt][1] = p10 + p11;\n"
        "            }\n")
    probabilities = probabilities.replace("pa[nt >> 1]", "pa_local[nt >> 1]")
    denominator = take("denominator_update", 484, 485)
    final_denominator = take("final_denominator", 670, 673)
    shift = take("tile_shift", 490, 507)
    pv = take("pv_body", 580, 654)
    scale_prepare = take("native_pv_scale_prepare", 583, 597)
    pv = replace_once(pv, scale_prepare, "")
    one_scale = take("pv_scale_one_j", 586, 596)
    one_scale = one_scale.replace("key * Groups + grp", "key * Groups + side * 2 + grp")
    one_scale = one_scale.replace("(key + 1) * Groups + grp",
                                  "(key + 1) * Groups + side * 2 + grp")
    one_scale = one_scale.replace("(key + 8) * Groups + grp",
                                  "(key + 8) * Groups + side * 2 + grp")
    one_scale = one_scale.replace("(key + 9) * Groups + grp",
                                  "(key + 9) * Groups + side * 2 + grp")
    one_scale = one_scale.replace("vsc[j][", "vsc[")
    one_scale = "".join("    " + line for line in one_scale.splitlines(keepends=True))
    pv = replace_once(pv, "for (int grp = 0; grp < Groups; ++grp)",
                      "for (int grp = 0; grp < Groups / 2; ++grp)")
    pv = replace_once(pv, "for (int j = 0; j < PVKs; ++j) {\n",
                      "for (int j = 0; j < PVKs; ++j) {\n"
                      "                    unsigned vsc[2];\n" + one_scale)
    pv = replace_once(pv, "const int db    = db0 + 2 * q2;",
                      "const int db    = side * 8 + db0 + 2 * q2;")
    pv = pv.replace("vsc[j][", "vsc[")

    body = (HERE / "kernel.in.cuh").read_text(encoding="utf-8")
    replacements = {
        "DISPATCH_AND_ZERO": dispatch, "Q_PREPARE": prepare,
        "QK_GROUP": qk, "QK_MAX": maximum, "ONLINE_MAX": online,
        "PROBABILITIES": probabilities, "TILE_SHIFT": shift, "PV_BODY": pv,
        "DENOMINATOR_UPDATE": denominator, "FINAL_DENOMINATOR": final_denominator,
    }
    for marker, value in replacements.items():
        body = replace_once(body, "@" + marker + "@", value.rstrip("\n"))
    require(re.search(r"@[A-Z_]+@", body) is None, "unexpanded template marker")
    require("q_scale_r" not in body, "undistributed Q scale")
    require("unsigned vsc[PVKs]" not in body, "old V-scale live array")
    require("float score[QKNt][4]" in body and "constexpr int QKNt = 4;" in body,
            "QK score live state changed")
    require(body.count("__syncthreads();") == 7, "barrier source inventory changed")
    require(body.count("__syncwarp(FullMask);") == 1, "carry alias warp guard changed")
    require("float high_pair_sum[QKNt][2];" in body, "serial high-half carry missing")
    require("max_s[row0] = inv_l0;" in body and
            "const float inv_l0 = max_s[row0];" in body, "partner denominator missing")
    require(body.count("tile_shift = shift_s[pair];") == 1, "partner tile shift missing")
    require(body.count("__fmaf_rn(qs0 * ks0") == 1 and
            body.count("causal_prompt_i8_fast_mma_f16_acc") == 2,
            "source arithmetic extraction changed")
    paired = "".join(lines[:131]) + "\n" + body + "\n}\n"
    receipt = {
        "source_pins": pins,
        "probe_source_pins": {
            path.name: digest(path.read_bytes())
            for path in sorted(HERE.iterdir())
            if path.is_file() and path.suffix in (".py", ".cu", ".cuh", ".sh")
        },
        "native_source_sha256": digest(native_bytes),
        "native_source_pin_count": len(pins),
        "OFF_byte_exact": paired != native and native_bytes == native.encode("utf-8"),
        "native_generated_sha256": digest(native_bytes),
        "paired_generated_sha256": digest(paired.encode("utf-8")),
        "extractions": extracted,
        "source_static_checks": "PASS",
        "new_math_lowering_requires_actual_compile_review": True,
        "scope": "compile-only SM86, D256 Hq24 Hkv4 K8/V8, PagedKVBatchMetadata<false>",
    }
    return native_bytes, paired.encode("utf-8"), receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    native, paired, receipt = generate(args.source_root.resolve())
    if args.self_test:
        receipt["cpu_proofs"] = prove()
    if args.output_root is not None:
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / NATIVE).write_bytes(native)
        (args.output_root / PAIRED).write_bytes(paired)
    receipt["compile_run"] = False
    receipt["GPU_run"] = False
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
