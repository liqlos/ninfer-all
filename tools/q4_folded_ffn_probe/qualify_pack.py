import argparse
import hashlib
import importlib.util
import json
import pathlib
import re


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ptx_entry(text, symbol):
    match = re.search(r"\.entry\s+" + re.escape(symbol) +
                      r"\s*\((.*?)\)\s*(?:\.[^\n{]+\n\s*)*\{", text, re.S)
    assert match is not None
    depth = 1
    for end in range(match.end(), len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            return match.group(1), text[match.end():end]
    raise ValueError("Unterminated PTX entry: " + symbol)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = args.build_root.resolve()
    helper = pathlib.Path(__file__).resolve().parent.parent / "q6_full_warp_probe/sass_gate.py"
    spec = importlib.util.spec_from_file_location("sass_parser", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generation = json.loads((root / "generation.json").read_text())
    assert generation["status"] == "PASS_BOUNDED_FOLDED16_PACK_TRANSFORM"
    assert generation["kernel_proof"]["initialization_and_k_body_byte_identical"]
    assert generation["kernel_proof"]["additional_post_k_barriers"] == 1
    assert generation["kernel_proof"]["extra_dynamic_shared_bytes"] == 0
    assert generation["pack_proof"]["native_pack_arithmetic_and_stores_byte_identical"]
    for name, digest in generation["generated_files"].items():
        assert sha(root / "generated" / name) == digest
    functions = module.functions((root / "sass-on.txt").read_text())
    resources = (root / "resources-on.txt").read_text()
    ptx = (root / "variant-on.o.ptx").read_text()
    kernels = {}
    for name, body in functions.items():
        if "q4_folded_ffn_pack_kernelIL" not in name:
            continue
        identity = re.search(r"kernelIL.*?([01])ELb([01])EE", name)
        assert identity is not None, name
        layout, diagnostic = identity.groups()
        key = layout + "/" + diagnostic
        assert key not in kernels
        items = module.instructions(body)
        resource = module.resources(resources, name)
        assert resource["REG"] <= 255 and resource["STACK"] == resource["LOCAL"] == 0
        assert not any(op.startswith(("LDL", "STL")) for _, op, _ in items)
        params, ptx_body = ptx_entry(ptx, name)
        assert name + "_param_3" in ptx_body
        for index in (4, 5, 6):
            param = name + "_param_" + str(index)
            assert param in params
            assert (param in ptx_body) == (diagnostic == "1")
        stores = [op for _, op, _ in items if op.startswith("STG")]
        assert stores
        if diagnostic == "0":
            assert all("U16" not in op and "U8" not in op for op in stores)
        kernels[key] = {"symbol": name, "resources": resource,
                        "opcodes": module.histogram(items),
                        "sass_body_sha256": hashlib.sha256(body.encode()).hexdigest()}
    assert set(kernels) == {"0/0", "0/1", "1/0", "1/1"}
    assert "q4_folded_ffn_pack_kernel" not in (root / "symbols-disabled-variant.txt").read_text()
    result = {"status": "PASS_STATIC_PACK_RESOURCE_AND_TIMED_STORE_GATE",
              "shared_bytes": 57856, "ctas": 2720, "packed_output_bytes": 25067520,
              "kernels": kernels, "timed_diagnostic_pointer_params_unused_in_ptx": True,
              "off_has_no_folded_pack_kernel": True,
              "numerical_gpu_validation": False, "performance_claim": False,
              "generation_sha256": sha(root / "generation.json"),
              "ptx_sha256": sha(root / "variant-on.o.ptx"),
              "objects": {name: sha(root / name) for name in ("variant-on.o", "variant-off.o")},
              "analyzer_sha256": sha(pathlib.Path(__file__)), "parser_sha256": sha(helper)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
