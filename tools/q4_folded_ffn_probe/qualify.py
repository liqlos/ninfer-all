import argparse
import hashlib
import importlib.util
import json
import pathlib
import re


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    assert generation["status"] == "PASS_BOUNDED_FOLDED16_TRANSFORM"
    assert generation["native_dot_header_byte_identical_except_provenance"]
    assert generation["schedule"]["native_loop_equal_after_explicit_pointer_and_bound_substitution"]
    text = (root / "sass-on.txt").read_text()
    resources = (root / "resources-on.txt").read_text()
    functions = module.functions(text)
    selected = {name: body for name, body in functions.items() if "q4_folded_ffn_kernelILb" in name}
    assert len(selected) == 2
    kernels = {}
    for name, body in selected.items():
        diagnostic = re.search(r"q4_folded_ffn_kernelILb([01])EE", name).group(1)
        items = module.instructions(body)
        resource = module.resources(resources, name)
        assert resource["REG"] <= 255 and resource["STACK"] == resource["LOCAL"] == 0
        assert not any(op.startswith(("LDL", "STL")) for _, op, _ in items)
        kernels[diagnostic] = {"symbol": name, "resources": resource,
                               "opcodes": module.histogram(items),
                               "sass_body_sha256": hashlib.sha256(body.encode()).hexdigest()}
    assert set(kernels) == {"0", "1"}
    ptx_path = root / "variant-on.o.ptx"
    ptx = ptx_path.read_text()
    symbol = kernels["0"]["symbol"]
    match = re.search(r"\.entry\s+" + re.escape(symbol) +
                      r"\s*\((.*?)\)\s*(?:\.[^\n{]+\n\s*)*\{(.*?)\n\}", ptx, re.S)
    assert match is not None
    params, body = match.groups()
    for index in (4, 5):
        name = symbol + "_param_" + str(index)
        assert name in params and name not in body
    assert (root / "variant-on.o").is_file() and (root / "variant-off.o").is_file()
    assert "q4_folded_ffn_kernel" not in (root / "symbols-disabled-variant.txt").read_text()
    for name, digest in generation["generated_headers"].items():
        assert sha(root / "generated/ggml-cuda" / name) == digest
    result = {"status": "PASS_STATIC_RESOURCE_AND_TIMED_STORE_GATE",
              "shared_bytes": 57856, "ctas": 2720, "kernels": kernels,
              "timed_diagnostic_pointer_params_unused_in_ptx": True,
              "off_has_no_folded_kernel": True,
              "numerical_gpu_validation": False, "performance_claim": False,
              "generation_sha256": sha(root / "generation.json"),
              "ptx_sha256": sha(ptx_path),
              "objects": {name: sha(root / name) for name in ("variant-on.o", "variant-off.o")},
              "analyzer_sha256": sha(pathlib.Path(__file__)), "parser_sha256": sha(helper)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
