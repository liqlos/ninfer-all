import argparse
import hashlib
import json
import pathlib
import shlex
import subprocess


SYMBOL = (
    "_ZN6ninfer3ops6detail22gguf_swiglu_linear_addERKNS_6TensorERKNS_6Weight"
    "EPS6_S7_RS2_RNS_11DeviceArenaEP11CUstream_st"
)
BUILD = pathlib.Path("/build")
SOURCE = pathlib.Path("/src")
HERE = pathlib.Path(__file__).resolve().parent
OUTPUT = BUILD / "q4-folded-ffn/runtime"
PRODUCER = BUILD / "q4-folded-ffn/pack/variant-on.o"
PRODUCER_SHA256 = "2e6f108cc813467f9bba6701b84ec24d7b1926ea4cc5800a3fb263fa86058d16"
FFN_OBJECT = BUILD / "src/models/CMakeFiles/ninfer_model_runtime.dir/qwen3_5/execution/ffn.cpp.o"
REAL_OBJECT = BUILD / "src/ops/CMakeFiles/ninfer_ops.dir/linear/gguf/gguf_linear.cpp.o"
SOURCE_PINS = {
    "src/core/tensor.h": "99bdbd15aab32e189165b98ede5efd94e2c900c20ce1313acb41b4173b0f188e",
    "src/core/weight.h": "42ef6bd886b36ea3cd2bc39f34939742a2368081cb7ac583f76fb21ad498fe1d",
    "src/core/arena.h": "869c06ffe4afe0bc88c8a9dcfdfd8f296ddd3202867a6d021aacc3ee3577e21e",
    "src/core/dtype.h": "13ab4e833debe24baf9294a9d8eefc3f61a182b914e8a128d1725be3d93574ed",
    "src/ops/linear/gguf/gguf_linear.h": "fc77e8acd05a061e6d0a76c47310959cfcec63e75e6364ada35f087e1afb3c6c",
    "src/ops/linear/gguf/ggml_bridge.h": "63dc7967143129a4e454824d0176cfbf9e6c8502b4b88423acdd291e74856b76",
    "src/ops/linear/gguf/ggml_bridge.cu": "670d0910869d8b1183b02d37f7ab2f554def75b9ff7ba7152562ce65327608e6",
    "src/ops/linear/gguf/ggml_bridge_mmq.cuh": "2513cb559034db1422841508c1c251e060ec087fe7b92643b2b539f687a7b15b",
}
PACK_PINS = {
    "pack_api.h": "7f71a33fc05ebe7dfe10e5276649336359cb540d24036a684aa55d1ced2f0218",
    "variant_pack.cu": "657900e9c369eddb7f218d810c7b652e69702ec2608194c15bb2a4bfde2b1914",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_command(arguments):
    return subprocess.check_output(arguments, cwd=BUILD, text=True)


def validate_pins():
    pins = {SOURCE / name: digest for name, digest in SOURCE_PINS.items()}
    pins.update({HERE.parent / "q4_folded_ffn_probe" / name: digest
                 for name, digest in PACK_PINS.items()})
    pins.update({
        PRODUCER: PRODUCER_SHA256,
        FFN_OBJECT: "328bfbb4fb10fb8ffe5167e44652a1c1f55933cf7dc9e5b6ed49fc3b0287ec00",
        REAL_OBJECT: "249d7b58062f93ab3c0963b0d7f178d168bebbb6505ff594d77cfc09d75fa4c9",
    })
    for path, expected in pins.items():
        require(path.is_file(), "Missing pinned input: " + str(path))
        require(sha(path) == expected, "Pinned input changed: " + str(path))
    return {str(path): digest for path, digest in pins.items()}


def validate_seam():
    undefined = read_command(["nm", "--undefined-only", str(FFN_OBJECT)])
    require(any(line.split() == ["U", SYMBOL] for line in undefined.splitlines()),
            "FFN object no longer has the external seam")
    relocations = read_command(["readelf", "-rW", str(FFN_OBJECT)])
    require(any("R_X86_64_PLT32" in line and SYMBOL in line
                for line in relocations.splitlines()), "FFN call relocation changed")
    defined = read_command(["nm", "--defined-only", str(REAL_OBJECT)])
    require(any(line.split()[1:] == ["T", SYMBOL] for line in defined.splitlines()
                if len(line.split()) == 3), "Original FFN symbol is missing")
    for path in (FFN_OBJECT, REAL_OBJECT):
        sections = read_command(["readelf", "-SW", str(path)])
        require(".gnu.lto" not in sections, "LTO object is unsupported: " + str(path))


def original_link():
    commands = read_command(["ninja", "-t", "commands", "apps/ninfer-serve"]).splitlines()
    require(bool(commands), "Ninja returned no link command")
    tokens = shlex.split(commands[-1])
    require(tokens[:2] == [":", "&&"] and tokens[-2:] == ["&&", ":"],
            "Unexpected Ninja command envelope")
    command = tokens[2:-2]
    require(command[0] == "/usr/bin/c++" and command.count("-o") == 1,
            "Unexpected server compiler or output options")
    require(command[command.index("-o") + 1] == "apps/ninfer-serve",
            "Unexpected original server output")
    require(not any("flto" in item or "dlto" in item or "--wrap" in item or
                    item in ("&&", ";", "|") for item in command),
            "Unsupported link options")
    inputs = [item for item in command if item.endswith((".o", ".a", ".so"))]
    require(len(inputs) == 29 and len(set(inputs)) == 29,
            "Expected exactly 29 original file inputs")
    require([item for item in inputs if item.endswith(".o")] ==
            ["apps/CMakeFiles/ninfer-serve.dir/serve/main.cpp.o"],
            "Original server object set changed")
    paths = [pathlib.Path(item) if item.startswith("/") else BUILD / item for item in inputs]
    require(all(path.is_file() for path in paths), "An original link input is missing")
    return command, paths


def metadata(paths):
    return {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in paths}


def compile_command(arm):
    return [
        "/usr/bin/c++", "-O3", "-DNDEBUG", "-std=gnu++20", "-DNINFER_SM8X_COMPAT=1",
        "-Wimplicit-fallthrough", "-Werror=implicit-fallthrough", "-ffile-prefix-map=/src=.",
        "-I/src/include", "-I/src/src", "-isystem",
        "/usr/local/cuda/targets/x86_64-linux/include",
        "-DNINFER_LAB_FOLDED16_RUNTIME=" + str(int(arm == "on")),
        "-c", str(HERE / "wrapper.cpp"), "-o", str(OUTPUT / ("wrapper-" + arm + ".o")),
    ]


def wrapped_link(original, arm):
    result = list(original)
    output_index = result.index("-o") + 1
    result[output_index] = str(OUTPUT / ("ninfer-serve-" + arm))
    insertion = result.index("apps/CMakeFiles/ninfer-serve.dir/serve/main.cpp.o") + 1
    extra = [str(OUTPUT / ("wrapper-" + arm + ".o")), str(PRODUCER), "-Wl,--wrap=" + SYMBOL]
    result[insertion:insertion] = extra
    restored = result[:insertion] + result[insertion + len(extra):]
    restored[output_index] = "apps/ninfer-serve"
    require(restored == original, "Link adaptation changed original inputs or flags")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Read-only preflight by default; --build compiles only lab wrappers and links lab servers.")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    require(BUILD.is_dir() and SOURCE.is_dir(), "Run inside the existing ninfer-builder")
    require(read_command(["/usr/bin/c++", "-dumpfullversion"]).strip() == "13.3.0",
            "Pinned host compiler changed")
    pins = validate_pins()
    validate_seam()
    original, inputs = original_link()
    before = metadata(inputs)
    source_sha = sha(HERE / "wrapper.cpp")
    commands = {arm: {"compile": compile_command(arm), "link": wrapped_link(original, arm)}
                for arm in ("off", "on")}
    result = {
        "status": "SOURCE_ONLY_PREFLIGHT_PASS",
        "lab_only": True,
        "product_integration": False,
        "gpu_run": False,
        "original_input_count": len(inputs),
        "original_link": original,
        "wrapper_sha256": source_sha,
        "pins": pins,
        "commands": commands,
        "diagnostic_env": "NINFER_LAB_FOLDED16_HOST_ADMISSIONS=1",
        "diagnostic_scope": "host admissions/capture only, not CUDA kernel counts or graph replays",
    }
    if args.build:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for arm in ("off", "on"):
            require(metadata(inputs) == before, "Original link inputs changed")
            require(sha(HERE / "wrapper.cpp") == source_sha, "Wrapper source changed")
            validate_pins()
            subprocess.run(commands[arm]["compile"], cwd=BUILD, check=True)
            subprocess.run(commands[arm]["link"], cwd=BUILD, check=True)
            require(metadata(inputs) == before, "Original link inputs changed during build")
            symbols = read_command(["nm", "--defined-only", str(OUTPUT / ("ninfer-serve-" + arm))])
            names = {line.split()[-1] for line in symbols.splitlines() if line.split()}
            require(SYMBOL in names and "__wrap_" + SYMBOL in names,
                    "Linked real/wrapper symbols are missing")
            require("__real_" + SYMBOL not in names, "Unexpected defined __real alias")
        require(sha(HERE / "wrapper.cpp") == source_sha, "Wrapper source changed during build")
        validate_pins()
        result["status"] = "LAB_WRAPPERS_LINKED_NOT_RUNTIME_QUALIFIED"
        result["outputs"] = {
            arm: {"path": str(OUTPUT / ("ninfer-serve-" + arm)),
                  "sha256": sha(OUTPUT / ("ninfer-serve-" + arm))}
            for arm in ("off", "on")
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
