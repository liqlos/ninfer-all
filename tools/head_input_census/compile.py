import argparse
import ast
import hashlib
import json
import pathlib
import subprocess


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD = pathlib.Path("/build")
OUTPUT = BUILD / "head-input-census"
PINS = {
    "src/ops/linear/gguf/ggml_bridge.h": "63dc7967143129a4e454824d0176cfbf9e6c8502b4b88423acdd291e74856b76",
    "src/ops/linear/gguf/ggml_bridge.cu": "670d0910869d8b1183b02d37f7ab2f554def75b9ff7ba7152562ce65327608e6",
    "src/ops/linear/gguf/ggml_bridge_internal.cuh": "8b78c42a2cb01b10e1227987cae22b3d0a030684070d6f35a213b033fb3c9d90",
    "src/ops/linear/gguf/ggml_bridge_vec.cuh": "0a0c82d6cede1707f91044efa70aeb39a536eeecc53b479b5c976e92c82e6596",
    "src/ops/linear/gguf/gguf_linear.cpp": "dbdc9738871b5b95ab74e4893e792c5a796658472d01931641097ca0c88d445f",
    "tools/mtp_proposal_census/observer.h": "d8753dfbd9fb34588d1713844db12edf06186988b478d5c8731587d01046dfc5",
    "tools/mtp_proposal_census/observer.cu": "d7a5626a29b4d2290730ace8696d2bbf706e25824a8a8c97e0a825575434c5bb",
    "tools/mtp_proposal_census/build.py": "4ee26bd2a592e66639721fb49e5c20ee275c10f984304b12e663710b0a2e5ac0",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def base_driver(root):
    for name, expected in PINS.items():
        path = (ROOT if name.startswith("tools/") else root) / name
        require(sha(path) == expected, "Source pin changed: " + name)
    path = ROOT / "tools/mtp_proposal_census/build.py"
    scope = {"__file__": str(path), "__name__": "_head_census_base"}
    exec(compile(path.read_text(), str(path), "exec"), scope)
    scope["source_check"](root)
    return scope


def source_check(root):
    base = base_driver(root)
    wrapper = (HERE / "wrapper.cpp").read_text()
    require("#define NINFER_LAB_HEAD_INPUT_CENSUS 0" in wrapper, "Observer must default OFF")
    for name, symbol in base["SYMBOLS"].items():
        require('#define CENSUS_' + name + ' "' + symbol + '"' in wrapper,
                "Changed existing seam: " + name)
    for name in ("compile.py", "proof.py"):
        ast.parse((HERE / name).read_text(), filename=name)
    return base, {name: sha(HERE / name) for name in
                  ("observer.h", "observer.cu", "wrapper.cpp", "compile.py", "proof.py", "PROOF.md")}


def read(arguments):
    return subprocess.check_output(arguments, cwd=BUILD, text=True)


def vector_seam():
    caller = BUILD / "src/ops/CMakeFiles/ninfer_ops.dir/linear/gguf/gguf_linear.cpp.o"
    definition = BUILD / "src/ops/CMakeFiles/ninfer_ggml_quants.dir/linear/gguf/ggml_bridge.cu.o"
    names = [line.split()[-1] for line in read(["nm", "--undefined-only", str(caller)]).splitlines()
             if line.split()]
    names = [name for name in names if name.startswith("_ZN6ninfer3ops4gguf14vector_productE")]
    require(len(names) == 1, "Expected one external public vector_product call")
    symbol = names[0]
    expected = ("ninfer::ops::gguf::vector_product(ninfer::ops::gguf::GgmlType, void const*, long, "
                "int, int, void const*, int, ninfer::ops::gguf::VectorOutput const&, CUstream_st*)")
    require(read(["c++filt", symbol]).strip() == expected, "Public vector_product ABI changed")
    require(any("R_X86_64_PLT32" in line and symbol in line
                for line in read(["readelf", "-rW", str(caller)]).splitlines()),
            "No native external call relocation for vector_product")
    require(any(line.split()[1:] == ["T", symbol]
                for line in read(["nm", "--defined-only", str(definition)]).splitlines()),
            "No native vector_product definition")
    for path in (caller, definition):
        require(".gnu.lto" not in read(["readelf", "-SW", str(path)]), "LTO disables wrap proof")
    dry = read(["ninja", "-n", str(caller.relative_to(BUILD)), str(definition.relative_to(BUILD))])
    require("no work to do" in dry and "[1/" not in dry, "Native vector dependencies stale")
    return symbol, {str(path): sha(path) for path in (caller, definition)}


def recipe(base, symbol, original):
    common = ["-O3", "-DNDEBUG", "-DNINFER_SM8X_COMPAT=1", "-I/src/include", "-I/src/src",
              "-I/src/third_party", "-I/src/third_party/llama-jinja", "-I" + str(HERE)]
    cuda = ["/usr/local/cuda/bin/nvcc", *common, "-std=c++20", "-arch=sm_86",
            "--expt-relaxed-constexpr", "-lineinfo", "-Xptxas=-v"]
    compiles = [cuda + ["-c", str(path), "-o", str(OUTPUT / name)] for path, name in (
        (HERE / "observer.cu", "head-observer.o"),
        (HERE.parent / "mtp_proposal_census/observer.cu", "join-observer.o"))]
    links = {}
    for arm in ("off", "on"):
        obj = OUTPUT / ("wrapper-" + arm + ".o")
        compiles.append(["/usr/bin/c++", *common, "-std=gnu++20", "-DUTF8PROC_STATIC",
            "-isystem", "/usr/local/cuda/targets/x86_64-linux/include",
            "-Wimplicit-fallthrough", "-Werror=implicit-fallthrough", "-ffile-prefix-map=/src=.",
            "-DNINFER_LAB_HEAD_INPUT_CENSUS=" + str(int(arm == "on")),
            '-DHEAD_VECTOR_SYMBOL="' + symbol + '"',
            "-c", str(HERE / "wrapper.cpp"), "-o", str(obj)])
        link = list(original)
        link[link.index("-o") + 1] = str(OUTPUT / ("ninfer-serve-" + arm))
        at = link.index("apps/CMakeFiles/ninfer-serve.dir/serve/main.cpp.o") + 1
        extra = [str(obj), str(OUTPUT / "head-observer.o"), str(OUTPUT / "join-observer.o")]
        extra += ["-Wl,--wrap=" + name for name in [*base["SYMBOLS"].values(), symbol]]
        link[at:at] = extra
        restored = link[:at] + link[at + len(extra):]
        restored[restored.index("-o") + 1] = "apps/ninfer-serve"
        require(restored == original, "Link recipe changed original input or flags")
        links[arm] = link
    return {"compile_only": compiles, "unexecuted_link_recipes": links}


def main():
    parser = argparse.ArgumentParser(description="Offline by default; no link, model run, or GPU call.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--native-preflight", action="store_true")
    mode.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    native = args.native_preflight or args.compile_only
    root = pathlib.Path("/src") if native else ROOT
    base, sources = source_check(root)
    result = {"status": "OFFLINE_SOURCE_ONLY_PASS", "source_sha256": sources,
              "build_run": False, "link_run": False, "gpu_run": False,
              "runtime_qualified": False, "native_numerical_certificate_qualified": False}
    if native:
        require(read(["/usr/bin/c++", "-dumpfullversion"]).strip() == "13.3.0", "Compiler changed")
        result["base_object_sha256"] = base["native_check"]()
        symbol, objects = vector_seam()
        original, inputs = base["original_link"]()
        commands = recipe(base, symbol, original)
        result.update(status="NATIVE_ABI_PREFLIGHT_PASS", vector_symbol=symbol,
                      vector_object_sha256=objects, commands=commands)
        if args.compile_only:
            frozen = {str(path): sha(path) for path in inputs}
            for relative, expected in result["base_object_sha256"].items():
                path = BUILD / relative
                require(sha(path) == expected, "Native census object changed: " + relative)
                frozen[str(path)] = expected
            for path, expected in objects.items():
                require(sha(pathlib.Path(path)) == expected, "Native vector object changed: " + path)
                frozen[path] = expected
            OUTPUT.mkdir(parents=True, exist_ok=True)
            for command in commands["compile_only"]:
                subprocess.run(command, cwd=BUILD, check=True)
            require(source_check(root)[1] == sources, "Observer sources changed during compile")
            require(all(sha(pathlib.Path(path)) == digest for path, digest in frozen.items()),
                    "Native input changed during compile")
            result.update(status="OBJECTS_COMPILED_NOT_LINKED", build_run=True,
                object_sha256={path.name: sha(path) for path in OUTPUT.glob("*.o")})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
