import argparse
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys
import time


sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
SYMBOLS = {
    "READ": "_ZNK6ninfer8artifact9InputFile11read_directEmSt4spanISt4byteLm18446744073709551615EE",
    "DESTROY": "_ZN6ninfer8artifact9InputFileD1Ev",
    "DESTROY_BASE": "_ZN6ninfer8artifact9InputFileD2Ev",
    "MATERIALIZE": "_ZN6ninfer8artifact11materializeERKNS0_6ReaderEONS0_19MaterializationPlanERNS_13DeviceContextEPKNS_15StartupObserverESt10unique_ptrINS_19EvictableWeightPoolESt14default_deleteISC_EE",
}
SOURCE_PINS = {
    "src/artifact/file_io.h": "1192eec515bfb311e6676c6285c5b606a08cbda68f3def0a784050046db25477",
    "src/artifact/file_io.cpp": "1fd39c681a4424dc21effbccdd4de6e36b029faeaa5db90ad262e82edcd92f0d",
    "src/artifact/reader.h": "980d4281e9a65a0ce063de011fcd8f1a340f1f9cc16c795a2b43f60027ef0fe7",
    "src/artifact/reader.cpp": "afb6b1ab81d8ad88efac0bd4b2dc5c45aaf36e152af0d76e0e8893b5510a7eec",
    "src/artifact/materializer.h": "3065b2585bb79cfd2cf82c99d7f867565bc1c540d89e77d26edf5ec18d6a8d2a",
    "src/artifact/materializer.cpp": "e0e561da4d37d4fc4a43797d2c00e2c082e2e80f448fc49fd390029268ed2466",
    "src/models/qwen3_5/load.cpp": "e63f18d2f9b2b6e4ffb582ef35e431f46d6792e99f146aa530cb476d03717c5c",
}
ARTIFACT = "src/artifact/CMakeFiles/ninfer_artifact.dir/"
OBJECTS = {
    "input": ARTIFACT + "file_io.cpp.o",
    "reader": ARTIFACT + "reader.cpp.o",
    "materializer": ARTIFACT + "materializer.cpp.o",
    "load": "src/models/CMakeFiles/ninfer_model_loading.dir/qwen3_5/load.cpp.o",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_check(root):
    for relative, expected in SOURCE_PINS.items():
        require(sha(root / relative) == expected, "Native source changed: " + relative)
    wrapper = (HERE / "wrapper.cpp").read_text(encoding="utf-8")
    require("#define NINFER_LAB_WEIGHT_READ_AHEAD 0" in wrapper, "Default OFF contract changed")
    for key, symbol in SYMBOLS.items():
        require('#define PROBE_' + key + ' "' + symbol + '"' in wrapper, "ABI spelling changed: " + key)
    for name in ("wrapper.cpp", "read_ahead.h", "preflight.cpp", "native_io.cpp"):
        text = (HERE / name).read_text(encoding="utf-8")
        require("//" not in text and "/*" not in text, "New code comment: " + name)
        require(not any(token in text for token in ("cudaMemcpy", "cudaMalloc", "cudaFree", "cudaDeviceSynchronize", "cudaStreamSynchronize", ".detach(")),
                "GPU operation or detached thread: " + name)
    compile((HERE / "build.py").read_text(encoding="utf-8"), str(HERE / "build.py"), "exec")
    return {name: sha(HERE / name) for name in ("wrapper.cpp", "read_ahead.h", "preflight.cpp", "native_io.cpp", "build.py")}


def read(build, arguments):
    return subprocess.check_output(arguments, cwd=build, text=True)


def native_check(build):
    defined = {key: read(build, ["nm", "--defined-only", str(build / relative)])
               for key, relative in OBJECTS.items()}
    for name in ("READ", "DESTROY", "DESTROY_BASE"):
        require(any(line.split()[1:] == ["T", SYMBOLS[name]] for line in defined["input"].splitlines()),
                "Missing native InputFile definition: " + name)
    require(any(line.split()[1:] == ["T", SYMBOLS["MATERIALIZE"]]
                for line in defined["materializer"].splitlines()), "Missing native materialize definition")
    for caller, names in (("reader", ("READ", "DESTROY")), ("load", ("MATERIALIZE",))):
        undefined = read(build, ["nm", "--undefined-only", str(build / OBJECTS[caller])])
        relocations = read(build, ["readelf", "-rW", str(build / OBJECTS[caller])])
        for name in names:
            symbol = SYMBOLS[name]
            require(any(line.split() == ["U", symbol] for line in undefined.splitlines()),
                    "External call is absent: " + name)
            require(any("R_X86_64_PLT32" in line and symbol in line for line in relocations.splitlines()),
                    "External call relocation is absent: " + name)
    for relative in OBJECTS.values():
        require(".gnu.lto" not in read(build, ["readelf", "-SW", str(build / relative)]),
                "LTO seam is unsupported")
    dry = read(build, ["ninja", "-n", *OBJECTS.values()])
    require("no work to do" in dry and "[1/" not in dry, "Native loader dependencies are stale: " + dry)
    return {relative: sha(build / relative) for relative in OBJECTS.values()}


def original_link(build):
    tokens = shlex.split(read(build, ["ninja", "-t", "commands", "apps/ninfer-serve"]).splitlines()[-1])
    require(tokens[:2] == [":", "&&"] and tokens[-2:] == ["&&", ":"], "Unexpected Ninja envelope")
    command = tokens[2:-2]
    require(command[0] == "/usr/bin/c++" and command.count("-o") == 1, "Unexpected native linker")
    require(command[command.index("-o") + 1] == "apps/ninfer-serve", "Unexpected native output")
    require(not any("flto" in item or "dlto" in item or "--wrap" in item or item in ("&&", ";", "|")
                    for item in command), "Unsupported native link flags")
    inputs = [item for item in command if item.endswith((".o", ".a", ".so"))]
    require(len(inputs) == len(set(inputs)) == 29, "Expected exactly 29 original link inputs")
    require([item for item in inputs if item.endswith(".o")] ==
            ["apps/CMakeFiles/ninfer-serve.dir/serve/main.cpp.o"], "Native object set changed")
    paths = [pathlib.Path(item) if item.startswith("/") else build / item for item in inputs]
    require(all(path.is_file() for path in paths), "Native link input is missing")
    return command, paths


def commands(original, source, build, output, arm, model):
    common = ["/usr/bin/c++", "-O3", "-DNDEBUG", "-std=gnu++20", "-pthread", "-I" + str(HERE)]
    cpu = [*common, str(HERE / "preflight.cpp"), "-o", str(output / "preflight")]
    native_io = [*common, "-I" + str(source / "include"), "-I" + str(source / "src"),
                 str(HERE / "native_io.cpp"), str(build / OBJECTS["input"]),
                 "-o", str(output / "native-io")]
    wrapper = [*common, "-DNINFER_SM8X_COMPAT=1", "-DUTF8PROC_STATIC",
               "-I" + str(source / "include"), "-I" + str(source / "src"),
               "-isystem", "/usr/local/cuda/targets/x86_64-linux/include",
               "-DNINFER_LAB_WEIGHT_READ_AHEAD=" + str(int(arm == "on")),
               "-ffile-prefix-map=" + str(source) + "=.",
               "-c", str(HERE / "wrapper.cpp"), "-o", str(output / "wrapper.o")]
    link = list(original)
    link[link.index("-o") + 1] = str(output / "ninfer-serve")
    at = link.index("apps/CMakeFiles/ninfer-serve.dir/serve/main.cpp.o") + 1
    extra = [str(output / "wrapper.o"), "-pthread"]
    extra += ["-Wl,--wrap=" + symbol for symbol in SYMBOLS.values()]
    link[at:at] = extra
    restored = link[:at] + link[at + len(extra):]
    restored[restored.index("-o") + 1] = "apps/ninfer-serve"
    require(restored == original, "Native link changed beyond the host wrapper")
    return {"cpu_compile": cpu, "cpu_contract": [str(output / "preflight")],
            "native_io_compile": native_io, "native_io_contract": [str(output / "native-io"), str(model)],
            "wrapper_compile": wrapper, "link": link}


def main():
    parser = argparse.ArgumentParser(description="Bounded loader read-ahead; source-only unless --build is explicit.")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--native-io-model", type=pathlib.Path)
    parser.add_argument("--arm", choices=("off", "on"), default="off")
    parser.add_argument("--build-root", type=pathlib.Path, default=pathlib.Path("/build"))
    parser.add_argument("--source-root", type=pathlib.Path, default=pathlib.Path("/src"))
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    require(not (args.offline and args.build), "Offline and build are mutually exclusive")
    require(not args.cpu_only or args.build, "CPU-only requires an explicit build")
    require(not args.build or args.native_io_model is not None, "Build requires explicit native-I/O model")
    source = HERE.parents[1] if args.offline else args.source_root.resolve()
    sources = source_check(source)
    result = {"status": "SOURCE_ONLY_PREFLIGHT_PASS", "source_sha256": sources,
              "gpu_run": False, "fake_native_cpu_contract_run": False, "native_io_contract_run": False,
              "native_build_run": False,
              "abi_symbols": SYMBOLS, "max_extra_buffer_bytes": 4 * 64 * 2**20,
              "speed_claim": False, "arm": args.arm}
    if args.offline:
        print(json.dumps(result, indent=2))
        return
    build = args.build_root.resolve()
    require(read(build, ["/usr/bin/c++", "-dumpfullversion"]).strip() == "13.3.0",
            "Host compiler changed")
    result["native_object_sha256"] = native_check(build)
    original, inputs = original_link(build)
    owned_output = build / "weight-read-ahead"
    output = args.output.resolve() if args.output else owned_output / args.arm
    require(output.is_relative_to(owned_output) and output != owned_output and
            not output.is_relative_to(source) and not output.is_relative_to(HERE),
            "Outputs must stay inside the owned external build/weight-read-ahead subtree")
    recipe = commands(original, source, build, output, args.arm, args.native_io_model)
    result.update(status="NATIVE_READ_ONLY_PREFLIGHT_PASS", native_link_input_count=len(inputs),
                  original_link=original, commands=recipe)
    if args.build:
        pins = {str(path): sha(path) for path in inputs}
        for relative, expected in result["native_object_sha256"].items():
            path = build / relative
            require(sha(path) == expected, "Native loader object changed before build: " + relative)
            pins[str(path)] = expected
        output.mkdir(parents=True, exist_ok=True)
        operations = ["cpu_compile", "cpu_contract", "native_io_compile", "native_io_contract"]
        if not args.cpu_only:
            operations += ["wrapper_compile", "link"]
        result["phases"] = {}
        for operation in operations:
            require(source_check(source) == sources, "Probe source changed")
            require(all(sha(pathlib.Path(path)) == digest for path, digest in pins.items()),
                    "Native link input changed")
            started = time.perf_counter()
            process = subprocess.run(recipe[operation], cwd=build, text=True, capture_output=True)
            result["phases"][operation] = {"seconds": time.perf_counter() - started,
                                           "returncode": process.returncode,
                                           "stdout": process.stdout, "stderr": process.stderr}
            require(process.returncode == 0, operation + ": " + process.stderr + process.stdout)
            if operation == "cpu_contract":
                result["fake_native_cpu_contract_run"] = True
            if operation == "native_io_contract":
                qualification = json.loads(process.stdout)
                require(qualification["status"] == "PASS" and not qualification["GPU_run"] and
                        not qualification["performance_claim"], "Native byte-equality gate failed")
                result["native_io_contract_run"] = True
            if operation == "wrapper_compile":
                undefined = read(build, ["nm", "--undefined-only", str(output / "wrapper.o")])
                require(not any(name in undefined for name in
                    ("cudaMemcpy", "cudaMalloc", "cudaFree", "cudaDeviceSynchronize", "cudaStreamSynchronize", "cuMemcpy")),
                    "Host wrapper added a CUDA call")
        require(source_check(source) == sources, "Probe source changed during build")
        require(all(sha(pathlib.Path(path)) == digest for path, digest in pins.items()),
                "Native link input changed during build")
        if args.cpu_only:
            result.update(status="CPU_CONTRACTS_PASS_WRAPPER_NOT_BUILT", native_input_sha256=pins,
                          native_io_executable_sha256=sha(output / "native-io"),
                          fake_native_executable_sha256=sha(output / "preflight"))
            print(json.dumps(result, indent=2))
            return
        names = {line.split()[-1] for line in read(build,
                 ["nm", "--defined-only", str(output / "ninfer-serve")]).splitlines() if line.split()}
        require(all(symbol in names and "__wrap_" + symbol in names for symbol in SYMBOLS.values()),
                "Linked seam is missing")
        require(not any("__real_" + symbol in names for symbol in SYMBOLS.values()),
                "Unexpected real alias definition")
        require(source_check(source) == sources, "Probe source changed during build")
        require(all(sha(pathlib.Path(path)) == digest for path, digest in pins.items()),
                "Native link input changed during build")
        result.update(status="BUILT_NOT_LOADER_QUALIFIED", native_build_run=True,
                      native_input_sha256=pins, executable=str(output / "ninfer-serve"),
                      executable_sha256=sha(output / "ninfer-serve"),
                      wrapper_object_sha256=sha(output / "wrapper.o"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
