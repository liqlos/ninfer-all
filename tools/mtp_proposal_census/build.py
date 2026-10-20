import argparse
import hashlib
import json
import pathlib
import shlex
import subprocess


HERE = pathlib.Path(__file__).resolve().parent
BUILD = pathlib.Path("/build")
SOURCE = pathlib.Path("/src")
OUTPUT = BUILD / "mtp-proposal-census"
MODEL_OBJECTS = "src/models/CMakeFiles/ninfer_model_runtime.dir/qwen3_5/"
SYMBOLS = {
    "CAPTURE": "_ZN6ninfer6models7qwen3_59execution24capture_mtp_decode_batchERNS2_15MtpBatchContextEijjNS2_27MtpCausalAttentionEnvelopesERNS_21DecodeGraphDefinitionE",
    "RUN": "_ZN6ninfer6models7qwen3_59execution16mtp_decode_batchERNS2_15MtpBatchContextEijjNS2_27MtpCausalAttentionEnvelopesEPNS_21DecodeGraphExecutableE",
    "PROPOSE": "_ZN6ninfer6models7qwen3_59execution11TextContext17mtp_propose_batchERKNS_6TensorERS4_S7_",
    "ACCEPT": "_ZN6ninfer3ops32speculative_accept_greedy_draftsERKNS_6TensorES3_S3_S3_RS1_S4_S4_S4_S4_iPKNS0_14SamplingConfigERNS_11DeviceArenaEP11CUstream_st",
    "INSTALL": "_ZN6ninfer6models7qwen3_56detail11ProgramImpl16install_samplingERNS2_13SequenceStateERNS2_14RequestControlERKNS_3ops14SamplingConfigE",
    "SYNC": "_ZNK6ninfer13DeviceContext11synchronizeEv",
}
SEAMS = {
    "CAPTURE": (MODEL_OBJECTS + "program/graphs.cpp.o",
                MODEL_OBJECTS + "program/speculative/mtp.cpp.o"),
    "RUN": (MODEL_OBJECTS + "program/decode.cpp.o",
            MODEL_OBJECTS + "program/speculative/mtp.cpp.o"),
    "PROPOSE": (MODEL_OBJECTS + "program/speculative/mtp.cpp.o",
                MODEL_OBJECTS + "execution/text.cpp.o"),
    "ACCEPT": (MODEL_OBJECTS + "program/speculative/target_verification.cpp.o",
               "src/ops/CMakeFiles/ninfer_ops.dir/wrapper/speculative_round.cpp.o"),
    "INSTALL": (MODEL_OBJECTS + "program/prefill.cpp.o",
                MODEL_OBJECTS + "program/decode.cpp.o"),
    "SYNC": (MODEL_OBJECTS + "program/decode.cpp.o",
             "src/core/CMakeFiles/ninfer_core.dir/device.cu.o"),
}
SOURCE_PINS = {
    "src/core/tensor.h": "99bdbd15aab32e189165b98ede5efd94e2c900c20ce1313acb41b4173b0f188e",
    "src/core/arena.h": "869c06ffe4afe0bc88c8a9dcfdfd8f296ddd3202867a6d021aacc3ee3577e21e",
    "src/core/device.h": "f4b87ac4b493a58d29970ab233f69d0ddc5a2a6f5a8526893cd16f7315cabbd0",
    "src/core/dtype.h": "13ab4e833debe24baf9294a9d8eefc3f61a182b914e8a128d1725be3d93574ed",
    "include/ninfer/types.h": "b8f831d75158a530d8a9b27195ff91ba3dc1c7e406ff00a66b106b0c37f43975",
    "include/ninfer/ops/sampling.h": "5dab7ce91eb4c3c9a6331f53245cf6a4f4a2a2465475710fcceffbb2e814d0f3",
    "include/ninfer/ops/speculative_round.h": "5b6f1c70d32f48344a39cfe45def796f1735b42156b3fc7a4d46fe6bb6cad0e3",
    "src/models/load_options.h": "edf3c0877b98814aa515863c30f5d8bdc9027fa8f191b8a18d007200f3023c83",
    "src/models/qwen3_5/config.h": "7933f8b59cf519703a26f29c68d42f8d841f06dc62ef27b3da3ec540b96f2eb5",
    "src/models/qwen3_5/model.h": "2ba240fa43659e68a2cd3e763431d644183389eaa2886c436ad1afaed8e89aad",
    "src/models/qwen3_5/execution/parameters.h": "7d2b9b5b4afd682d8b02dad394075221081173433f73abc50dc8934883add2a8",
    "src/models/qwen3_5/execution/text.h": "afd2d7350038e76fb62e071de53fa012f9022b1ccf514c6a97303d0b9d4ffac6",
    "src/models/qwen3_5/execution/text.cpp": "69bd7ed90f3b10881e587bfcc421fc13fbbc5d2863e2fafe5814615f2fc1c802",
    "src/models/qwen3_5/program/context.h": "ae51889db643c18edeb7bd3e5883922c3295d29a72133f56e6ae7d9261d26330",
    "src/models/qwen3_5/program/round_buffers.h": "723cf1f7fa04ed44dfcd713497c15bc3e06dc8c2d8b9b21fb155807a2283ca4a",
    "src/models/qwen3_5/program/round_buffers.cpp": "c8eab467cbb68a35bf6af224d8a42261e7d51f0b70836bd61bb71631817daca0",
    "src/models/qwen3_5/program/program_impl.h": "988323eda2f3a635a5cc8d69db27e1e7d246f567dae695fdd211047c52a0b410",
    "src/models/qwen3_5/program/speculative/mtp.cpp": "a153ebc54fc3da6e8687d5b44ecf997199ea31a1adc623a03ee522d64a838107",
    "src/models/qwen3_5/program/speculative/target_verification.cpp": "e9f304bc7ad81f8460cab9c346d107561ea429a366d250ca15acbb0e313d2c82",
    "src/models/qwen3_5/program/decode.cpp": "65891b0ab84926dddfa1ffac6aa1c3a0b44bcccffe669ce5a4d10273508b0094",
    "src/models/qwen3_5/program/prefill.cpp": "a2416e37e23f345e96ed68654f719ec5d782d35663552beee14164a8e8bcd6f8",
    "src/models/qwen3_5/program/graphs.cpp": "e4eaab4c512d18df5f6326bc7f2c45a5f29b6147d7fd2eec3949f1540f215630",
    "src/models/qwen3_5/program/graph_execution.h": "6e6d4e3c57770e38c38e732a7d7f200caa457c1c6f132e1ec00ef078c8684e4e",
    "src/ops/kernel/sampling_device.cuh": "bbc1ccadb57d710dad75df7342fed0d7b28b9a4a94e8dfdeca3b30d850480f3a",
    "src/ops/common/score_id_order.cuh": "acd7fcb5e78ee4f6303bdf048ac0541019985724b2e0d9dc11ac20e4819a0585",
    "src/ops/kernel/speculative_round.cuh": "14ddcab84633ab87dabbe518d00b09d67dcb9580c5d0f77a806ce35d2ee9a7e8",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(arguments):
    return subprocess.check_output(arguments, cwd=BUILD, text=True)


def source_check(root):
    for relative, expected in SOURCE_PINS.items():
        require(sha(root / relative) == expected, "Source pin changed: " + relative)
    wrapper = (HERE / "wrapper.cpp").read_text()
    for name, symbol in SYMBOLS.items():
        require('#define CENSUS_' + name + ' "' + symbol + '"' in wrapper,
                "ABI declaration changed: " + name)
    return {name: sha(HERE / name)
            for name in ("observer.h", "observer.cu", "wrapper.cpp", "build.py", "check.py")}


def native_check():
    objects = sorted({name for pair in SEAMS.values() for name in pair})
    for name, (caller, definition) in SEAMS.items():
        symbol = SYMBOLS[name]
        undefined = read(["nm", "--undefined-only", str(BUILD / caller)])
        require(any(line.split() == ["U", symbol] for line in undefined.splitlines()),
                "Missing external call: " + name)
        relocations = read(["readelf", "-rW", str(BUILD / caller)])
        require(any("R_X86_64_PLT32" in line and symbol in line
                    for line in relocations.splitlines()), "Missing call relocation: " + name)
        defined = read(["nm", "--defined-only", str(BUILD / definition)])
        require(any(line.split()[1:] == ["T", symbol] for line in defined.splitlines()),
                "Missing native implementation: " + name)
    for relative in objects:
        require(".gnu.lto" not in read(["readelf", "-SW", str(BUILD / relative)]),
                "LTO object: " + relative)
    dry = read(["ninja", "-n", *objects])
    require("no work to do" in dry and "[1/" not in dry, "Native dependencies are stale: " + dry)
    return {relative: sha(BUILD / relative) for relative in objects}


def original_link():
    tokens = shlex.split(read(["ninja", "-t", "commands", "apps/ninfer-serve"]).splitlines()[-1])
    require(tokens[:2] == [":", "&&"] and tokens[-2:] == ["&&", ":"], "Unexpected Ninja envelope")
    command = tokens[2:-2]
    require(command[0] == "/usr/bin/c++" and command.count("-o") == 1, "Unexpected native linker")
    require(command[command.index("-o") + 1] == "apps/ninfer-serve", "Unexpected native output")
    require(not any("flto" in x or "dlto" in x or "--wrap" in x or x in ("&&", ";", "|")
                    for x in command), "Unsupported native link flags")
    inputs = [x for x in command if x.endswith((".o", ".a", ".so"))]
    require(len(inputs) == len(set(inputs)) == 29, "Expected 29 original native inputs")
    require([x for x in inputs if x.endswith(".o")] ==
            ["apps/CMakeFiles/ninfer-serve.dir/serve/main.cpp.o"], "Native object set changed")
    paths = [pathlib.Path(x) if x.startswith("/") else BUILD / x for x in inputs]
    require(all(x.is_file() for x in paths), "Missing native input")
    return command, paths


def commands(original):
    common = ["-O3", "-DNDEBUG", "-DNINFER_SM8X_COMPAT=1", "-I/src/include", "-I/src/src",
              "-I/src/third_party", "-I/src/third_party/llama-jinja", "-I" + str(HERE)]
    cuda = ["/usr/local/cuda/bin/nvcc", *common, "-std=c++20", "-arch=sm_86",
            "--expt-relaxed-constexpr", "-lineinfo", "-Xptxas=-v",
            "-c", str(HERE / "observer.cu"), "-o", str(OUTPUT / "observer.o")]
    arms = {}
    for arm in ("off", "on"):
        compile_command = ["/usr/bin/c++", *common, "-std=gnu++20", "-DUTF8PROC_STATIC",
            "-isystem", "/usr/local/cuda/targets/x86_64-linux/include",
            "-Wimplicit-fallthrough", "-Werror=implicit-fallthrough", "-ffile-prefix-map=/src=.",
            "-DNINFER_LAB_MTP_CENSUS=" + str(int(arm == "on")),
            "-c", str(HERE / "wrapper.cpp"), "-o", str(OUTPUT / ("wrapper-" + arm + ".o"))]
        link = list(original)
        output_index = link.index("-o") + 1
        link[output_index] = str(OUTPUT / ("ninfer-serve-" + arm))
        at = link.index("apps/CMakeFiles/ninfer-serve.dir/serve/main.cpp.o") + 1
        extra = [str(OUTPUT / ("wrapper-" + arm + ".o")), str(OUTPUT / "observer.o")]
        extra += ["-Wl,--wrap=" + symbol for symbol in SYMBOLS.values()]
        link[at:at] = extra
        restored = link[:at] + link[at + len(extra):]
        restored[restored.index("-o") + 1] = "apps/ninfer-serve"
        require(restored == original, "Link adaptation changed original flags or inputs")
        arms[arm] = {"compile": compile_command, "link": link}
    return {"cuda": cuda, "arms": arms}


def main():
    parser = argparse.ArgumentParser(description="Lab census source preflight; no build by default.")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    require(not (args.offline and args.build), "Offline and build are mutually exclusive")
    sources = source_check(HERE.parents[1] if args.offline else SOURCE)
    result = {"status": "OFFLINE_SOURCE_PINS_PASS", "diagnostic_only": True, "gpu_run": False,
              "source_sha256": sources, "abi_symbols": SYMBOLS}
    if args.offline:
        print(json.dumps(result, indent=2))
        return
    require(read(["/usr/bin/c++", "-dumpfullversion"]).strip() == "13.3.0", "Host compiler changed")
    result["native_object_sha256"] = native_check()
    original, inputs = original_link()
    recipe = commands(original)
    result.update(status="NATIVE_READ_ONLY_PREFLIGHT_PASS", original_link=original,
                  original_input_count=len(inputs), commands=recipe,
                  nvcc=read(["/usr/local/cuda/bin/nvcc", "--version"]))
    if args.build:
        native_pins = {str(path): sha(path) for path in inputs}
        OUTPUT.mkdir(parents=True, exist_ok=True)
        subprocess.run(recipe["cuda"], cwd=BUILD, check=True)
        observer_hash = sha(OUTPUT / "observer.o")
        for arm, steps in recipe["arms"].items():
            require(source_check(SOURCE) == sources, "Observer source changed")
            require(all(sha(pathlib.Path(path)) == digest for path, digest in native_pins.items()),
                    "Native link input changed")
            subprocess.run(steps["compile"], cwd=BUILD, check=True)
            subprocess.run(steps["link"], cwd=BUILD, check=True)
            names = {line.split()[-1] for line in read(
                ["nm", "--defined-only", str(OUTPUT / ("ninfer-serve-" + arm))]).splitlines()
                     if line.split()}
            require(all(symbol in names and "__wrap_" + symbol in names
                        for symbol in SYMBOLS.values()), "Linked ABI seam missing")
            require(not any("__real_" + symbol in names for symbol in SYMBOLS.values()),
                    "Unexpected defined real alias")
        require(source_check(SOURCE) == sources and sha(OUTPUT / "observer.o") == observer_hash,
                "Observer inputs changed")
        require(all(sha(pathlib.Path(path)) == digest for path, digest in native_pins.items()),
                "Native link input changed during build")
        result.update(status="BUILT_NOT_RUNTIME_QUALIFIED", native_input_sha256=native_pins,
                      observer_object_sha256=observer_hash,
                      outputs={arm: {"path": str(OUTPUT / ("ninfer-serve-" + arm)),
                                     "sha256": sha(OUTPUT / ("ninfer-serve-" + arm))}
                               for arm in ("off", "on")})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
