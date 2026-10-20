import argparse
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time
import traceback
import uuid


sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


controller = module("head_link_controller", HERE / "run.py")
require = controller.require
lab = controller.lab


def compiled(path, digest):
    receipt = controller.checked(path, digest)
    require(receipt["status"] == "OBJECTS_COMPILED_NOT_LINKED" and receipt["returncode"] == 0 and
            receipt["GPU_run"] is False and receipt["link_run"] is False and
            receipt["production_unchanged"] is True, "Compile receipt is not qualified")
    native = receipt["native"]
    require(native["status"] == "OBJECTS_COMPILED_NOT_LINKED" and native["build_run"] is True and
            native["link_run"] is False and native["gpu_run"] is False and
            native["runtime_qualified"] is False and
            native["native_numerical_certificate_qualified"] is False and
            set(native["object_sha256"]) == {"join-observer.o", "head-observer.o", "wrapper-off.o", "wrapper-on.o"},
            "Compiled object contract differs")
    return receipt


def native_link(args):
    require(os.name == "posix" and HERE == pathlib.Path("/lab/source/tools/head_input_census"),
            "Native link requires the existing builder source bind")
    receipt = compiled(args.compile_receipt, args.compile_sha256)
    native = receipt["native"]
    recipe = module("head_native_recipe", HERE / "compile.py")
    base, sources = recipe.source_check(pathlib.Path("/src"))
    require(sources == native["source_sha256"], "Observer source differs from compile receipt")
    require(recipe.read(["/usr/bin/c++", "-dumpfullversion"]).strip() == "13.3.0", "Compiler changed")
    require(base["native_check"]() == native["base_object_sha256"], "Native seam objects changed")
    symbol, vector_objects = recipe.vector_seam()
    require(symbol == native["vector_symbol"] and vector_objects == native["vector_object_sha256"],
            "Native vector seam changed")
    original, inputs = base["original_link"]()
    commands = recipe.recipe(base, symbol, original)
    require(commands == native["commands"], "Current link recipe differs from compiled receipt")
    baseline = recipe.BUILD / "apps/ninfer-serve"
    require(lab.sha(baseline) == controller.BASELINE_SHA, "Native baseline is not W")
    dry = recipe.read(["ninja", "-n", "apps/ninfer-serve"])
    require("no work to do" in dry and "[1/" not in dry, "Original server dependencies are stale")
    objects = {str(recipe.OUTPUT / name): digest for name, digest in native["object_sha256"].items()}
    require(all(lab.sha(pathlib.Path(path)) == digest for path, digest in objects.items()),
            "Compiled observer object changed")
    frozen_inputs = {str(path): lab.sha(path) for path in inputs}
    frozen = {**frozen_inputs, **objects, **vector_objects,
              **{str(recipe.BUILD / path): digest for path, digest in native["base_object_sha256"].items()},
              str(baseline): controller.BASELINE_SHA}
    require(len(frozen_inputs) == 29, "Original native input count differs")
    output = recipe.OUTPUT / "ninfer-serve-on"
    require(not output.exists() and not (recipe.OUTPUT / "ninfer-serve-off").exists(),
            "An existing head executable occupies the experimental slot")
    command = commands["unexecuted_link_recipes"]["on"]
    require(command[command.index("-o") + 1] == str(output), "Unexpected ON output")
    started = time.monotonic()
    subprocess.run(command, cwd=recipe.BUILD, check=True)
    seconds = time.monotonic() - started
    require(recipe.source_check(pathlib.Path("/src"))[1] == sources, "Source changed during link")
    require(all(lab.sha(pathlib.Path(path)) == digest for path, digest in frozen.items()),
            "Native or observer input changed during link")
    names = {line.split()[-1] for line in recipe.read(["nm", "--defined-only", str(output)]).splitlines()
             if line.split()}
    symbols = [*base["SYMBOLS"].values(), symbol]
    require(all(name in names and "__wrap_" + name in names for name in symbols) and
            not any("__real_" + name in names for name in symbols), "Linked wrap seam missing or unresolved")
    with output.open("rb") as stream:
        require(stream.read(4) == b"\x7fELF", "Native ON output is not ELF")
    result = {"status": "ON_LINKED_NOT_RUNTIME_QUALIFIED", "compile_receipt_sha256": args.compile_sha256,
              "source_sha256": sources, "baseline_sha256": controller.BASELINE_SHA,
              "native_input_count": len(inputs), "native_input_sha256": frozen_inputs,
              "native_inputs_verified_unchanged_during_this_link": True,
              "historical_all_29_input_hashes_in_compile_receipt": False,
              "object_sha256": native["object_sha256"], "vector_symbol": symbol,
              "base_object_sha256": native["base_object_sha256"], "vector_object_sha256": vector_objects,
              "command": command, "link_seconds": seconds, "output_path": str(output),
              "executable_sha256": lab.sha(output), "gpu_run": False, "runtime_qualified": False,
              "native_numerical_certificate_qualified": False}
    print(json.dumps(result, indent=2))


def export(builder, native, token):
    path = controller.EXPORT
    require(path.parent.is_dir() and not path.exists(), "Experimental export slot already exists")
    partial = path.with_name(path.name + "." + token + ".partial")
    require(not partial.exists(), "Export temporary path exists")
    try:
        with partial.open("xb") as stream:
            result = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04", "--exec", "/usr/bin/docker", "exec",
                builder, "cat", native["output_path"]], stdout=stream, stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW, timeout=120)
        require(result.returncode == 0, "Export failed: " + result.stderr.decode(errors="replace")[-2000:])
        require(lab.sha(partial) == native["executable_sha256"], "Exported bytes differ")
        with partial.open("rb") as stream:
            require(stream.read(4) == b"\x7fELF", "Exported ON output is not ELF")
        require(not path.exists(), "Export slot changed while copying")
        partial.rename(path)
    finally:
        if partial.exists():
            partial.unlink()
    return controller.prior.file_identity(path)


def host_link(args):
    require(os.name == "nt", "Run the link controller in native Windows Python")
    require(args.tag is not None, "Link requires an exclusive tag")
    token = uuid.uuid4().hex
    lab.OWNER = "ninfer-head-link-" + token
    record = {"schema": "head-input-link-r1", "status": "INCOMPLETE", "started_at": lab.now(),
              "tag": args.tag, "controller_pid": os.getpid(), "gpu_run": False,
              "runtime_qualified": False, "compile_receipt_sha256": args.compile_sha256}
    probe, probe_name, claimed, config, production = None, None, False, None, None
    with controller.mutexes():
        try:
            controller.claim(args.tag)
            claimed = True
            receipt = compiled(args.compile_receipt, args.compile_sha256)
            controller.source_pins(receipt["native"])
            record["helper_sha256"] = controller.helpers()
            config = lab.sha(controller.ROOT / "broker/config.json")
            production = controller.common.production(lab)
            record["admission"] = controller.common.admission(lab, controller.guard)
            builder = record["admission"]["builder"]
            require(builder["id"] == receipt["admission"]["builder"]["id"] and
                    builder["image"] == receipt["admission"]["builder"]["image"] and
                    builder["id"] == receipt["builder_after"]["id"], "Existing builder identity changed")
            require(lab.sha(controller.BASELINE) == controller.BASELINE_SHA, "Existing exported W changed")
            require(not controller.EXPORT.exists(), "Experimental export slot already exists")
            relative = args.compile_receipt.resolve().relative_to(controller.ROOT.resolve()).as_posix()
            command = ["exec", builder["id"], "python3", "-B", "/lab/source/tools/head_input_census/link.py",
                       "--native", "--compile-receipt", "/lab/" + relative,
                       "--compile-sha256", args.compile_sha256]
            started = time.monotonic()
            result = lab.docker(*command, timeout=240, check=False)
            record.update(seconds=time.monotonic() - started, returncode=result.returncode,
                          stdout=result.stdout, stderr=result.stderr)
            require(result.returncode == 0, "Native link failed")
            native = json.loads(result.stdout)
            require(native["status"] == "ON_LINKED_NOT_RUNTIME_QUALIFIED" and
                    native["source_sha256"] == receipt["native"]["source_sha256"] and
                    native["object_sha256"] == receipt["native"]["object_sha256"] and
                    native["compile_receipt_sha256"] == args.compile_sha256 and
                    native["command"] == receipt["native"]["commands"]["unexecuted_link_recipes"]["on"],
                    "Native link receipt differs")
            record["native"] = native
            record["export_admission"] = controller.common.admission(lab, controller.guard)
            require(record["export_admission"]["builder"]["id"] == builder["id"], "Builder changed before export")
            record["exported_identity"] = export(builder["id"], native, token)
            bind = lab.WSL_ROOT + "/" + controller.EXPORT.relative_to(controller.ROOT).as_posix()
            probe_name = "ninfer-head-bind-" + token[:8]
            probe = lab.docker("create", "--name", probe_name,
                "--label", controller.prior.LABEL + "=" + token, "--network", "none", "--ulimit", "core=0",
                "-e", "NVIDIA_VISIBLE_DEVICES=void", "--mount",
                "type=bind,source=" + bind + ",target=" + controller.EXE + ",readonly",
                "--entrypoint", "sha256sum", lab.IMAGE, controller.EXE).stdout.strip()
            item = controller.prior.owned(probe, token)
            require(not item["HostConfig"].get("DeviceRequests") and not item["HostConfig"].get("Devices") and
                    any(m["Type"] == "bind" and not m["RW"] and m["Destination"] == controller.EXE
                        for m in item["Mounts"]), "Fresh export bind is not CPU-only/read-only")
            result = lab.docker("start", "--attach", probe)
            item = controller.prior.owned(probe, token)
            require(not item["State"]["Running"] and item["State"]["ExitCode"] == 0 and
                    result.stdout.split() == [native["executable_sha256"], controller.EXE], "Fresh bind hash differs")
            record["fresh_bind"] = {"container_id": item["Id"], "image": item["Image"], "gpu_requested": False,
                                    "sha256": native["executable_sha256"], "readonly": True}
            controller.prior.remove_owned(probe, token)
            probe = None
            probe_name = None
            require(controller.helpers() == record["helper_sha256"], "Controller changed during link/export")
            controller.source_pins(native)
            require(lab.sha(controller.BASELINE) == controller.BASELINE_SHA and
                    lab.sha(controller.EXPORT) == native["executable_sha256"], "Exported executable changed")
            record["status"] = "LINKED_EXPORTED_NOT_RUNTIME_QUALIFIED"
        except BaseException as exc:
            record.update(failure=type(exc).__name__ + ": " + str(exc), failure_traceback=traceback.format_exc())
        finally:
            if probe_name and not probe:
                try:
                    probe = controller.find_owned(probe_name, token)
                except BaseException as exc:
                    record.setdefault("cleanup_failures", []).append(str(exc))
            if probe:
                try:
                    controller.prior.remove_owned(probe, token)
                except BaseException as exc:
                    record.setdefault("cleanup_failures", []).append(str(exc))
            if claimed:
                try:
                    record["production_unchanged"] = (config is not None and
                        lab.sha(controller.ROOT / "broker/config.json") == config and
                        controller.common.production(lab) == production)
                    require(record["production_unchanged"], "Production changed during link/export")
                    record["builder_after"] = controller.guard.builder_idle(lab)
                    record["host_after"] = controller.guard.host_resources(lab)
                except BaseException as exc:
                    record.setdefault("cleanup_failures", []).append(str(exc))
                if record.get("cleanup_failures") or "failure" in record:
                    record["status"] = "INCOMPLETE"
                record["finished_at"] = lab.now()
                path = controller.OUT / (args.tag + "-link.json")
                controller.save(path, record)
                print(json.dumps({"path": str(path), "sha256": lab.sha(path), "status": record["status"]}))
    require(record["status"] == "LINKED_EXPORTED_NOT_RUNTIME_QUALIFIED", "Link/export incomplete")


def main():
    parser = argparse.ArgumentParser(description="Link only the existing ON observer objects; no compile or GPU run.")
    parser.add_argument("--native", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--tag")
    parser.add_argument("--compile-receipt", type=pathlib.Path, required=True)
    parser.add_argument("--compile-sha256", required=True)
    args = parser.parse_args()
    require(not sys.flags.optimize, "Run Python without -O")
    if args.native:
        native_link(args)
    else:
        host_link(args)


if __name__ == "__main__":
    main()
