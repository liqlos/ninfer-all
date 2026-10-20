import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import pathlib
import re
import subprocess
import sys
import time
import uuid


sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "results/mmq-two-input"
BINARY = ROOT / "artifacts/experimental/mmq-two-input-probe"
NATIVE = "/build/mmq-two-input/host/probe"
IMAGE = "sha256:501c92a90c3d85adc616455a6783bb5ae8b82008f469cf8dc3d6da538f75475b"
FIXTURES = {
    "q6": ("operators/q6-full-k17408.bin", 73113600,
           "a381c6be36f572170ccc9c7e4a3f8b25e5061e4eb633649399f3ec9759cb43e2"),
    "q4": ("operators/q4-full-k5120.bin", 100270080,
           "be10db58b96b5614830c01afe1e28f9d55c4581a6c481827fa7b9d885a6e2790"),
}
CASES = {"Q6Down": (5120, 17408, 400), "Q4Gate": (17408, 5120, 1360),
         "Q4Up": (17408, 5120, 1360)}
ORDERS = ["ABBA", "BAAB", "ABBA"]
BIND = """
import hashlib,json,os,sys
binary_sha,fixtures,launch_id=sys.argv[1:]
def sha(path):
    with open(path,'rb') as source:
        return hashlib.file_digest(source,'sha256').hexdigest()
if sha('/probe')!=binary_sha:
    raise RuntimeError('Fresh executable bind hash differs')
for family,(path,size,digest) in json.loads(fixtures).items():
    name='/fixtures/'+path
    if os.stat(name).st_size!=size or sha(name)!=digest:
        raise RuntimeError('Fresh real fixture bind differs: '+family)
print(json.dumps({'event':'bind_verified','binary_sha256':binary_sha,'launch_id':launch_id,
                  'fixture_sha256':{key:value[2] for key,value in json.loads(fixtures).items()},
                  'before_exec':True}),flush=True)
os.execv('/probe',['/probe','--fixtures-root','/fixtures','--launch-id',launch_id])
"""


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def close(actual, expected, context):
    require(type(actual) in (int, float) and math.isfinite(actual) and
            math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8), context)


def validate(events, launch_id, binary_sha, exit_code, expected):
    kinds = [row.get("event") for row in events]
    require("failure" not in kinds, "Executable reported a qualification failure")
    require(kinds == ["bind_verified", "start"] + ["fixture_verified"] * 2 +
            ["qualification"] * 3 + ["api_guards", "qualification_complete"] +
            ["screen"] * 3 + ["complete"], "Missing, extra or reordered host events")
    bind, start = events[:2]
    require(bind == {"event": "bind_verified", "binary_sha256": binary_sha,
                     "launch_id": launch_id, "fixture_sha256": {k: v[2] for k, v in FIXTURES.items()},
                     "before_exec": True}, "Fresh read-only bind identity differs")
    require(start["launch_id"] == launch_id and start["cc"] == 860 and
            start["device"] == "NVIDIA GeForce RTX 3090", "Actual executable/device identity differs")
    require(start["host_receipt"] == expected, "Executed host source/protocol identity differs")
    require({r["family"] for r in events[2:4]} == set(FIXTURES), "Real fixture families differ")
    for row in events[2:4]:
        _, size, digest = FIXTURES[row["family"]]
        require(row["bytes"] == size and row["sha256"] == digest, "Real fixture identity differs")
    for row, (case, (n, k, blocks)) in zip(events[4:7], CASES.items()):
        require(row["case"] == case and row["rows"] == n and row["k"] == k and
                row["columns"] == 1280 and row["samples"] == 2 and row["status"] == "PASS",
                "Case or qualification status differs")
        require(row["layout"] == ("D4" if case == "Q6Down" else "DS4"), "Native packed layout differs")
        require(len(row["input_sha256"]) == len(row["packed_sha256"]) == 2 and
                len(set(row["input_sha256"])) == len(set(row["packed_sha256"])) == 2,
                "Two independently different input panels were not exercised")
        require(row["plain"]["exact"] is True and row["store"]["exact"] is True and
                row["accumulate"]["exact"] is True and row["carry32"]["exact"] is True and
                row["carry32"]["calls"] == 32, "Exact output/carry gate failed")
        require(row["store"]["semantic_elements"] == row["accumulate"]["semantic_elements"] ==
                2 * n * 1280, "Independent BF16 semantic coverage differs")
        require(row["accumulate"]["carry_sensitive"] > 0 and row["carry32"]["carry_sensitive"] > 0,
                "Nonzero carry was not exercised")
        require(row["carry32"]["semantic_samples"] == 128 and row["api_negative_calls"] == 168,
                "Independent recurrence or compiled API coverage differs")
        for arm in ("control", "candidate"):
            samples = row["plain"]["oracle_" + arm]
            require(len(samples) == 2, "Independent oracle missed an input panel")
            for oracle in samples:
                require(oracle["samples"] == 64 and
                        0 <= oracle["maximum_bound_fraction"] <= 1 and
                        math.isfinite(oracle["maximum_absolute_error"]), "FP64 oracle gate failed")
            for key in ("routes", "store_routes", "accumulate_routes"):
                route = row[key][arm]
                require(route["projection_launches"] == (2 if arm == "control" else 1) and
                        route["grid_x"] == ([blocks, blocks] if arm == "control" else [2 * blocks]) and
                        route["block"] == [32, 8, 1] and route["dynamic_shared_bytes"] == 57856 and
                        route["pack_launches"] == 2 and route["outside_timing_events"] is True,
                        "Actual pack/projection dispatch differs")
        require(all(row[key] is True for key in
                    ("canaries", "padding", "immutable_weights", "immutable_inputs", "immutable_packs")),
                "Buffer/immutability gate failed")
    require(events[7]["status"] == events[8]["status"] == "PASS" and
            events[7]["api_negative_calls"] == 504 and events[7]["no_kernel_launches"] is True and
            events[7]["outputs_unchanged"] is True and events[8]["cases"] == 3 and
            events[8]["all_correctness_before_timing"] is True, "Compiled API/qualification barrier failed")
    endpoints = []
    for row, case in zip(events[9:12], CASES):
        require(row["case"] == case and row["mode"] == "StoreBf16" and
                row["scope"] == "two_native_packs_plus_projection" and len(row["blocks"]) == 3,
                "Timing case/scope differs")
        require(row["calls_per_sample"] == 8, "Timing sample length differs")
        aa, bb, gains = [], [], []
        for index, (block, order) in enumerate(zip(row["blocks"], ORDERS)):
            require(block["index"] == index and block["order"] == order and
                    len(block["milliseconds"]) == 4 and
                    all(type(v) in (int, float) and math.isfinite(v) and v > 0
                        for v in block["milliseconds"]), "Invalid timing block")
            a = [v for arm, v in zip(order, block["milliseconds"]) if arm == "A"]
            b = [v for arm, v in zip(order, block["milliseconds"]) if arm == "B"]
            gain = 100 * (1 - sum(b) / sum(a))
            close(block["control_mean_ms"], sum(a) / 2, "Control block mean differs")
            close(block["candidate_mean_ms"], sum(b) / 2, "Candidate block mean differs")
            close(block["time_reduction_percent"], gain, "Block time reduction differs")
            aa.extend(a); bb.extend(b); gains.append(gain)
        gain = 100 * (1 - sum(bb) / sum(aa))
        close(row["mean_time_reduction_percent"], gain, "Endpoint reduction differs")
        require(row["positive_blocks"] == sum(v > 0 for v in gains), "Positive-block count differs")
        endpoints.append({"case": case, "control_mean_ms": sum(aa) / 6,
                          "candidate_mean_ms": sum(bb) / 6, "time_reduction_percent": gain,
                          "block_time_reduction_percent": gains})
    done = events[-1]
    selected = endpoints[0]["time_reduction_percent"] >= 10 and all(
        v > 0 for v in endpoints[0]["block_time_reduction_percent"])
    require(exit_code == 0 and done["status"] == "PASS" and done["cases"] == 3 and
            done["timing_blocks"] == 9 and done["primary_screen_pass"] is selected and
            done["runtime_qualified"] is False and done["full_ffn_qualified"] is False,
            "Completed primary screen differs")
    close(done["primary_mean_time_reduction_percent"], endpoints[0]["time_reduction_percent"],
          "Primary reduction differs")
    require(done["decision"] == ("PROCEED_TO_PAIRED_FFN_GATE" if selected else "KILL_OPERATOR_SCREEN"),
            "Selection decision differs")
    return {"numerical_pass": True, "screen_selected": selected, "endpoints": endpoints,
            "statistical_confirmation": False, "runtime_qualified": False, "full_ffn_qualified": False}


def preflight(path, expected_sha):
    require(sha(path) == expected_sha, "Host build receipt changed")
    build = json.loads(path.read_text(encoding="utf-8"))
    recipe = module("two_input_host_recipe", HERE / "generate_host.py")
    generated, expected = recipe.prepare(ROOT / "source")
    require(all(build.get(key) == value for key, value in expected.items()),
            "Host metadata/source identity differs")
    require(build["generated_sha256"] == {name: hashlib.sha256(data).hexdigest()
                                           for name, data in generated.items()},
            "Raw extracted host helpers differ")
    require(build["kernel_recompiled"] is False and build["GPU_run"] is False,
            "Unexpected build scope")
    return build, expected


def owned(lab, ident, name, token):
    item = json.loads(lab.docker("inspect", ident).stdout)[0]
    require(item["Name"] == "/" + name and item["Image"] == IMAGE and
            item["Config"]["Labels"]["io.codex.ninfer-owner"] == token and
            item["Config"]["Labels"]["io.codex.ninfer-experiment"] == "mmq-two-input",
            "Container ownership differs")
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=pathlib.Path, required=True)
    parser.add_argument("--build-sha256", required=True)
    parser.add_argument("--tag", default="screen-r1")
    args = parser.parse_args()
    require(sys.platform == "win32" and not sys.flags.optimize, "Use Windows Python without -O")
    require(re.fullmatch(r"[a-z0-9-]+", args.tag), "Invalid result tag")
    build, expected = preflight(args.build, args.build_sha256)
    sys.path.insert(0, str(HERE.parent / "lab"))
    import profile_long_context as lab
    import replay_sse_boundary as guard
    common = module("mmq_admission", HERE.parent / "q4_t4_chain_probe/run.py")
    OUT.mkdir(parents=True, exist_ok=True)
    receipt, output = OUT / (args.tag + "-receipt.json"), OUT / (args.tag + ".jsonl")
    require(not receipt.exists() and not output.exists(), "Immutable result already exists")
    token = uuid.uuid4().hex
    lab.OWNER = "ninfer-mmq-two-input-" + token
    name, launch_id = "ninfer-mmq-two-input-" + token[:12], args.tag + "-" + token
    record = {"status": "INCOMPLETE", "started_at": lab.now(), "launch_id": launch_id,
              "controller_sha256": sha(pathlib.Path(__file__)), "build_sha256": args.build_sha256,
              "helper_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                                (pathlib.Path(common.__file__), pathlib.Path(lab.__file__), pathlib.Path(guard.__file__))},
              "runtime_qualified": False, "full_ffn_qualified": False, "statistical_confirmation": False}
    handles, ident, attempted, reservation = [], None, False, None
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.ReleaseMutex.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    try:
        for mutex in common.MUTEXES:
            h = kernel.CreateMutexW(None, False, "Local\\" + mutex)
            require(bool(h), "Cannot create lab mutex: " + mutex)
            if kernel.WaitForSingleObject(h, 0) not in (0, 128):
                kernel.CloseHandle(h)
                raise RuntimeError("Lab mutex busy: " + mutex)
            handles.append(h)
        record["config_sha256"] = sha(ROOT / "broker/config.json")
        record["production_before"] = common.production(lab)
        record["admission_before"] = common.admission(lab, guard)
        binary_sha = build["probe_sha256"]
        builder = record["admission_before"]["builder"]["id"]
        pins = {NATIVE: binary_sha, "/build/mmq-two-input/host/host_build.json": args.build_sha256,
                "/build/mmq-two-input/build_manifest.json": build["compile_manifest_sha256"],
                "/build/mmq-two-input/resource_gate.json": build["resource_gate_sha256"]}
        for value in build["linked_existing_objects"].values():
            pins[value["path"]] = value["sha256"]
        actual = {}
        for line in lab.docker("exec", builder, "sha256sum", *pins).stdout.splitlines():
            digest, name_in_builder = line.split(maxsplit=1)
            actual[name_in_builder] = digest
        require(actual == pins, "Native executable/build/object/resource identity changed")
        record["native_sha256"] = pins
        record["protocol"] = expected["protocol"]
        BINARY.parent.mkdir(parents=True, exist_ok=True)
        lab.docker("cp", builder + ":" + NATIVE,
                   lab.WSL_ROOT + "/artifacts/experimental/mmq-two-input-probe")
        require(sha(BINARY) == binary_sha, "Exported executable changed")
        reservation = token
        grant = lab.api("/v1/photo-gpu-reservations", {"owner": lab.OWNER,
                        "reservation_id": token, "ttl_seconds": 900}, "POST", 90)
        require(grant["state"] == "active" and grant["reservation_id"] == token and
                grant["owner"] == lab.OWNER, "Owned reservation failed")
        record["reservation"] = grant
        record["admission_reserved"] = common.admission(lab, guard, token)
        command = ["-B", "-c", BIND, binary_sha, json.dumps(FIXTURES), launch_id]
        mounts = [(lab.WSL_ROOT + "/artifacts/experimental/mmq-two-input-probe", "/probe")]
        mounts += [(lab.WSL_ROOT + "/fixtures/" + path, "/fixtures/" + path)
                   for path, _, _ in FIXTURES.values()]
        docker_args = ["create", "--name", name, "--label", "io.codex.ninfer-owner=" + token,
                       "--label", "io.codex.ninfer-experiment=mmq-two-input", "--gpus", "all"]
        for source, target in mounts:
            docker_args += ["--mount", "type=bind,source=" + source + ",target=" + target + ",readonly"]
        attempted = True
        ident = lab.docker(*docker_args, "--entrypoint", "python3", IMAGE, *command).stdout.strip()
        item = owned(lab, ident, name, token)
        require(item["Config"]["Entrypoint"] == ["python3"] and item["Config"]["Cmd"] == command,
                "Bound executable trampoline differs")
        require({m["Destination"]: (m["Source"], m["RW"], m["Type"]) for m in item["Mounts"]} ==
                {target: (source, False, "bind") for source, target in mounts}, "Read-only mounts differ")
        forbidden = ("LD_PRELOAD=", "LD_AUDIT=", "CUDA_INJECTION64_PATH=", "CUPTI_")
        require(not any(v.startswith(forbidden) for v in item["Config"].get("Env", [])),
                "Profiling or injection environment present")
        record["container_id"] = ident
        started = time.monotonic()
        ran = lab.docker("start", "-a", ident, timeout=450, check=False)
        record.update(wall_seconds=time.monotonic() - started, stderr=ran.stderr,
                      attach_exit_code=ran.returncode)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(ran.stdout)
        item = owned(lab, ident, name, token)
        require(not item["State"]["Running"] and item["State"]["ExitCode"] == ran.returncode,
                "Attach/container state differs")
        record["validation"] = validate([json.loads(s) for s in ran.stdout.splitlines() if s.strip()],
                                        launch_id, binary_sha, ran.returncode, expected)
        record["admission_after"] = common.admission(lab, guard, token, ident)
        record["status"] = "SELECTED" if record["validation"]["screen_selected"] else "SCREEN_REJECTED"
    except BaseException as error:
        record.update(status="FAILED", failure=type(error).__name__ + ": " + str(error))
    finally:
        try:
            if attempted:
                found = lab.docker("inspect", name, check=False)
                if found.returncode:
                    require("no such" in found.stderr.lower(), "Cannot prove owned container absent")
                else:
                    item = owned(lab, name, name, token)
                    if item["State"]["Running"]:
                        lab.docker("stop", "--time", "5", item["Id"])
                    lab.docker("rm", item["Id"])
                record["owned_container_removed_or_absent"] = True
            if reservation and (not attempted or record.get("owned_container_removed_or_absent")):
                released = lab.api("/v1/photo-gpu-reservations/" + token, {"owner": lab.OWNER}, "DELETE")
                require(released["state"] == "released" and released["reservation_id"] == token,
                        "Owned reservation release failed")
                record["reservation_released"] = True
            if "production_before" in record:
                require(common.production(lab) == record["production_before"] and
                        sha(ROOT / "broker/config.json") == record["config_sha256"], "Production changed")
                record["production_preserved"] = True
            require(sha(args.build) == args.build_sha256 and
                    sha(pathlib.Path(__file__)) == record["controller_sha256"], "Controller/build changed")
            preflight(args.build, args.build_sha256)
            if "native_sha256" in record:
                require(sha(BINARY) == build["probe_sha256"], "Export changed during GPU gate")
        except BaseException as error:
            record.update(status="FAILED", cleanup_or_preservation_failure=type(error).__name__ + ": " + str(error))
        for handle in reversed(handles):
            kernel.ReleaseMutex(handle)
            kernel.CloseHandle(handle)
        record["finished_at"] = lab.now()
        if output.exists():
            record["output_sha256"] = sha(output)
        with receipt.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
        print(json.dumps({key: record.get(key) for key in
                          ("status", "failure", "cleanup_or_preservation_failure", "reservation_released")}), flush=True)
    return {"SELECTED": 0, "SCREEN_REJECTED": 2}.get(record["status"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
