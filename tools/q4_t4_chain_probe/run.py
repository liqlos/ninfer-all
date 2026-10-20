import argparse
import collections
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
import urllib.request
import uuid


sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "results/q4-t4-chain"
BINARY = ROOT / "artifacts/experimental/q4-t4-chain-probe"
NATIVE = "/build/q4-t4-chain/host/probe"
IMAGE = "sha256:501c92a90c3d85adc616455a6783bb5ae8b82008f469cf8dc3d6da538f75475b"
FIXTURE = "operators/q4-full-k5120.bin"
FIXTURE_BYTES = 100270080
FIXTURE_SHA = "be10db58b96b5614830c01afe1e28f9d55c4581a6c481827fa7b9d885a6e2790"
HOST = {
    "main.cu": "c4d6bcf55d888de0c9133e331e9578c1f359bd5bd892651697d033df2f99ad44",
    "build_main.py": "97a0ada9710920cbbe27fe893f0d700f80cb69f800d1070638004c1c0f99b913",
}
HELPERS = {
    "profile_long_context.py": "1e6aab81f0138494d2bae3a561cc4cbbbe8870f663ba8f18dd24d895352e0bed",
    "replay_sse_boundary.py": "804ce1340e6a50904c7ac0e711957944bf567d4e94dc16819f208b7f0ce5f203",
}
MUTEXES = [
    "NInferV17RuntimeScreen", "NInferV14RuntimeMeasurement",
    "NInferV15ExportedOperatorSeries", "NInferV14CpuVariantPreparation",
    "NInferJonathanSpecShort20261018",
]
KERNELS = {
    "native": "_ZN6ninfer3ops4gguf6detail3vec6kernelIL9ggml_type12ELi4ELb0ELi5120ELb1ELb0ELi2EEEvNS2_7VecArgsE",
    "chain": "_ZN6ninfer3ops4gguf6detail4q4tc18q4_t4_chain_kernelENS2_7VecArgsE",
}
ROWS = (6144, 17408, 34816)
ORDERS = ("ABBA", "BAAB", "ABBA")
REJECTIONS = [
    "zero_rows", "negative_rows", "unsupported_K", "unsupported_T", "row_bytes", "output_stride",
    "null_weight", "null_qs", "null_ds", "null_output", "weight_alignment", "qs_alignment",
    "ds_alignment", "fp32_alignment", "bf16_alignment", "output_mode",
] * 2 + ["unsupported_chain_rows", "N1024_no_chain", "unknown_route"]
BIND_EXEC = """
import hashlib,json,os,sys
def sha(path):
    with open(path,"rb") as source:
        return hashlib.file_digest(source,"sha256").hexdigest()
binary_sha,fixture_sha,launch_id=sys.argv[1:]
fixture="/fixtures/operators/q4-full-k5120.bin"
if sha("/probe")!=binary_sha or os.stat(fixture).st_size!=100270080 or sha(fixture)!=fixture_sha:
    raise RuntimeError("Fresh read-only bind identity differs")
print(json.dumps({"event":"bind_verified","binary_sha256":binary_sha,"fixture_sha256":fixture_sha,
                  "fixture_bytes":100270080,"launch_id":launch_id,"before_exec":True}),flush=True)
os.execv("/probe",["/probe","--fixtures-root","/fixtures","--launch-id",launch_id])
"""


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def same_fields(row, expected, context):
    for key, value in expected.items():
        require(key in row and row[key] == value and (type(value) is not bool or type(row[key]) is bool),
                context + ": " + key)


def finite(value, minimum=0, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (
        value > minimum if positive else value >= minimum)


def close(actual, expected, field):
    require(type(actual) in (int, float) and math.isfinite(actual) and
            math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8), "Summary differs: " + field)


def route(value, arm, rows):
    same_fields(value, {"kernel": KERNELS[arm], "block_x": 128, "dynamic_shared_bytes": 0,
                        "outside_timing": True}, "Actual " + arm + " route")
    require(type(value["grid_x"]) is int and 0 < value["grid_x"] <= rows, "Invalid grid")
    if arm == "chain":
        require(value["grid_x"] == rows // 16, "Chain CTA ownership differs")


def oracle(value):
    same_fields(value, {
        "samples": 64,
        "minimum_correction": "original half dmin, 6-bit minima and native stored half input sum",
        "planar_to_oracle_adapter": "CPU literal copy only; never used for GPU preparation",
    }, "FP64 oracle")
    require(finite(value["maximum_absolute_error"]) and
            finite(value["maximum_bound_fraction"]) and value["maximum_bound_fraction"] <= 1,
            "FP64 oracle criterion failed")


def validate(events, launch_id, receipt, binary_sha, exit_code):
    kinds = [row.get("event") for row in events]
    if "failure" in kinds:
        require("qualification_complete" in kinds or "screen" not in kinds,
                "Timing occurred before numerical qualification")
        raise RuntimeError("Host qualification/execution failure: " +
                           json.dumps([row for row in events if row.get("event") == "failure"]))
    expected = ["bind_verified", "start", "fixture_verified"] + ["qualification"] * 3
    expected += ["api_guards", "qualification_complete"] + ["screen"] * 6 + ["complete"]
    require(kinds == expected, "Missing, extra or reordered host events")
    same_fields(events[0], {"binary_sha256": binary_sha, "fixture_sha256": FIXTURE_SHA,
                           "fixture_bytes": FIXTURE_BYTES, "launch_id": launch_id,
                           "before_exec": True}, "Fresh RO bind")
    same_fields(events[1], {
        "launch_id": launch_id, "device": "NVIDIA GeForce RTX 3090", "cc": 860,
        "host_receipt": receipt,
        "static_mechanism_review": "PARTIAL; original resource review is not numerical acceptance",
    }, "Host identity")
    same_fields(events[2], {"bytes": FIXTURE_BYTES, "sha256": FIXTURE_SHA,
                           "shared_native_chain_weight_backings": 1, "runtime_repack": False},
                "Full real fixture")
    for row, n in zip(events[3:6], ROWS):
        same_fields(row, {"N": n, "whole_modes_bit_exact": 3, "direct_accumulate_calls": 32,
                          "graph_nodes": 4, "graph_replays": 8,
                          "canaries_and_inputs_unchanged": True, "status": "PASS"}, "Qualification")
        require(valid_sha(row["input_sha256"]) and valid_sha(row["packed_sha256"]), "Input identity missing")
        for arm in ("native", "chain"):
            for mode in ("plain", "store", "accumulate"):
                route(row[arm + "_" + mode + "_route"], arm, n)
            oracle(row[arm + "_fp64_oracle"])
        for mode in ("store", "accumulate"):
            semantics = row[mode + "_semantics"]
            same_fields(semantics, {"elements": n * 4, "calls": 1, "host_rounding_boundary_selftests": 12},
                        "BF16 semantics")
            for key in ("carry_sensitive", "observed_final_halfway_even", "observed_final_halfway_odd",
                        "observed_final_adjacent_halfway"):
                require(type(semantics[key]) is int and 0 <= semantics[key] <= n * 4,
                        "Invalid BF16 semantic counter")
            require(semantics["carry_sensitive"] == 0 if mode == "store" else
                    semantics["carry_sensitive"] > 0, "BF16 carry semantics not exercised")
    api = events[6]
    same_fields(api, {
        "rejection_count": 35, "rejects_without_launch": True, "N1024_native_only": True,
        "buffer_extents_and_disjointness": "caller-owned; API has no byte extent or alias validator",
        "other_devices_or_default_OFF_exercised": False, "status": "PASS",
    }, "API guards")
    require(collections.Counter(api["rejections"]) == collections.Counter(REJECTIONS),
            "API rejection coverage differs")
    route(api["N1024_native_route"], "native", 1024)
    oracle(api["N1024_native_oracle"])
    same_fields(events[7], {"status": "PASS", "shape_count": 3, "mode_case_count": 9,
                           "timing_started": False, "planned_timing_endpoint_count": 6,
                           "N1024_native_guard": True, "all_numerical_and_API_guards": True},
                "Qualification barrier")
    summaries = []
    for row, (n, scope) in zip(events[8:14], [(n, s) for n in ROWS for s in ("projection_only", "prepared")]):
        primary = n == 34816 and scope == "prepared"
        same_fields(row, {"N": n, "scope": scope, "mode": "StoreBF16", "primary": primary,
                          "warmup_order": "ABAB", "calls_per_sample": 32, "CI": False,
                          "fused_gate_up_qualified": False, "end_to_end_claim": False}, "Screen")
        require(len(row["warmups_ms"]) == 4 and all(finite(x, positive=True) for x in row["warmups_ms"]),
                "Invalid warmup samples")
        require(len(row["blocks"]) == 3, "Screen block count differs")
        aa, bb, gains = [], [], []
        for block, order in zip(row["blocks"], ORDERS):
            require(block["order"] == order and len(block["samples_ms"]) == 4 and
                    all(finite(x, positive=True) for x in block["samples_ms"]), "Invalid timing block")
            a = [x for arm, x in zip(order, block["samples_ms"]) if arm == "A"]
            b = [x for arm, x in zip(order, block["samples_ms"]) if arm == "B"]
            gain = 100 * (1 - sum(b) / sum(a))
            close(block["native_ms"], sum(a) / 2, "block.native_ms")
            close(block["chain_ms"], sum(b) / 2, "block.chain_ms")
            close(block["time_gain_percent"], gain, "block.time_gain_percent")
            aa.extend(a); bb.extend(b); gains.append(gain)
        gain = 100 * (1 - sum(bb) / sum(aa))
        positive = sum(x > 0 for x in gains)
        accepted = gain >= 1 and positive >= 2 if primary else gain >= -1
        close(row["native_mean_ms"], sum(aa) / 6, "native_mean_ms")
        close(row["chain_mean_ms"], sum(bb) / 6, "chain_mean_ms")
        close(row["mean_time_gain_percent"], gain, "mean_time_gain_percent")
        same_fields(row, {"positive_blocks": positive, "criterion_pass": accepted}, "Selection")
        summaries.append({"N": n, "scope": scope, "primary": primary, "mean_time_gain_percent": gain,
                          "block_time_gain_percent": gains, "criterion_pass": accepted})
    accepted = all(row["criterion_pass"] for row in summaries)
    same_fields(events[-1], {
        "launch_id": launch_id, "status": "PASS" if accepted else "SCREEN_REJECTED",
        "shape_count": 3, "mode_case_count": 9, "timing_endpoint_count": 6, "numerical_PASS": True,
        "screen_accepted": accepted, "warmup_order": "ABAB", "calls_per_sample": 32,
        "block_orders": list(ORDERS), "CI": False, "end_to_end_claim": False,
        "fused_gate_up_qualified": False, "runtime_integration": False,
    }, "Host completion")
    require(exit_code == (0 if accepted else 2), "Host exit code disagrees with completed screen")
    return {"numerical_pass": True, "correctness_shapes": 3, "mode_cases": 9, "timing_endpoints": 6,
            "all_correctness_before_timing": True, "oracle_samples_per_arm_per_shape": 64,
            "screen_accepted": accepted, "endpoints": summaries,
            "decision": "consider_next_merged_projection_gate" if accepted else "close",
            "statistical_confirmation": False, "end_to_end_claim": False,
            "fused_gate_up_qualified": False, "static_mechanism_review": "PARTIAL"}


def sources():
    for name, expected in HOST.items():
        require(sha(HERE / name) == expected, "Frozen host changed: " + name)
    for name, expected in HELPERS.items():
        require(sha(ROOT / "source/tools/lab" / name) == expected, "Controller helper changed: " + name)
    host = module("q4tc_frozen_host", HERE / "build_main.py")
    generated, expected = host.prepare(ROOT / "source")
    return host, generated, expected


def preflight(path, expected_sha):
    require(valid_sha(expected_sha) and sha(path) == expected_sha, "Supplied actual host build receipt changed")
    build = json.loads(path.read_text(encoding="utf-8"))
    host, generated, expected = sources()
    same_fields(build, expected, "Build/source identity")
    require(build["generated_sha256"] == {name: hashlib.sha256(raw).hexdigest() for name, raw in generated.items()},
            "Built generated helpers differ from frozen raw extraction")
    require(re.search(r"\bV13\.1\.115\b", build["toolchain"]) and
            build["linked_existing_object"] == "/build/q4-t4-chain/chain.o" and
            valid_sha(build["probe_sha256"]) and valid_sha(build["host_object_sha256"]) and
            valid_sha(build["existing_manifest_raw_sha256"]) and build["host_dependency_sha256"],
            "Incomplete linked-host provenance")
    require(build["link_command"][-1] == NATIVE and
            build["compile_command"][-2:] == ["-o", "/build/q4-t4-chain/host/main.o"] and
            "-Xlinker=--wrap=__cudaLaunchKernel" in build["link_command"],
            "Expected host-only build/link recipe differs")
    return build, expected, host


def inspect(lab, target):
    return json.loads(lab.docker("inspect", target).stdout)[0]


def owned(lab, target, ident, name, token):
    value = inspect(lab, target)
    labels = value["Config"].get("Labels") or {}
    require(value["Image"] == IMAGE and value["Name"] == "/" + name and
            (ident is None or value["Id"] == ident) and
            labels.get("io.codex.ninfer-owner") == token and
            labels.get("io.codex.ninfer-experiment") == "q4-t4-chain",
            "Container ownership mismatch; refusing mutation")
    return value


def admission(lab, guard, token=None, ident=None):
    status = lab.api("/v1/status")
    a, w = status["admission"], status["windows_gpu"]
    grant = a["photo_gpu_reservation"]
    require(not status["leases"] and not status["agents"] and not status["queue"] and
            not status["inference"]["active_requests"] and not status["inference"]["active_sequences"] and
            not a["runtime_maintenance"]["active"] and not a["active_batches"] and
            not status.get("external_growth_pressure", {}).get("active"), "Broker or foreign work is active")
    require(grant["active"] is bool(token) and (grant["owner"] == lab.OWNER if token else not a["draining"]),
            "Reservation admission differs")
    require(w["available"] and w["external_reserve_fraction"] == 0.3, "Windows reserve30 admission failed")
    allowed = {"dwm", "explorer", "ChatGPT", "Codex", "chrome", "msedge", "firefox", "csrss"}
    if ident:
        allowed |= {"vmmemWSL", "vmmem"}
    require(all(p.get("process") in allowed for p in w["process_memory"]), "Foreign GPU process")
    require(len(status["capacity_estimate"]) == 1, "Unexpected GPU inventory")
    capacity = status["capacity_estimate"][0]
    require(capacity["sample_age_seconds"] <= 15 and capacity["measurement"] == "measured" and
            capacity["external_reserve_satisfied"] and capacity["available_llm_budget_bytes"] >= 24_000_000_000,
            "Stale or insufficient GPU budget")
    for current in lab.docker("ps", "-q").stdout.split():
        item = inspect(lab, current)
        if item["HostConfig"].get("DeviceRequests") or item["HostConfig"].get("Devices"):
            require(item["Id"] == ident and item["Image"] == IMAGE and
                    (item["Config"].get("Labels") or {}).get("io.codex.ninfer-owner") == token,
                    "Foreign GPU container")
    service = (
        "$s=Get-Service homelab-comfyui -ErrorAction SilentlyContinue;"
        "@{running=([bool]($s -and $s.Status -eq 'Running'));"
        "listening=[bool](Get-NetTCPConnection -State Listen -LocalPort 8188 "
        "-ErrorAction SilentlyContinue)}|ConvertTo-Json -Compress"
    )
    comfy = json.loads(lab.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-WindowStyle",
                              "Hidden", "-Command", service]).stdout)
    if comfy["running"] or comfy["listening"]:
        with urllib.request.urlopen("http://127.0.0.1:8188/queue", timeout=5) as response:
            queue = json.load(response)
        require(not queue["queue_running"] and not queue["queue_pending"], "ComfyUI queue is active")
    return {"at": lab.now(), "host": guard.host_resources(lab), "builder": guard.builder_idle(lab),
            "capacity": capacity, "process_memory": w["process_memory"], "comfy": comfy,
            "owned_reservation": bool(token), "reservation_status": grant}


def production(lab):
    result = lab.docker("inspect", "gpu-broker-ninfer", check=False)
    if result.returncode:
        require("no such object" in result.stderr.lower() or "no such container" in result.stderr.lower(),
                "Cannot inspect production")
        return None
    item = json.loads(result.stdout)[0]
    return {
        "id": item["Id"], "image": item["Image"],
        "state": {key: item["State"][key] for key in ("Status", "Running", "Pid", "StartedAt", "FinishedAt")},
        "config_sha256": hashlib.sha256(json.dumps(item["Config"], sort_keys=True).encode()).hexdigest(),
        "host_config_sha256": hashlib.sha256(json.dumps(item["HostConfig"], sort_keys=True).encode()).hexdigest(),
    }


def exported(lab, builder, build, build_sha):
    pins = {NATIVE: build["probe_sha256"], "/build/q4-t4-chain/host/host_build.json": build_sha,
            "/build/q4-t4-chain/chain.o": build["kernel_object_sha256"],
            "/build/q4-t4-chain/build_manifest.json": build["existing_manifest_raw_sha256"]}
    lines = lab.docker("exec", builder, "sha256sum", *pins).stdout.splitlines()
    actual = {}
    for line in lines:
        value, name = line.split(maxsplit=1)
        actual[name] = value
    require(actual == pins, "Native binary or parent build/object identity changed")
    require(not BINARY.is_dir(), "Selected export path is a directory")
    BINARY.parent.mkdir(parents=True, exist_ok=True)
    lab.docker("cp", builder + ":" + NATIVE,
               lab.WSL_ROOT + "/artifacts/experimental/q4-t4-chain-probe")
    require(BINARY.stat().st_size > 0 and sha(BINARY) == build["probe_sha256"], "Fresh export differs")
    return {"path": str(BINARY), "bytes": BINARY.stat().st_size, "sha256": build["probe_sha256"],
            "native_parent_sha256": pins}


def main():
    parser = argparse.ArgumentParser(description="One isolated exported Q4 T4 operator gate; no build or confirmation.")
    parser.add_argument("--build", type=pathlib.Path, required=True)
    parser.add_argument("--build-sha256", required=True)
    parser.add_argument("--tag", default="screen-r1")
    args = parser.parse_args()
    require(sys.platform == "win32" and not sys.flags.optimize, "Use Windows Python without -O")
    require(re.fullmatch(r"[a-z0-9-]+", args.tag), "Invalid immutable result tag")
    sources()
    sys.path.insert(0, str(ROOT / "source/tools/lab"))
    import profile_long_context as lab
    import replay_sse_boundary as guard
    require(lab.ROOT == ROOT and guard.IMAGE == IMAGE and guard.MUTEXES == MUTEXES, "Lab helper contract differs")
    receipt_path, output = OUT / (args.tag + "-receipt.json"), OUT / (args.tag + ".jsonl")
    require(OUT.is_dir() and not receipt_path.exists() and not output.exists(), "Result directory missing or tag exists")
    build, expected, host = preflight(args.build, args.build_sha256)
    token = uuid.uuid4().hex
    lab.OWNER = "ninfer-q4-t4-chain-" + token
    name, launch_id = "ninfer-lab-q4-t4-chain-" + token[:12], args.tag + "-" + token
    record = {
        "schema": "q4-t4-chain-exported-operator-r1", "status": "INCOMPLETE", "started_at": lab.now(),
        "owner": lab.OWNER, "launch_id": launch_id, "container_name": name,
        "controller_sha256": sha(pathlib.Path(__file__)), "helper_sha256": HELPERS,
        "build_path": str(args.build.resolve()), "build_sha256": args.build_sha256,
        "host_source_sha256": HOST, "kernel_source_sha256": host.FROZEN,
        "parent_compile_manifest_canonical_sha256": host.MANIFEST_CANONICAL_SHA,
        "protocol": expected["protocol"], "statistical_confirmation": False,
        "end_to_end_claim": False, "fused_gate_up_qualified": False, "runtime_integration": False,
    }
    handles, ident, reservation, create_attempted = [], None, None, False
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.ReleaseMutex.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    with receipt_path.open("x", encoding="utf-8") as target:
        try:
            record["production_config_sha256"] = sha(ROOT / "broker/config.json")
            record["production_before"] = production(lab)
            for mutex in MUTEXES:
                handle = kernel.CreateMutexW(None, False, "Local\\" + mutex)
                require(bool(handle), "Cannot create mutex: " + mutex)
                if kernel.WaitForSingleObject(handle, 0) not in (0, 128):
                    kernel.CloseHandle(handle)
                    raise RuntimeError("Laboratory mutex busy: " + mutex)
                handles.append(handle)
            record["mutexes"] = MUTEXES
            record["admission_before"] = admission(lab, guard)
            record["export"] = exported(lab, record["admission_before"]["builder"]["id"],
                                        build, args.build_sha256)
            require((ROOT / "fixtures" / FIXTURE).stat().st_size == FIXTURE_BYTES, "Real fixture extent differs")
            reservation = token
            grant = lab.api("/v1/photo-gpu-reservations", {"owner": lab.OWNER,
                            "reservation_id": token, "ttl_seconds": 900}, "POST", 90)
            same_fields(grant, {"state": "active", "reservation_id": token, "owner": lab.OWNER}, "Owned grant")
            record["reservation"] = grant
            record["admission_after_reservation"] = admission(lab, guard, token)
            command = ["-B", "-c", BIND_EXEC, build["probe_sha256"], FIXTURE_SHA, launch_id]
            create_attempted = True
            created = lab.docker(
                "create", "--name", name, "--label", "io.codex.ninfer-owner=" + token,
                "--label", "io.codex.ninfer-experiment=q4-t4-chain", "--gpus", "all",
                "--mount", "type=bind,source=" + lab.WSL_ROOT +
                "/artifacts/experimental/q4-t4-chain-probe,target=/probe,readonly",
                "--mount", "type=bind,source=" + lab.WSL_ROOT + "/fixtures,target=/fixtures,readonly",
                "--entrypoint", "python3", IMAGE, *command).stdout.strip()
            require(valid_sha(created), "Invalid created container ID")
            ident = created
            record["container_id"] = ident
            item = owned(lab, ident, ident, name, token)
            require(item["Config"]["Entrypoint"] == ["python3"] and item["Config"]["Cmd"] == command,
                    "Bound identity/exec trampoline changed")
            mounts = {m["Destination"]: (m["Source"], m["RW"], m["Type"]) for m in item["Mounts"]}
            require(mounts == {
                "/probe": (lab.WSL_ROOT + "/artifacts/experimental/q4-t4-chain-probe", False, "bind"),
                "/fixtures": (lab.WSL_ROOT + "/fixtures", False, "bind")}, "Fresh RO mounts differ")
            require(item["HostConfig"].get("DeviceRequests"), "Owned container has no GPU request")
            forbidden = ("LD_PRELOAD=", "LD_AUDIT=", "CUDA_INJECTION64_PATH=", "CUPTI_")
            require(not any(v.startswith(forbidden) for v in item["Config"].get("Env", [])),
                    "Container has profiling or injection environment")
            require(sha(BINARY) == build["probe_sha256"], "Export changed before start")
            started = time.monotonic()
            with output.open("x", encoding="utf-8") as stdout:
                process = subprocess.Popen(
                    ["wsl.exe", "-d", "Ubuntu-24.04", "--exec", "/usr/bin/docker", "start", "-a", ident],
                    stdout=stdout, stderr=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
                try:
                    _, record["stderr"] = process.communicate(timeout=450)
                finally:
                    if process.poll() is None:
                        process.kill()
                        _, record["stderr"] = process.communicate(timeout=10)
                    record["attach_exit_code"] = process.returncode
                    record["wall_seconds"] = time.monotonic() - started
            ended = owned(lab, ident, ident, name, token)
            require(not ended["State"]["Running"], "Host is still running after attach")
            record["container_exit_code"] = ended["State"]["ExitCode"]
            require(record["container_exit_code"] == record["attach_exit_code"], "Attach/container exit codes differ")
            events = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
            record["host_events"] = [row.get("event") for row in events]
            record["validation"] = validate(events, launch_id, expected, build["probe_sha256"],
                                             record["container_exit_code"])
            record["admission_after_run"] = admission(lab, guard, token, ident)
            record["status"] = "PASS" if record["validation"]["screen_accepted"] else "SCREEN_REJECTED"
        except BaseException as error:
            record.update(status="FAILED", failure=type(error).__name__ + ": " + str(error))
        finally:
            try:
                if create_attempted:
                    found = lab.docker("inspect", name, check=False)
                    if found.returncode:
                        require("no such object" in found.stderr.lower() or
                                "no such container" in found.stderr.lower(), "Cannot prove owned container absent")
                    else:
                        item = owned(lab, name, ident, name, token)
                        if item["State"]["Running"]:
                            lab.docker("stop", "--time", "5", item["Id"])
                        lab.docker("rm", item["Id"])
                    record["owned_container_removed_or_absent"] = True
                if reservation and (not create_attempted or record.get("owned_container_removed_or_absent")):
                    released = lab.api("/v1/photo-gpu-reservations/" + reservation,
                                       {"owner": lab.OWNER}, "DELETE")
                    same_fields(released, {"reservation_id": reservation, "state": "released"}, "Owned release")
                    record["reservation_released"] = True
            except BaseException as error:
                record.update(status="FAILED", cleanup_failure=type(error).__name__ + ": " + str(error))
            try:
                if output.exists():
                    record["output_sha256"] = sha(output)
                if "production_config_sha256" in record:
                    require(sha(ROOT / "broker/config.json") == record["production_config_sha256"],
                            "Production configuration changed")
                if "production_before" in record:
                    require(production(lab) == record["production_before"], "Production container changed")
                require(sha(args.build) == args.build_sha256 and
                        sha(pathlib.Path(__file__)) == record["controller_sha256"], "Immutable input changed")
                if "export" in record:
                    require(sha(BINARY) == build["probe_sha256"], "Export changed during gate")
                sources()
                record["sources_and_production_preserved"] = True
            except BaseException as error:
                record.update(status="FAILED", preservation_failure=type(error).__name__ + ": " + str(error))
            for handle in reversed(handles):
                released, closed = kernel.ReleaseMutex(handle), kernel.CloseHandle(handle)
                if not released or not closed:
                    record.update(status="FAILED", mutex_cleanup_failure=True)
            record["finished_at"] = lab.now()
            json.dump(record, target, indent=2)
            target.write("\n")
            target.flush()
            print(json.dumps(record, indent=2), flush=True)
    return {"PASS": 0, "SCREEN_REJECTED": 2}.get(record["status"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
