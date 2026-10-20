import ctypes
import hashlib
import json
import math
import pathlib
import re
import subprocess
import sys
import time
import uuid


sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[3]
PROBE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source/tools/lab"))
import profile_long_context as lab
import replay_sse_boundary as guard
import run_switch_matrix as matrix


OUT = ROOT / "results/attention-v-shift"
IMAGE = "sha256:501c92a90c3d85adc616455a6783bb5ae8b82008f469cf8dc3d6da538f75475b"
BINARY = ROOT / "artifacts/experimental/attention-v-shift-probe"
ENTRYPOINT = "/lab/artifacts/experimental/attention-v-shift-probe"
EXPORT = OUT / "host-export-r1.json"
PROTOCOL = OUT / "screen-protocol-r2.json"
TAG = "screen-r2"
TIMING_SCOPE = "full_attention_launcher_cuda_event_loop"
PASS = "PASS_EXPORTED_ATTENTION_CORRECTNESS_AND_EXPLORATORY_SCREEN"
SUFFIX = ("INS0_23CausalAttentionGeometryILi24ELi4ELi1EEENS0_20PagedKVBatchMetadataILb0EEE"
          "Li8ELb0ELNS0_11KvKeyCodingE0EEEvPK13__nv_bfloat16PKaSB_PK6__halfSE_T0_PKifPS7_i")
KERNELS = {
    "off": "_ZN6ninfer3ops50causal_attention_prompt_i8_fast_v_shift_off_kernel" + SUFFIX,
    "on": "_ZN6ninfer3ops49causal_attention_prompt_i8_fast_v_shift_on_kernel" + SUFFIX,
}
CASES = [
    ("tail1-valid1-low-boundary", 1, 192, 256),
    ("tail33-valid63-high-boundary", 33, 158, 256),
    ("tail127-valid1-maxfinite", 127, 130, 384),
    ("tail129-valid63-stage-reuse", 129, 126, 384),
    ("long-T1280-base31488", 1280, 31488, 65536),
]
ORDERS = ["ABBA", "BAAB", "ABBA"]
FROZEN = {
    "generate.py": "cb252ee03bffbf414ea2dc2ec96adf93b30ecd59aa6f953fdbcf77275ce1c8c8",
    "variant.cu": "63d701e5c6680dfa645080ae6121552e92caea9108ef1e6742adeeefd7ca3a80",
    "probe_api.h": "8502a3ae87dd1ce19828615440407dde92c6609b93ced5546c58f362a97f6d83",
    "build.sh": "f4c9d4d35dd471c34b7a109322f72d93b420177b0652ec8205a1940f99babb6e",
}
HOST = {
    "main.cu": "18aeef36bcde1a43cc46df7b3b710f4f5c8337c89fd59f0bf71d70ea0d73fe67",
    "generate_host.py": "9c425f1a4aac97d0c25751bf8b5ecb8f09458d7cb45ebde5f13cf698f6bc742c",
    "build_main.sh": "4c2f9da6669fc02c7a5a45e51d303fc32480ed5512aa2761a03d3821ecfdab67",
}
HELPERS = ["profile_long_context.py", "replay_sse_boundary.py",
           "run_switch_matrix.py", "build_switch_matrix.py"]
SELECTION = {
    "minimum_time_reduction_percent": 1.0,
    "all_blocks_positive": True,
    "action_if_met": "consider_independent_confirmation",
    "action_otherwise": "close",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def finite(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (
        value > 0 if positive else value >= 0)


def check_pins(root, pins):
    require(bool(pins), "Empty source pin map")
    for name, expected in pins.items():
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()) and re.fullmatch(r"[0-9a-f]{64}", expected),
                "Invalid source pin: " + name)
        require(lab.sha(path) == expected, "Source pin changed: " + name)


def check_sources(generation):
    require(generation["frozen_source_sha256"] == FROZEN, "Frozen kernel source pins changed")
    require(generation["host_source_sha256"] == HOST, "Frozen host source pins changed")
    check_pins(PROBE, FROZEN)
    check_pins(PROBE, HOST)
    for field, count in (("raw_helper_pins", 5), ("native_source_pins", 14)):
        require(len(generation[field]) == count, "Source pin count changed: " + field)
        check_pins(ROOT / "source", generation[field])


def preflight(export, protocol):
    require(export["status"] == "EXPORTED_LINKED_ATTENTION_V_SHIFT_HOST_NOT_GPU_QUALIFIED"
            and export["gpu_used"] is False and export["builder"]["image"] == IMAGE,
            "Export identity or qualification status changed")
    require(BINARY.stat().st_size == export["binary_bytes"] > 0
            and lab.sha(BINARY) == export["binary_sha256"], "Exported binary changed")
    generation = export["generation"]
    check_sources(generation)
    for name, expected in {
        "schema": 1, "scope": "V_shift_paired_correctness_and_exploratory_attention_screen",
        "cuda_version": "13.1.115", "architecture": "sm_86",
        "compiled_dependency_count": 428, "compiled_dependencies_checked": True,
        "compiled_dependency_sha256": "724b1137d46e612b9ed12260f2ccd77996681e0c9bba52e9053f0baa5e036471",
        "kernel_recompiled": False, "gpu_run": False, "source_only": False,
        "check_only": False, "launch_wrapper": "__cudaLaunchKernel",
        "relative_l2": 0.00315, "gross_absolute": 0.0011,
        "gross_relative_to_max_reference": 0.0078125,
        "short_cases_tokens_base": [[1, 192], [33, 158], [127, 130], [129, 126]],
        "long_case": {"tokens": 1280, "base": 31488, "context_capacity": 65536,
                      "FP64_token_head_rows": [[0, 0], [127, 7], [128, 12], [1279, 23]],
                      "synthetic_represented_inputs": True},
        "screen": {"warmup_order": "ABAB", "block_orders": ORDERS, "calls_per_sample": 32,
                   "long_case_only": True, "scope": "full_attention",
                   "statistical_confirmation": False},
    }.items():
        require(generation[name] == expected, "Generation contract changed: " + name)
    for name, expected in {
        "off.o": "0b3dfa49b6017dc59f9545a12d94adb205b95eb755401269faad70b671db2f69",
        "on.o": "22214d9e34b536756d11dc107b00eadf2f390f7ebd736d1f3549364ff04b0ae7",
    }.items():
        require(generation["compiled_artifact_pins"][name] == expected, "Compiled object changed: " + name)
    require(len(generation["compiled_artifact_pins"]) == 8
            and len(generation["generated_sha256"]) == 4
            and len(generation["raw_regions"]) == 11
            and all(r["raw_without_substitutions"] is True for r in generation["raw_regions"].values()),
            "Generated helper or compiled artifact coverage changed")
    require(export["exact_native_fatbins_preserved"] is True, "Native fatbins were not preserved exactly")
    fatbins = export["kernel_fatbin_containment"]
    require(fatbins["linked"]["bytes"] == 2109216
            and re.fullmatch(r"[0-9a-f]{64}", fatbins["linked"]["sha256"]),
            "Linked fatbin identity changed")
    for arm in KERNELS:
        original = list(export["cubins"][arm].values())
        require(len(original) == 1 and re.fullmatch(r"[0-9a-f]{64}", original[0]),
                "Original cubin identity missing: " + arm)
        proof = fatbins[arm]
        size, offset = (1050936, 3848) if arm == "off" else (1054432, 1054784)
        require(proof["bytes"] == size and proof["identical_byte_offset_in_linked_fatbin"] == offset
                and re.fullmatch(r"[0-9a-f]{64}", proof["sha256"])
                and offset + size <= fatbins["linked"]["bytes"],
                "Exact original fatbin containment changed: " + arm)
    require(fatbins["off"]["sha256"] != fatbins["on"]["sha256"], "OFF/ON fatbin identities coincide")
    resource_text = export["resources_linked"]
    require(set(re.findall(r"^arch = (\S+)", resource_text, re.M)) == {"sm_86"},
            "Linked architecture changed")
    resources = {}
    functions = re.findall(r"^\s*Function (\S+):\s*\n\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)",
                           resource_text, re.M)
    require(len(functions) == 2 and {row[0] for row in functions} == set(KERNELS.values()),
            "Linked kernel resource coverage changed")
    for arm, kernel in KERNELS.items():
        row = next(row for row in functions if row[0] == kernel)
        values = tuple(int(value) for value in row[1:])
        require(values == ((255, 8, 0, 0) if arm == "off" else (255, 0, 16, 0)),
                "Linked resource gate failed: " + arm)
        resources[arm] = dict(zip(("registers", "stack_bytes", "static_shared_bytes", "local_bytes"), values))
        resources[arm].update(dynamic_shared_bytes=100352, total_shared_bytes=100352 + values[2])
    for name, expected in {
        "schema": 1, "tag": TAG, "scope": TIMING_SCOPE,
        "host_export_sha256": lab.sha(EXPORT), "controller_sha256": lab.sha(pathlib.Path(__file__)),
        "correctness_cases": [case[0] for case in CASES], "all_correctness_before_timing": True,
        "synthetic_represented_inputs": True, "model_activation_capture": False,
        "long_case": generation["long_case"],
        "warmups": {"order": "ABAB", "samples": 4, "calls_per_sample": 32},
        "blocks": {"orders": ORDERS, "samples_per_block": 4, "calls_per_sample": 32},
        "selection_rule": SELECTION, "no_outlier_deletion": True,
        "no_optional_extension_or_second_screen": True, "statistical_confirmation": False,
        "cupti": False, "includes_possible_host_attribute_gaps": True,
        "full_request_result": False,
    }.items():
        require(protocol[name] == expected, "Preregistered protocol changed: " + name)
    require(set(protocol["helper_sha256"]) == set(HELPERS), "Controller helper pin coverage changed")
    check_pins(ROOT / "source/tools/lab", protocol["helper_sha256"])
    require(protocol["production_config_sha256"] == lab.sha(ROOT / "broker/config.json"),
            "Preregistered production configuration changed")
    return resources, {k: v for k, v in generation.items() if k not in ("generated_sha256", "check_only")}


def validate(rows, expected_receipt):
    expected_events = (["attention_v_shift_host_start"] + ["paired_qualification"] * 5
                       + ["exploratory_attention_screen", "attention_v_shift_host_complete"])
    require([r["event"] for r in rows] == expected_events, "Incomplete, failed or reordered host output")
    start = rows[0]
    require(start["device"] == "NVIDIA GeForce RTX 3090" and start["runtime_version"] == 13010
            and start["correctness_only"] is False, "Host device/runtime/mode changed")
    require(start["source_receipt"] == expected_receipt, "Embedded source receipt differs from export")
    for row, (name, tokens, base, capacity) in zip(rows[1:6], CASES):
        long_case = name == CASES[-1][0]
        for field, expected in {
            "case": name, "tokens": tokens, "base": base, "context_capacity": capacity,
            "visible_key_extent": base + tokens, "last_page_valid_keys": (base + tokens - 1) % 64 + 1,
            "mapping": "identity" if long_case else "fragmented",
            "bf16_elements_compared": tokens * 24 * 256, "synthetic_represented_inputs": True,
            "model_activation_capture": False, "all_bf16_outputs_bit_exact": True,
            "finite_and_canaries_passed": True, "cache_and_inputs_byte_immutable": True,
            "guards_passed": True,
        }.items():
            require(row[field] == expected, name + ": qualification changed: " + field)
        for arm, kernel in KERNELS.items():
            require(row[arm + "_route"] == {
                "kernel": kernel, "grid_x": (tokens + 127) // 128, "grid_y": 24, "block_x": 256,
                "dynamic_shared_bytes": 100352, "outside_timing_events": True,
            }, name + ": actual route changed: " + arm)
        oracle = row["oracle"]
        for field, expected in {
            "passed": True, "query_head_rows": 4 if long_case else tokens * 24,
            "output_elements": 1024 if long_case else tokens * 24 * 256,
            "full_output_oracle": not long_case, "represented_physical_bytes_decoded": True,
            "key_inverse_hadamard_precision": "FP64", "relative_l2_limit": 0.00315,
            "gross_absolute": 0.0011, "gross_relative_to_max_reference": 0.0078125,
        }.items():
            require(oracle[field] == expected, name + ": oracle contract changed: " + field)
        require(finite(oracle["gross_error_limit"], True), name + ": invalid gross error limit")
        for arm in KERNELS:
            relative, absolute = oracle[arm + "_relative_l2"], oracle[arm + "_max_absolute_error"]
            require(finite(relative) and relative <= 0.00315 and finite(absolute)
                    and absolute <= oracle["gross_error_limit"], name + ": FP64 criterion failed: " + arm)
    screen = rows[6]
    for field, expected in {
        "case": CASES[-1][0], "timing_scope": "full_attention_only",
        "calls_per_sample": 32, "warmup_samples": 4, "warmup_order": "ABAB",
        "same_cache_and_activation_all_arms": True, "repacking_or_uploads_between_arms": False,
        "diagnostics_or_copies_inside_events": False, "post_block_exactness_and_immutability_passed": True,
        "statistical_confirmation": False, "full_request_result": False,
    }.items():
        require(screen[field] == expected, "Screen contract changed: " + field)
    warm = screen["warmup_ms_per_call"]
    require(len(warm) == 4 and all(finite(x, True) for x in warm), "Invalid warmup samples")
    require(len(screen["blocks"]) == 3, "Screen block count changed")
    a, b, reductions = [], [], []
    for index, (block, order) in enumerate(zip(screen["blocks"], ORDERS)):
        values = block["ms_per_call"]
        require(block["block"] == index and block["order"] == order and len(values) == 4
                and all(finite(x, True) for x in values), "Invalid screen block: " + str(index))
        aa = [value for arm, value in zip(order, values) if arm == "A"]
        bb = [value for arm, value in zip(order, values) if arm == "B"]
        a.extend(aa)
        b.extend(bb)
        reductions.append(100.0 * (sum(aa) - sum(bb)) / sum(aa))
    reduction = 100.0 * (sum(a) - sum(b)) / sum(a)
    for field, expected in (("off_mean_ms", sum(a) / 6), ("on_mean_ms", sum(b) / 6),
                            ("time_reduction_percent", reduction)):
        require(type(screen[field]) in (int, float) and math.isfinite(screen[field])
                and math.isclose(screen[field], expected, rel_tol=1e-10, abs_tol=1e-8),
                "Reported timing summary differs from all raw samples: " + field)
    require(rows[-1] == {
        "event": "attention_v_shift_host_complete", "qualified_cases": 5, "screen_completed": True,
        "statistical_confirmation": False, "full_request_result": False,
    }, "Host did not complete the fixed screen")
    eligible = reduction >= 1.0 and all(value > 0 for value in reductions)
    return {
        "correctness_cases": 5, "all_correctness_before_timing": True,
        "qualification_order_evidence": "frozen_host_control_flow_and_ordered_output",
        "long_bf16_elements_compared": 1280 * 24 * 256, "long_bf16_bytes_per_arm": 1280 * 24 * 256 * 2,
        "long_FP64_rows": 4, "long_FP64_output_elements": 1024, "long_full_FP64_oracle": False,
        "timing_endpoints": 1, "timing_scope": TIMING_SCOPE, "includes_possible_host_attribute_gaps": True,
        "off_mean_ms": sum(a) / 6, "on_mean_ms": sum(b) / 6, "time_reduction_percent": reduction,
        "block_time_reduction_percent": reductions, "selection_rule": SELECTION,
        "decision": SELECTION["action_if_met" if eligible else "action_otherwise"],
        "statistical_confirmation": False, "full_request_result": False,
    }


def production_snapshot():
    result = lab.docker("inspect", "gpu-broker-ninfer", check=False)
    if result.returncode:
        require("no such object" in result.stderr.lower() or "no such container" in result.stderr.lower(),
                "Cannot inspect production container: " + result.stderr)
        return None
    item = json.loads(result.stdout)[0]
    return {
        "id": item["Id"], "image": item["Image"],
        "state": {key: item["State"][key] for key in ("Status", "Running", "Pid", "StartedAt", "FinishedAt")},
        "config_sha256": digest(json.dumps(item["Config"], sort_keys=True).encode()),
        "host_config_sha256": digest(json.dumps(item["HostConfig"], sort_keys=True).encode()),
    }


def main():
    require(sys.platform == "win32" and len(sys.argv) == 1, "Run once in Windows Python without arguments")
    receipt, output = OUT / (TAG + "-receipt.json"), OUT / (TAG + ".jsonl")
    require(not output.exists(), "Immutable screen output already exists")
    record = {"status": "INCOMPLETE", "started_at": lab.now(), "exploratory_only": True,
              "synthetic_represented_inputs": True, "model_activation_capture": False,
              "timing_scope": TIMING_SCOPE, "includes_possible_host_attribute_gaps": True,
              "statistical_confirmation": False, "full_request_result": False, "cupti": False,
              "performance_claim": False}
    reservation, ident, handles, export, protocol = None, None, [], None, None
    create_attempted, production_checked = False, False
    ownership = uuid.uuid4().hex
    lab.OWNER = "ninfer-attention-v-shift-" + ownership
    name = "ninfer-lab-attention-v-shift-" + ownership[:12]
    record["owner"] = lab.OWNER
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.ReleaseMutex.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    with receipt.open("x", encoding="utf-8") as target:
        try:
            record["production_config_sha256_before"] = lab.sha(ROOT / "broker/config.json")
            for path, field in ((pathlib.Path(__file__), "controller_sha256"),
                                (EXPORT, "export_receipt_sha256"), (PROTOCOL, "protocol_sha256")):
                record[field] = lab.sha(path)
            export, protocol = load(EXPORT), load(PROTOCOL)
            resources, expected_receipt = preflight(export, protocol)
            record.update(binary_sha256=export["binary_sha256"], resources=resources,
                          helper_sha256=protocol["helper_sha256"], container_name=name)
            require(len(guard.MUTEXES) == 5, "Lab mutex coverage changed")
            for mutex in guard.MUTEXES:
                handle = kernel.CreateMutexW(None, False, "Local\\" + mutex)
                require(bool(handle), "Cannot create lab mutex: " + mutex)
                if kernel.WaitForSingleObject(handle, 0) not in (0, 128):
                    kernel.CloseHandle(handle)
                    raise RuntimeError("Laboratory mutex busy: " + mutex)
                handles.append(handle)
            record["mutexes"] = guard.MUTEXES
            record["admission"] = lab.admission()
            record["builder_admission"] = guard.builder_idle(lab)
            record["production_before"] = production_snapshot()
            production_checked = True
            reservation = uuid.uuid4().hex
            record["reservation_id"] = reservation
            grant = lab.api("/v1/photo-gpu-reservations", {
                "owner": lab.OWNER, "reservation_id": reservation, "ttl_seconds": 900}, "POST", 90)
            require(grant["state"] == "active" and grant["reservation_id"] == reservation
                    and grant["owner"] == lab.OWNER,
                    "Owned reservation not granted")
            record["owned_admission"] = lab.admission(True)
            create_attempted = True
            ident = lab.docker("create", "--name", name, "--label", "io.codex.ninfer-owner=" + reservation,
                "--label", "io.codex.ninfer-experiment=attention-v-shift", "--gpus", "all",
                "--mount", "type=bind,source=" + lab.WSL_ROOT + ",target=/lab,readonly",
                "--entrypoint", ENTRYPOINT, IMAGE).stdout.strip()
            require(re.fullmatch(r"[0-9a-f]{64}", ident), "Invalid owned container ID")
            record["container_id"] = ident
            item = json.loads(lab.docker("inspect", ident).stdout)[0]
            require(item["Image"] == IMAGE and item["Name"] == "/" + name
                    and item["Config"]["Labels"]["io.codex.ninfer-owner"] == reservation
                    and item["Config"]["Entrypoint"] == [ENTRYPOINT] and not item["Config"].get("Cmd"),
                    "Owned container image/entrypoint/arguments changed")
            require(any(m["Destination"] == "/lab" and not m["RW"] and m["Source"] == lab.WSL_ROOT
                        for m in item["Mounts"]), "Expected read-only /lab bind mount missing")
            forbidden = ("LD_PRELOAD=", "LD_AUDIT=", "CUDA_INJECTION64_PATH=", "CUPTI_")
            require(not any(value.startswith(forbidden) for value in item["Config"].get("Env", [])),
                    "Timing container has profiling or injection environment")
            require(lab.sha(BINARY) == export["binary_sha256"], "Binary changed before launch")
            command = ["wsl.exe", "-d", "Ubuntu-24.04", "--exec", "/usr/bin/docker", "start", "-a", ident]
            started = time.monotonic()
            with output.open("x", encoding="utf-8") as stdout:
                process = subprocess.Popen(command, stdout=stdout, stderr=subprocess.PIPE, text=True,
                                           creationflags=subprocess.CREATE_NO_WINDOW)
                try:
                    deadline = time.monotonic() + 15
                    while process.poll() is None and time.monotonic() < deadline:
                        proof = lab.docker("exec", ident, "sha256sum", "/proc/1/exe", check=False)
                        if proof.returncode == 0:
                            value = proof.stdout.split()
                            require(len(value) == 2 and value[1] == "/proc/1/exe"
                                    and value[0] == export["binary_sha256"], "Running executable hash mismatch")
                            record["running_executable_sha256"] = value[0]
                            break
                        time.sleep(0.05)
                    require(record.get("running_executable_sha256") == export["binary_sha256"],
                            "No full running /proc/1/exe hash proof")
                    _, record["stderr"] = process.communicate(timeout=450)
                finally:
                    if process.poll() is None:
                        process.kill()
                    if "stderr" not in record:
                        _, record["stderr"] = process.communicate(timeout=10)
                    record.update(attach_exit_code=process.returncode, wall_seconds=time.monotonic() - started)
            record["output_sha256"] = lab.sha(output)
            ended = json.loads(lab.docker("inspect", ident).stdout)[0]
            require(not ended["State"]["Running"], "Attention process still running after attach completed")
            record["container_exit_code"] = ended["State"]["ExitCode"]
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
            record["host_events"] = [row.get("event") for row in rows]
            record["host_failures"] = [row for row in rows if row.get("event") in
                                       ("oracle_failure", "attention_v_shift_host_failure")]
            require(record["attach_exit_code"] == record["container_exit_code"] == 0,
                    "Host failed: " + json.dumps(record["host_failures"]))
            record["validation"] = validate(rows, expected_receipt)
            record["after_run_admission"] = matrix.serving_admission(reservation)
            record["status"] = PASS
        except BaseException as exc:
            record.update(status="FAILED", failure=type(exc).__name__ + ": " + str(exc))
        finally:
            try:
                if create_attempted:
                    found = lab.docker("inspect", name, check=False)
                    if found.returncode:
                        require("no such object" in found.stderr.lower() or "no such container" in found.stderr.lower(),
                                "Cannot prove owned container absent: " + found.stderr)
                    else:
                        item = json.loads(found.stdout)[0]
                        require(item["Image"] == IMAGE and item["Name"] == "/" + name
                                and (not ident or item["Id"] == ident)
                                and item["Config"]["Labels"]["io.codex.ninfer-owner"] == reservation
                                and item["Config"]["Labels"]["io.codex.ninfer-experiment"] == "attention-v-shift",
                                "Container ownership mismatch; refusing cleanup")
                        if item["State"]["Running"]:
                            record["owned_container_stopped_by_controller"] = True
                            lab.docker("stop", "--time", "5", item["Id"])
                        elif "container_exit_code" not in record:
                            record["container_exit_code"] = item["State"]["ExitCode"]
                        lab.docker("rm", item["Id"])
                    record["owned_container_removed_or_absent"] = True
                if reservation and (not create_attempted or record.get("owned_container_removed_or_absent")):
                    released = lab.api("/v1/photo-gpu-reservations/" + reservation,
                                       {"owner": lab.OWNER}, "DELETE")
                    require(released["reservation_id"] == reservation and released["state"] == "released",
                            "Owned reservation not released")
                    record["reservation_released"] = True
            except BaseException as exc:
                record.update(status="FAILED", cleanup_failure=type(exc).__name__ + ": " + str(exc))
            try:
                if output.exists():
                    record["output_sha256"] = lab.sha(output)
                    if "host_events" not in record:
                        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
                                if line.strip()]
                        record["host_events"] = [row.get("event") for row in rows]
                        record["host_failures"] = [row for row in rows if row.get("event") in
                                                   ("oracle_failure", "attention_v_shift_host_failure")]
            except BaseException as exc:
                record.update(status="FAILED", output_parse_failure=type(exc).__name__ + ": " + str(exc))
            try:
                if "production_config_sha256_before" in record:
                    require(lab.sha(ROOT / "broker/config.json") == record["production_config_sha256_before"],
                            "Production configuration changed")
                    record["production_config_unchanged"] = True
                if production_checked:
                    record["production_after"] = production_snapshot()
                    require(record["production_after"] == record["production_before"], "Production container changed")
                    record["production_container_unchanged"] = True
                for path, field in ((EXPORT, "export_receipt_sha256"), (PROTOCOL, "protocol_sha256"),
                                    (pathlib.Path(__file__), "controller_sha256")):
                    if field in record:
                        require(lab.sha(path) == record[field], "Immutable screen input changed: " + str(path))
                if export is not None:
                    require(lab.sha(BINARY) == export["binary_sha256"], "Exported binary changed during screen")
                    check_sources(export["generation"])
                if protocol is not None:
                    check_pins(ROOT / "source/tools/lab", protocol["helper_sha256"])
            except BaseException as exc:
                record.update(status="FAILED", preservation_failure=type(exc).__name__ + ": " + str(exc))
            for handle in reversed(handles):
                released = kernel.ReleaseMutex(handle)
                closed = kernel.CloseHandle(handle)
                if not released or not closed:
                    record.update(status="FAILED", mutex_cleanup_failure=True)
            record["finished_at"] = lab.now()
            json.dump(record, target, indent=2)
            target.flush()
            print(json.dumps(record, indent=2), flush=True)
    if record["status"] != PASS:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
