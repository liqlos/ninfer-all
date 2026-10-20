import argparse
import collections
import hashlib
import json
import pathlib
import re
import sqlite3

import profile_long_context as lab

ROOT = pathlib.Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "artifacts/experimental"
Q4_MANGLE = re.compile(r"^_ZN6ninfer3ops4gguf6detail3vec6kernelIL9ggml_type12ELi4ELb[01]ELi5120ELb1ELb0ELi2EEEvNS2_7VecArgsE$")
HASH = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_COMMON_ON = {
    "NINFER_GGUF_Q4_COOPERATIVE_DECODE",
    "NINFER_GGUF_Q6_COOPERATIVE_DECODE",
    "NINFER_GGUF_MMQ_STORE_FUSION",
}
DP4A = "NINFER_GGUF_DP4A_SHORT_CHAINS"
Q6_X4 = "NINFER_GGUF_Q6_MMQ_A_X4"


def digest(path):
    value = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def enabled(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.upper() in {"ON", "OFF", "TRUE", "FALSE", "1", "0"}:
        return value.upper() in {"ON", "TRUE", "1"}
    raise ValueError("Build option is not a boolean value: " + repr(value))


def block_defines(text, object_name):
    lines = text.splitlines()
    matches = []
    for index, line in enumerate(lines):
        if not line.startswith("build ") or ": " not in line:
            continue
        output, rule = line[6:].split(": ", 1)
        if not output.endswith("/" + object_name) or not rule.startswith("CUDA_COMPILER"):
            continue
        block = []
        for next_line in lines[index + 1:]:
            if next_line.startswith("build "):
                break
            if next_line.startswith((" ", "\t")):
                block.append(next_line.strip())
        define_rows = [entry[len("DEFINES = "):] for entry in block if entry.startswith("DEFINES = ")]
        if len(define_rows) != 1:
            raise ValueError("Expected one DEFINES entry for " + object_name)
        matches.append(set(define_rows[0].split()))
    if len(matches) != 1:
        raise ValueError("Expected one build.ninja object block for " + object_name)
    return matches[0]


def stage_ranges(connection, global_pid):
    tables = {row[0] for row in connection.execute("select name from sqlite_master where type='table'")}
    if not {"StringIds", "NVTX_EVENTS"} <= tables:
        raise ValueError("Nsight SQLite lacks StringIds or NVTX_EVENTS")
    nvtx_columns = {row[1] for row in connection.execute("pragma table_info(NVTX_EVENTS)")}
    required = {"start", "end", "eventType", "textId", "globalTid"}
    if not required <= nvtx_columns:
        raise ValueError("NVTX_EVENTS lacks range fields")
    strings = dict(connection.execute("select id,value from StringIds"))
    query = "select start,end,textId,globalTid from NVTX_EVENTS where eventType=59 and end is not null order by start"
    ranges = collections.defaultdict(list)
    for start, end, text_id, thread in connection.execute(query):
        if thread is None or (int(thread) & ~0xFFFFFF) != global_pid:
            continue
        label = strings.get(text_id, "")
        if label:
            ranges[label].append((int(start), int(end)))
    return ranges


def measured_windows(ranges):
    prefills = ranges.get("prefill", [])
    groups = []
    for start, end in prefills:
        if not groups or start - groups[-1][-1][1] > 1_000_000:
            groups.append([])
        groups[-1].append((start, end))
    if len(groups) != 3:
        raise ValueError("Expected startup, 32-token warmup, and one measured prefill group; saw " + str(len(groups)))
    prefill = groups[-1]
    decode = [window for window in ranges.get("decode", []) if window[0] >= prefill[-1][1]]
    if not decode:
        raise ValueError("No decode ranges follow the measured prefill group")
    return prefill, decode


def fully_contained(start, end, windows):
    return any(left <= start and end <= right for left, right in windows)


def summarize(rows, include_graph):
    counts = collections.Counter()
    names = set()
    for row in rows:
        names.add(row["demangled_name"])
        shape = (row["registers_per_thread"], tuple(row["grid"]), tuple(row["block"]))
        if include_graph:
            shape += (row["graph_node_id"] is not None, row["graph_id"] is not None)
        counts[shape] += 1
    result = {"invocation_count": len(rows), "kernel_names": sorted(names), "shapes": []}
    for shape, count in sorted(counts.items(), key=lambda item: repr(item[0])):
        registers, grid, block, *graph = shape
        item = {
            "registers_per_thread": registers,
            "grid": list(grid),
            "block": list(block),
            "count": count,
        }
        if include_graph:
            item["has_graph_node_id"] = graph[0]
            item["has_graph_id"] = graph[1]
        result["shapes"].append(item)
    return result


def verify(profile_path, sqlite_path):
    profile_path = pathlib.Path(profile_path).resolve(strict=True)
    sqlite_path = pathlib.Path(sqlite_path).resolve(strict=True)
    profile = load(profile_path)
    if profile.get("mode") != "profile" or profile.get("status") != "CAPTURED":
        raise ValueError("Runtime profile receipt is not a completed profile capture")
    capture = profile_path.parent / profile["capture"]["file"]
    assert capture.resolve().parent == profile_path.parent
    binding = load(sqlite_path.with_suffix(".export.json"))
    assert binding["capture_sha256"] == profile["capture"]["sha256"]
    assert digest(capture) == binding["capture_after_import_sha256"]
    assert binding["profile_receipt_sha256"] == digest(profile_path)
    assert binding["sqlite_sha256"] == digest(sqlite_path)
    report = capture.with_suffix(".nsys-rep") if capture.suffix == ".qdstrm" else capture
    assert digest(report) == binding["report_sha256"]
    arm = profile.get("arm")
    if arm not in {"w", "q4", "q6", "both"}:
        raise ValueError("Unexpected switch-matrix arm")
    export = profile.get("export")
    if not isinstance(export, dict) or export.get("arm") != arm:
        raise ValueError("Profile receipt is not bound to the matching export manifest")
    files = export.get("files")
    if not isinstance(files, dict) or not HASH.fullmatch(str(files.get("ninfer-serve", ""))):
        raise ValueError("Export manifest lacks the ninfer-serve SHA-256")
    executable_sha = profile.get("running_executable_sha256")
    if executable_sha != files["ninfer-serve"]:
        raise ValueError("Running executable SHA differs from the profile export manifest")
    source_manifest_sha = export.get("source_manifest_sha256")
    if not HASH.fullmatch(str(source_manifest_sha or "")):
        raise ValueError("Export manifest lacks a source manifest SHA-256")
    build_receipt_sha = profile.get("build_receipt_sha256")
    if not HASH.fullmatch(str(build_receipt_sha or "")):
        raise ValueError("Profile receipt lacks the pinned build receipt SHA-256")
    options = profile.get("build_options")
    if not isinstance(options, dict):
        raise ValueError("Profile receipt lacks build options")
    expected_dp4a = arm in {"q4", "both"}
    expected_q6_x4 = arm in {"q6", "both"}
    expected = dict.fromkeys(EXPECTED_COMMON_ON, True)
    expected[DP4A] = expected_dp4a
    expected[Q6_X4] = expected_q6_x4
    for name, value in expected.items():
        if name not in options or enabled(options[name]) != value:
            raise ValueError("Build option binding mismatch for " + name)
    ninja_sha = files.get("build.ninja")
    ninja_path = ARTIFACTS / "build.ninja"
    if not HASH.fullmatch(str(ninja_sha or "")) or digest(ninja_path) != ninja_sha:
        raise ValueError("Current exported build.ninja does not match the profile export manifest")
    ninja = ninja_path.read_text(encoding="utf-8")
    q4_defines = block_defines(ninja, "ggml_vec_q4_k.cu.o")
    q6_defines = block_defines(ninja, "ggml_mmq_q6_k.cu.o")
    q4_binding = "-DNINFER_GGUF_DP4A_SHORT_CHAINS=1" in q4_defines
    q6_binding = "-DNINFER_GGUF_Q6_MMQ_A_X4=1" in q6_defines
    if q4_binding != expected_dp4a or q6_binding != expected_q6_x4:
        raise ValueError("Per-translation-unit compile binding differs from the arm flags")

    samples = profile.get("samples")
    if not isinstance(samples, list) or len(samples) != 1:
        raise ValueError("Profile receipt must contain exactly one measured sample")
    sample = samples[0]
    done = sample.get("request_done")
    request = done.get("request") if isinstance(done, dict) else None
    if not isinstance(request, dict):
        raise ValueError("Measured sample has no request_done request payload")
    request_id = done.get("request_id")
    if isinstance(request_id, bool) or not isinstance(request_id, int) or request_id <= 0:
        raise ValueError("Measured request_done has no positive request ID")
    if request.get("request_id") != request_id:
        raise ValueError("Measured request_done request payload ID is inconsistent")
    if request.get("requested_output_tokens") != 512:
        raise ValueError("Measured request_done is not the frozen 512-token request")
    if sample.get("usage", {}).get("completion_tokens") != 512:
        raise ValueError("Measured streamed usage does not contain 512 output tokens")
    engine_timing = done.get("engine_timing")
    if not isinstance(engine_timing, dict):
        raise ValueError("Measured request_done lacks engine timing range counts")
    expected_prefill_count = engine_timing.get("units", {}).get("prefill")
    expected_decode_count = engine_timing.get("decode", {}).get("rounds")
    if not isinstance(expected_prefill_count, int) or not isinstance(expected_decode_count, int):
        raise ValueError("Measured request_done has incomplete prefill/decode range counts")

    db = sqlite3.connect("file:" + sqlite_path.as_posix() + "?mode=ro", uri=True)
    db.execute("pragma query_only=on")
    tables = {row[0] for row in db.execute("select name from sqlite_master where type='table'")}
    if not {"StringIds", "NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_KERNEL"} <= tables:
        raise ValueError("Nsight SQLite lacks required NVTX or kernel activity tables")
    kernel_columns = {row[1] for row in db.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    kernel_required = {"start", "end", "mangledName", "demangledName", "registersPerThread", "gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ", "graphNodeId", "graphId"}
    if not kernel_required <= kernel_columns:
        raise ValueError("Kernel activity lacks fields needed for graph route proof")
    contexts = db.execute("select distinct globalPid,contextId from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    assert len(contexts) == 1, "Capture must contain one GPU process/context"
    ranges = stage_ranges(db, int(contexts[0][0]))
    prefill, decode = measured_windows(ranges)
    if len(prefill) != expected_prefill_count:
        raise ValueError("Measured prefill NVTX range count differs from request_done engine timing")
    if len(decode) != expected_decode_count:
        raise ValueError("Measured decode NVTX range count differs from request_done engine timing")
    strings = dict(db.execute("select id,value from StringIds"))
    query = "select start,end,mangledName,demangledName,registersPerThread,gridX,gridY,gridZ,blockX,blockY,blockZ,graphNodeId,graphId from CUPTI_ACTIVITY_KIND_KERNEL order by start"
    q4_rows, q6_rows = [], []
    for row in db.execute(query):
        start, end, mangled_id, name_id, registers, gx, gy, gz, bx, by, bz, node, graph = row
        if end is None:
            continue
        mangled = strings.get(mangled_id, "")
        name = strings.get(name_id, "")
        shape = {
            "mangled_name": mangled,
            "demangled_name": name,
            "registers_per_thread": int(registers) if registers is not None else None,
            "grid": [int(gx), int(gy), int(gz)],
            "block": [int(bx), int(by), int(bz)],
            "graph_node_id": int(node) if node is not None else None,
            "graph_id": int(graph) if graph is not None else None,
        }
        if Q4_MANGLE.fullmatch(mangled) and node is not None and graph is not None and fully_contained(int(start), int(end), decode):
            shape["graph_node_id"] = int(node)
            shape["graph_id"] = int(graph)
            q4_rows.append(shape)
        if "mul_mat_q<(ggml_type)14," in name and fully_contained(int(start), int(end), prefill):
            q6_rows.append(shape)
    db.close()
    if not q4_rows:
        raise ValueError("Exact Q4 T4 K5120 cooperative Rows2 graph kernel was absent from measured prefill")
    expected_registers = 96 if expected_dp4a else 99
    registers = sorted({row["registers_per_thread"] for row in q4_rows})
    if registers != [expected_registers]:
        raise ValueError("Q4 T4 K5120 register route mismatch: expected " + str(expected_registers) + ", saw " + repr(registers))
    if not q6_rows:
        raise ValueError("Q6_K MMQ graph kernel was absent from measured prefill")
    return {
        "schema": "ninfer-switch-matrix-nonstartup-route-v1",
        "status": "PASS",
        "performance_claim": False,
        "timing_claim": False,
        "arm": arm,
        "executable_sha256": executable_sha,
        "profile_receipt": {"path": str(profile_path), "sha256": digest(profile_path)},
        "sqlite": {"path": str(sqlite_path), "sha256": digest(sqlite_path)},
        "export": {
            "source_manifest_sha256": source_manifest_sha,
            "ninfer_serve_sha256": executable_sha,
            "build_receipt_sha256": build_receipt_sha,
            "build_ninja_sha256": ninja_sha,
        },
        "compile_binding": {
            "build_options": {name: enabled(options[name]) for name in sorted(expected)},
            "q4_vector_translation_unit_dp4a_enabled": q4_binding,
            "q6_mmq_translation_unit_a_x4_enabled": q6_binding,
        },
        "request": {
            "request_id": request_id,
            "prompt_tokens": sample.get("usage", {}).get("prompt_tokens"),
            "completion_tokens": sample.get("usage", {}).get("completion_tokens"),
            "output_sha256": sample.get("output_sha256"),
        },
        "prefill_selection": {
            "method": "three ordered prefill groups; startup and distinct 32-token warmup excluded; final group cross-checked to request_done engine timing",
            "ranges": [{"start_ns": left, "end_ns": right} for left, right in prefill],
            "expected_range_count": expected_prefill_count,
            "startup_and_warmup_excluded": True,
        },
        "decode_selection": {
            "method": "decode ranges after final measured prefill group, cross-checked to request_done engine timing",
            "ranges": [{"start_ns": left, "end_ns": right} for left, right in decode],
            "expected_range_count": expected_decode_count,
        },
        "q4_t4_k5120": {
            "expected_registers_per_thread": expected_registers,
            **summarize(q4_rows, True),
        },
        "q6_mmq": {
            "ggml_type": 14,
            **summarize(q6_rows, False),
            "a_x4_route_interpretation": "The exported build.ninja translation-unit binding and build receipt identify whether the Q6 A-x4 compile-time body is enabled. Nsight reports the shared inlined MMQ kernel symbol and does not independently identify the A-x4 body or instruction sequence.",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--sqlite")
    parser.add_argument("--output")
    args = parser.parse_args()
    profile_path = pathlib.Path(args.profile).resolve(strict=True)
    profile = load(profile_path)
    capture = profile_path.parent / profile["capture"]["file"]
    assert capture.resolve().parent == profile_path.parent
    sqlite_path = capture.with_suffix(".sqlite")
    if args.sqlite:
        assert pathlib.Path(args.sqlite).resolve() == sqlite_path.resolve()
    binding_path = sqlite_path.with_suffix(".export.json")
    if not binding_path.exists():
        assert digest(capture) == profile["capture"]["sha256"]
        assert not sqlite_path.exists()
        lab.admission()
        report = capture
        if capture.suffix == ".qdstrm":
            report = capture.with_suffix(".nsys-rep")
            assert not report.exists()
            lab.docker("exec", "ninfer-builder",
                       "/opt/nvidia/nsight-compute/2025.4.1/host/linux-desktop-glibc_2_11_3-x64/QdstrmImporter",
                       "--input-file", "/lab/results/switch-matrix/" + capture.name,
                       "--output-file", "/lab/results/switch-matrix/" + report.name, timeout=180)
        lab.docker("exec", "ninfer-builder", "/usr/local/bin/nsys", "export",
                   "--type", "sqlite", "--output", "/lab/results/switch-matrix/" + sqlite_path.name,
                   "/lab/results/switch-matrix/" + report.name, timeout=180)
        with binding_path.open("x", encoding="utf-8") as handle:
            json.dump({"capture_sha256": profile["capture"]["sha256"],
                       "capture_after_import_sha256": digest(capture),
                       "report_sha256": digest(report),
                       "profile_receipt_sha256": digest(profile_path),
                       "sqlite_sha256": digest(sqlite_path)}, handle, indent=2)
    destination = pathlib.Path(args.output) if args.output else profile_path.parent / (profile.get("arm", "unknown") + "-route.json")
    receipt = verify(profile_path, sqlite_path)
    encoded = (json.dumps(receipt, indent=2) + "\n").encode("utf-8")
    with destination.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
    print(json.dumps({"status": receipt["status"], "output": str(destination.resolve()), "sha256": hashlib.sha256(encoded).hexdigest()}))


if __name__ == "__main__":
    main()
