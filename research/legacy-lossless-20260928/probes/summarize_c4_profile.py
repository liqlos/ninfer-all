import argparse
import collections
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "results"


def category(name):
    if name.startswith("_Z9mul_mat_q"):
        return "mmq_q4" if "ggml_type12" in name else "mmq_q6" if "ggml_type14" in name else "mmq_other"
    if name.startswith("_Z24mul_mat_q_stream_k_fixup"):
        return "mmq_fixup"
    if "3vec6kernel" in name:
        return "vector_q4" if "ggml_type12" in name else "vector_q6" if "ggml_type14" in name else "vector_other"
    if "store_plane_kernel" in name:
        return "store_plane"
    if "quantize_matrix_kernel" in name:
        return "quantize_matrix"
    if "quantize_vector_kernel" in name:
        return "quantize_vector"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--tail-ms", type=int)
    parser.add_argument("--offset-ms", type=int)
    parser.add_argument("--window-ms", type=int)
    args = parser.parse_args()
    events_path = ROOT / f"profile-{args.tag}-server-events.jsonl"
    trace_path = ROOT / f"runtime-kernels-{args.tag}.jsonl"
    suffix = f"-tail-{args.tail_ms}" if args.tail_ms else f"-offset-{args.offset_ms}-window-{args.window_ms}" if args.offset_ms is not None else ""
    output = ROOT / f"profile-{args.tag}-summary{suffix}.json"
    if output.exists():
        raise FileExistsError(output)
    events = [json.loads(line) for line in events_path.open(encoding="utf-8")]
    request_record = json.loads((ROOT / f"profile-{args.tag}-requests.json").read_text(encoding="utf-8"))
    requests = request_record.get("requests", [])
    if requests:
        start = min(int(request["host_start_epoch_ns"]) for request in requests)
        end = max(int(request["host_end_epoch_ns"]) for request in requests)
        clock_source = "container_date"
    else:
        start = min(event["timestamp_unix_ms"] for event in events if event.get("event") == "request_start") * 1_000_000
        end = max(event["timestamp_unix_ms"] for event in events if event.get("event") == "request_done") * 1_000_000
        clock_source = "server_events"
    if args.tail_ms:
        if args.tail_ms <= 0 or args.tail_ms * 1_000_000 >= end - start:
            raise ValueError("tail window must be positive and shorter than the request window")
        start = end - args.tail_ms * 1_000_000
    if args.offset_ms is not None:
        if args.tail_ms or args.offset_ms < 0 or not args.window_ms or args.window_ms <= 0 or (args.offset_ms + args.window_ms) * 1_000_000 >= end - start:
            raise ValueError("offset window must be positive and within the request window")
        original_start = start
        start = original_start + args.offset_ms * 1_000_000
        end = start + args.window_ms * 1_000_000
    grouped = collections.defaultdict(lambda: {"count": 0, "duration_ns": 0})
    widths = collections.Counter()
    errors = []
    malformed_lines = 0
    included = 0
    with trace_path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines += 1
                continue
            if item.get("type") == "trace_error":
                errors.append(item)
            if item.get("type") != "kernel" or not start <= item["start_ns"] <= end:
                continue
            included += 1
            kind = category(item["name"])
            grouped[kind]["count"] += 1
            grouped[kind]["duration_ns"] += item["duration_ns"]
            if kind.startswith("mmq_") and kind != "mmq_fixup":
                width = item["name"].split("ELi", 1)[1].split("E", 1)[0]
                widths[f"{kind}_J{width}"] += 1
    total_ns = sum(group["duration_ns"] for group in grouped.values())
    result = {
        "tag": args.tag,
        "tail_ms": args.tail_ms,
        "offset_ms": args.offset_ms,
        "window_ms": args.window_ms,
        "clock_source": clock_source,
        "request_window_ms": (end - start) / 1_000_000,
        "warmup_batch": bool(request_record.get("warmup")),
        "kernel_count": included,
        "sum_kernel_ms": total_ns / 1_000_000,
        "groups": {key: {**value, "duration_ms": value["duration_ns"] / 1_000_000,
                         "share_of_sum_kernel_time": value["duration_ns"] / total_ns}
                   for key, value in sorted(grouped.items())},
        "mmq_tile_counts": dict(sorted(widths.items())),
        "trace_errors": errors,
        "malformed_trace_lines": malformed_lines,
        "source": {"events": str(events_path), "trace": str(trace_path)},
        "limitation": "Request-window kernel-time sums include graph capture and startup work after request start; concurrent kernels can overlap and are not API latency.",
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "groups": {key: round(value["duration_ms"], 2) for key, value in result["groups"].items()},
                      "mmq_tile_counts": result["mmq_tile_counts"], "trace_errors": errors}))


if __name__ == "__main__":
    main()
