import argparse
import ctypes
import hashlib
import http.server
import json
import pathlib
import platform
import re
import shutil
import socket
import statistics
import sys
import time
import urllib.request
import uuid


IMAGE = "sha256:501c92a90c3d85adc616455a6783bb5ae8b82008f469cf8dc3d6da538f75475b"
PORT = 18043
PREFILL_SECONDS = 1.0
CHUNK_SECONDS = 0.005
PARTS = [f"{i:03d}:alpha-Бета-界\n" for i in range(64)]
PAYLOAD = "".join(PARTS)
MUTEXES = [
    "NInferV17RuntimeScreen", "NInferV14RuntimeMeasurement",
    "NInferV15ExportedOperatorSeries", "NInferV14CpuVariantPreparation",
    "NInferJonathanSpecShort20261018",
]
PAIRS = [
    ["windows_published", "container_loopback"],
    ["container_loopback", "windows_published"],
    ["container_loopback", "windows_published"],
    ["windows_published", "container_loopback"],
]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def own_sha():
    return digest(pathlib.Path(__file__).read_bytes())


def clocks():
    values = {"perf_counter_ns": time.perf_counter_ns(), "utc_ns": time.time_ns(),
              "process_ns": time.process_time_ns()}
    if sys.platform == "win32":
        kernel = ctypes.WinDLL("kernel32")
        kernel.GetTickCount64.restype = ctypes.c_uint64
        values["tick_ns"] = kernel.GetTickCount64() * 1000000
    elif sys.platform.startswith("linux"):
        values["monotonic_raw_ns"] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        values["boottime_ns"] = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    return values


def durations(start, end):
    require(start.keys() == end.keys(), "Clock inventory changed")
    return {key: (end[key] - value) / 1e9 for key, value in start.items()}


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def request_body(request_id):
    return {
        "model": "synthetic-sse-boundary",
        "stream": True,
        "request_id": request_id,
        "messages": [{"role": "user", "content": "synthetic-body-" * 1024}],
    }


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(5)
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path != "/health":
            self.send_error(404)
            return
        payload = encoded({"owner": self.server.owner, "source_sha256": self.server.source_sha})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.close_connection = True

    def do_POST(self):
        entered = time.perf_counter_ns()
        if self.server.completed_requests >= 10:
            self.send_error(429)
            return
        size = int(self.headers.get("Content-Length", "0"))
        if self.path != "/v1/chat/completions" or not 0 < size <= 65536:
            self.send_error(400)
            return
        body = self.rfile.read(size)
        clock_start = clocks()
        origin = clock_start["perf_counter_ns"]
        request = json.loads(body)
        request_id = request["request_id"]
        require(body == encoded(request_body(request_id)), "Unexpected synthetic request body")
        self.server.completed_requests += 1
        timing = {
            "origin": "complete HTTP request body received, before JSON validation and response headers",
            "body_read_seconds": (origin - entered) / 1e9,
            "body_bytes": len(body),
            "request_sha256": digest(body),
            "modeled_prefill_seconds": PREFILL_SECONDS,
            "modeled_chunk_interval_seconds": CHUNK_SECONDS,
            "content_write_begin_seconds": [],
            "content_write_end_seconds": [],
        }

        def relative():
            return (time.perf_counter_ns() - origin) / 1e9

        def chunk(payload):
            self.connection.sendall(f"{len(payload):x}\r\n".encode("ascii") + payload + b"\r\n")

        def event(delta, finish=None, extra=None):
            value = {
                "id": "chatcmpl-" + request_id,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            value.update(extra or {})
            chunk(b"data: " + encoded(value) + b"\n\n")

        timing["headers_write_begin_seconds"] = relative()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()
        timing["headers_write_end_seconds"] = relative()
        timing["role_write_begin_seconds"] = relative()
        event({"role": "assistant"})
        timing["role_write_end_seconds"] = relative()
        prefill_start = time.perf_counter_ns()
        timing["prefill_start_seconds"] = (prefill_start - origin) / 1e9
        for i, part in enumerate(PARTS):
            delay = (prefill_start - time.perf_counter_ns()) / 1e9 + PREFILL_SECONDS + i * CHUNK_SECONDS
            if delay > 0:
                time.sleep(delay)
            timing["content_write_begin_seconds"].append(relative())
            event({"content": part})
            timing["content_write_end_seconds"].append(relative())
        timing["synthetic_work_complete_seconds"] = relative()
        timing["clock_durations_seconds"] = durations(clock_start, clocks())
        timing["terminal_prepare_seconds"] = relative()
        event({}, "length", {
            "diagnostic": {
                "synthetic": True,
                "owner": self.server.owner,
                "source_sha256": self.server.source_sha,
                "server_python": self.server.python,
                "server_platform": self.server.platform,
                "timing": timing,
            },
        })
        chunk(b"data: [DONE]\n\n")
        try:
            self.connection.sendall(b"0\r\n")
            self.connection.sendall(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True


def server(args):
    require(own_sha() == args.expected_sha, "Server source pin mismatch")
    with http.server.HTTPServer(("0.0.0.0", 8080), Handler) as httpd:
        httpd.owner = args.owner
        httpd.source_sha = args.expected_sha
        httpd.python = sys.version
        httpd.platform = platform.platform()
        httpd.completed_requests = 0
        httpd.timeout = 1
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            httpd.handle_request()


def client(url, request_id, expected_sha, owner):
    require(own_sha() == expected_sha, "Client source pin mismatch")
    body = encoded(request_body(request_id))
    request = urllib.request.Request(
        url, body, {"Content-Type": "application/json", "Accept": "text/event-stream"})
    clock_start = clocks()
    started_ns = clock_start["perf_counter_ns"]
    raw_parts, text_parts, arrivals = [], [], []
    first_data, done_time, diagnostic, finish_reason = None, None, None, None
    response_id = None
    saw_done = False
    with urllib.request.urlopen(request, timeout=10) as response:
        headers_time = (time.perf_counter_ns() - started_ns) / 1e9
        http_status = response.status
        headers = dict(response.headers)
        for raw in response:
            seen_ns = time.perf_counter_ns()
            raw_parts.append(raw)
            if not raw.startswith(b"data: "):
                continue
            seen = (seen_ns - started_ns) / 1e9
            if first_data is None:
                first_data = seen
            payload = raw[6:].strip()
            if payload == b"[DONE]":
                saw_done = True
                done_time = seen
                break
            value = json.loads(payload)
            response_id = response_id or value.get("id")
            require(value.get("id") == "chatcmpl-" + request_id, "SSE request identity mismatch")
            diagnostic = value.get("diagnostic") or diagnostic
            for choice in value.get("choices", []):
                content = choice.get("delta", {}).get("content", "") or ""
                if content:
                    text_parts.append(content)
                    arrivals.append(seen)
                finish_reason = choice.get("finish_reason") or finish_reason
    clock_end = clocks()
    elapsed = (clock_end["perf_counter_ns"] - started_ns) / 1e9
    output = "".join(text_parts)
    raw_response = b"".join(raw_parts)
    require(http_status == 200 and saw_done and finish_reason == "length", "Incomplete SSE terminal")
    require(text_parts == PARTS and output == PAYLOAD, "Synthetic content differs from known payload")
    require(diagnostic and diagnostic["synthetic"], "Missing server diagnostic")
    require(diagnostic["owner"] == owner and diagnostic["source_sha256"] == expected_sha,
            "Server identity or source pin mismatch")
    timing = diagnostic["timing"]
    require(timing["request_sha256"] == digest(body), "Server received a different request body")
    emits = timing["content_write_begin_seconds"]
    require(len(emits) == len(arrivals), "Server/client content count mismatch")
    return {
        "request_id": request_id, "response_id": response_id,
        "request_sha256": digest(body), "source_sha256": expected_sha,
        "client_python": sys.version, "client_platform": platform.platform(),
        "http_status": http_status, "response_headers": headers, "saw_done": saw_done,
        "finish_reason": finish_reason, "known_payload_equal": True,
        "output_bytes": len(output.encode("utf-8")), "output_sha256": digest(output.encode("utf-8")),
        "response_body_bytes": len(raw_response),
        "response_sha256": digest(raw_response),
        "elapsed_seconds": elapsed, "headers_seconds": headers_time,
        "clock_durations_seconds": durations(clock_start, clock_end),
        "first_data_seconds": first_data, "first_content_seconds": arrivals[0],
        "done_seconds": done_time, "content_arrival_seconds": arrivals,
        "server": diagnostic,
        "boundary_seconds": {
            "elapsed_minus_server_work": elapsed - timing["synthetic_work_complete_seconds"],
            "first_content_lag": arrivals[0] - emits[0],
            "first_data_lag": first_data - timing["role_write_begin_seconds"],
            "headers_lag": headers_time - timing["headers_write_begin_seconds"],
        },
        "content_lag_change_seconds": [
            (arrival - arrivals[0]) - (emit - emits[0])
            for arrival, emit in zip(arrivals, emits)
        ],
    }


def host_resources(lab):
    script = (
        "$m=Get-CimInstance Win32_PerfRawData_PerfOS_Memory;"
        "@{free_commit=([uint64]$m.CommitLimit-[uint64]$m.CommittedBytes)}|ConvertTo-Json -Compress"
    )
    host = json.loads(lab.run([
        "powershell.exe", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden",
        "-Command", script]).stdout)
    host["c_free_bytes"] = shutil.disk_usage("C:/").free
    require(host["free_commit"] >= 24 * 2**30, "Free commit below 24 GiB")
    require(host["c_free_bytes"] >= 20 * 2**30, "C: free space below 20 GiB")
    return host


def inspect(lab, target):
    return json.loads(lab.docker("inspect", target).stdout)[0]


def builder_idle(lab):
    value = inspect(lab, "ninfer-builder")
    require(value["Image"] == IMAGE and value["State"]["Running"], "Builder image/state mismatch")
    require(not value["HostConfig"].get("DeviceRequests"), "Builder has GPU device requests")
    top = lab.docker("top", value["Id"], "-eo", "pid,args").stdout.splitlines()[1:]
    require([line.split(None, 1)[1] for line in top if line.strip()] == ["sleep infinity"],
            "Builder is not idle")
    return {"id": value["Id"], "image": value["Image"], "processes": top}


def owned(lab, target, owner, name):
    value = inspect(lab, target)
    labels = value["Config"].get("Labels") or {}
    require(value["Image"] == IMAGE and value["Name"] == "/" + name
            and labels.get("io.codex.ninfer-owner") == owner
            and labels.get("io.codex.ninfer-experiment") == "sse-boundary",
            "Container ownership/image mismatch; refusing mutation")
    require(not value["HostConfig"].get("DeviceRequests")
            and not value["HostConfig"].get("Devices")
            and value["HostConfig"].get("Runtime") == "runc", "Container is not CPU-only")
    return value


def controller(args):
    require(sys.platform == "win32", "Controller must run in Windows Python")
    sys.dont_write_bytecode = True
    import profile_long_context as lab

    require(re.fullmatch(r"[a-z0-9-]+", args.tag), "Invalid result tag")
    output = lab.ROOT / "results/sse-boundary" / (args.tag + ".json")
    require(not output.exists(), "Immutable result already exists")
    source_sha = own_sha()
    config = lab.ROOT / "broker/config.json"
    owner = uuid.uuid4().hex
    name = "ninfer-lab-sse-" + owner[:12]
    record = {
        "schema": "ninfer-synthetic-sse-boundary-v1", "started_at": lab.now(),
        "diagnostic_only": True, "ninfer_benchmark": False, "performance_claim": False,
        "source_sha256": source_sha, "helper_sha256": lab.sha(pathlib.Path(lab.__file__)),
        "config_sha256_before": lab.sha(config), "image": IMAGE, "owner": owner,
        "container_name": name, "mutexes": MUTEXES, "pair_order": PAIRS,
        "workload": {"prefill_seconds": PREFILL_SECONDS, "content_chunks": len(PARTS),
                     "chunk_interval_seconds": CHUNK_SECONDS, "payload": PAYLOAD,
                     "payload_sha256": digest(PAYLOAD.encode("utf-8")),
                     "request_body_bytes": len(encoded(request_body("0" * 32)))},
        "limits": [
            "Synthetic Python HTTP server; not NInfer, cpp-httplib, engine, broker or GPU timing.",
            "One SSE event per manually framed HTTP chunk; TCP_NODELAY enabled.",
            "No lifecycle log flush or engine queue is executed.",
            "Boundary differences include ingress, parsing, serialization and return path.",
            "All clock subtraction uses local relative durations; no absolute clocks are compared.",
            "Windows versus container also changes Python/OS, not just the forwarding path.",
            "Four fixed exploratory pairs and one warmup per route; no significance claim.",
            "Parser breaks at [DONE], matching frozenbench; it does not validate HTTP zero-chunk receipt.",
        ],
        "warmups": [], "requests": [], "errors": [],
    }
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.ReleaseMutex.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handles, ident = [], None
    create_attempted = False
    try:
        for mutex in MUTEXES:
            handle = kernel.CreateMutexW(None, False, "Local\\" + mutex)
            require(handle, "CreateMutexW failed for " + mutex)
            if kernel.WaitForSingleObject(handle, 0) not in (0, 128):
                kernel.CloseHandle(handle)
                raise RuntimeError("Laboratory mutex busy: " + mutex)
            handles.append(handle)
        require(not output.exists(), "Immutable result already exists")
        record["host_before"] = host_resources(lab)
        record["builder_before"] = builder_idle(lab)
        with socket.socket() as port_check:
            port_check.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            port_check.bind(("127.0.0.1", PORT))
        record["port_free_before"] = True
        create_attempted = True
        created_id = lab.docker(
            "create", "--name", name,
            "--label", "io.codex.ninfer-owner=" + owner,
            "--label", "io.codex.ninfer-experiment=sse-boundary",
            "--runtime", "runc", "--env", "NVIDIA_VISIBLE_DEVICES=void",
            "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "32", "--memory", "256m",
            "--publish", f"127.0.0.1:{PORT}:8080",
            "--mount", "type=bind,source=" + lab.WSL_ROOT + "/source/tools/lab,target=/lab,readonly",
            "--entrypoint", "python3", IMAGE, "-B", "/lab/replay_sse_boundary.py",
            "server", "--expected-sha", source_sha, "--owner", owner).stdout.strip()
        require(len(created_id) == 64 and all(c in "0123456789abcdef" for c in created_id),
                "Docker did not return a full container ID")
        ident = created_id
        value = owned(lab, ident, owner, name)
        require(value["HostConfig"]["PortBindings"] == {
            "8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": str(PORT)}]},
            "Unexpected published port")
        require(len(value["Mounts"]) == 1 and value["Mounts"][0]["Destination"] == "/lab"
                and not value["Mounts"][0]["RW"], "Unexpected diagnostic mount")
        record["container_id"] = ident
        lab.docker("start", ident)
        ready = False
        for _ in range(20):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1) as response:
                    health = json.load(response)
                require(health == {"owner": owner, "source_sha256": source_sha},
                        "Unexpected health identity")
                ready = True
                break
            except (OSError, ValueError):
                time.sleep(0.25)
        require(ready, "Synthetic server did not become ready")

        def measure(route, pair, position):
            require(own_sha() == source_sha, "Diagnostic source changed during replay")
            request_id = uuid.uuid4().hex
            if route == "windows_published":
                result = client(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                request_id, source_sha, owner)
            else:
                result = json.loads(lab.docker(
                    "exec", ident, "python3", "-B", "/lab/replay_sse_boundary.py", "client",
                    "--url", "http://127.0.0.1:8080/v1/chat/completions",
                    "--request-id", request_id, "--expected-sha", source_sha,
                    "--owner", owner, timeout=20).stdout)
            require(result["known_payload_equal"] and result["saw_done"]
                    and result["source_sha256"] == source_sha, "Client validation failed")
            result.update(route=route, pair=pair, position=position)
            return result

        for position, route in enumerate(PAIRS[0]):
            record["warmups"].append(measure(route, None, position))
        for pair, order in enumerate(PAIRS):
            for position, route in enumerate(order):
                record["requests"].append(measure(route, pair, position))
        differences = []
        for pair in range(len(PAIRS)):
            by_route = {row["route"]: row for row in record["requests"] if row["pair"] == pair}
            differences.append({
                "pair": pair,
                "windows_minus_container_seconds": {
                    metric: by_route["windows_published"]["boundary_seconds"][metric]
                    - by_route["container_loopback"]["boundary_seconds"][metric]
                    for metric in by_route["windows_published"]["boundary_seconds"]
                },
            })
        record["paired_differences"] = differences
        record["median_windows_minus_container_seconds"] = {
            metric: statistics.median(row["windows_minus_container_seconds"][metric] for row in differences)
            for metric in differences[0]["windows_minus_container_seconds"]
        }
        record["host_after"] = host_resources(lab)
        record["builder_after"] = builder_idle(lab)
        require(record["builder_before"]["id"] == record["builder_after"]["id"], "Builder changed")
    except BaseException as error:
        record["errors"].append(type(error).__name__ + ": " + str(error))
    finally:
        if create_attempted:
            try:
                value = owned(lab, ident or name, owner, name)
                ident = value["Id"]
                record["container_id"] = ident
                if value["State"]["Running"]:
                    lab.docker("stop", "--time", "2", ident, timeout=15)
                owned(lab, ident, owner, name)
                logs = lab.docker("logs", "--tail", "30", ident, check=False)
                record["container_logs"] = (logs.stdout + logs.stderr)[-6000:]
                lab.docker("rm", ident)
                remaining = lab.docker("ps", "-aq", "--no-trunc",
                                       "--filter", "id=" + ident).stdout.split()
                require(ident not in remaining, "Owned container remains after cleanup")
                record["container_removed"] = True
            except BaseException as error:
                record["errors"].append("cleanup " + type(error).__name__ + ": " + str(error))
                record["container_removed"] = False
        try:
            record["config_sha256_after"] = lab.sha(config)
            record["source_sha256_after"] = own_sha()
            require(record["config_sha256_after"] == record["config_sha256_before"],
                    "Broker config changed during diagnostic")
            require(record["source_sha256_after"] == source_sha, "Diagnostic source changed")
        except BaseException as error:
            record["errors"].append(type(error).__name__ + ": " + str(error))
        record["finished_at"] = lab.now()
        record["status"] = "failed" if record["errors"] else "complete"
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as stream:
                json.dump(record, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
        finally:
            for handle in reversed(handles):
                kernel.ReleaseMutex(handle)
                kernel.CloseHandle(handle)
    print(json.dumps({"status": record["status"], "result": str(output),
                      "errors": record["errors"]}, ensure_ascii=False))
    return 1 if record["errors"] else 0


def main():
    parser = argparse.ArgumentParser(description="CPU-only synthetic SSE boundary replay; not a NInfer benchmark")
    modes = parser.add_subparsers(dest="mode", required=True)
    control = modes.add_parser("controller")
    control.add_argument("--tag", default="replay-r2")
    for mode in ("server", "client"):
        command = modes.add_parser(mode)
        command.add_argument("--expected-sha", required=True)
        command.add_argument("--owner", required=True)
        if mode == "client":
            command.add_argument("--url", required=True)
            command.add_argument("--request-id", required=True)
    args = parser.parse_args()
    if args.mode == "server":
        server(args)
        return 0
    if args.mode == "client":
        print(json.dumps(client(args.url, args.request_id, args.expected_sha, args.owner),
                         ensure_ascii=True))
        return 0
    return controller(args)


if __name__ == "__main__":
    sys.exit(main())
