import hashlib
import http.server
import ipaddress
import json
import os
import sqlite3
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
BROKER = os.environ.get("GPU_BROKER_API", "http://127.0.0.1:8096")
PORT = int(os.environ.get("DASH_PORT", "8081"))
BROKER_ROOT = os.environ.get("GPU_BROKER_ROOT", os.path.abspath(os.path.join(ROOT, "..", "gpu-broker")))
TAILSCALE = os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Tailscale", "tailscale.exe")
OWNER_PEERS = (0, frozenset())


def owner_peer(address):
    global OWNER_PEERS
    if time.monotonic() >= OWNER_PEERS[0]:
        peers = set()
        try:
            result = subprocess.run([TAILSCALE, "status", "--json"], capture_output=True, text=True, timeout=3, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            status = json.loads(result.stdout)
            owner = status.get("Self", {}).get("UserID")
            if owner and status.get("BackendState") == "Running":
                for peer in status.get("Peer", {}).values():
                    if peer.get("UserID") == owner:
                        peers.update(ipaddress.ip_address(value) for value in peer.get("TailscaleIPs", []))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        OWNER_PEERS = (time.monotonic() + 30, frozenset(peers))
    return address in OWNER_PEERS[1]


def page_revision():
    with open(os.path.join(ROOT, "index.html"), "rb") as page:
        return hashlib.sha256(page.read()).hexdigest()[:16]


def dashboard_data(offset):
    with open(os.path.join(BROKER_ROOT, "config.json"), encoding="utf-8") as config_file:
        database = json.load(config_file)["database"]
    path = os.path.abspath(os.path.join(BROKER_ROOT, database))
    connection = sqlite3.connect("file:" + urllib.parse.quote(path.replace("\\", "/"), safe="/:") + "?mode=ro", uri=True, timeout=2)
    connection.row_factory = sqlite3.Row
    try:
        now = time.time()
        jobs = connection.execute("SELECT state,COUNT(*) AS total FROM agent_jobs GROUP BY state").fetchall()
        batches = connection.execute("SELECT state,COUNT(*) AS total FROM inference_batches GROUP BY state").fetchall()
        items = connection.execute("SELECT state,COUNT(*) AS total FROM inference_batch_items GROUP BY state").fetchall()
        requests = connection.execute("SELECT COUNT(*) FROM inference_calls").fetchone()[0]
        recent = connection.execute("SELECT started FROM inference_calls WHERE started>=? ORDER BY started", (now - 86400,)).fetchall()
        job_activity = connection.execute("SELECT created FROM agent_jobs WHERE created>=?", (now - 86400,)).fetchall()
        buckets = [{"time": int(now // 3600 - 23 + index) * 3600, "requests": 0, "jobs": 0} for index in range(24)]
        for row in recent:
            index = int(row["started"] // 3600 - now // 3600 + 23)
            if 0 <= index < 24:
                buckets[index]["requests"] += 1
        for row in job_activity:
            index = int(row["created"] // 3600 - now // 3600 + 23)
            if 0 <= index < 24:
                buckets[index]["jobs"] += 1
        durations = connection.execute("SELECT AVG(updated-created) FROM agent_jobs WHERE state IN ('completed','failed','cancelled','preempted','interrupted') AND updated>=created").fetchone()[0]
        speed = connection.execute("SELECT COUNT(*),SUM(output_tokens),SUM(elapsed) FROM (SELECT output_tokens,elapsed FROM inference_calls WHERE status='completed' AND elapsed>0 AND output_tokens>0 ORDER BY started DESC LIMIT 100)").fetchone()
        history = connection.execute("""
            SELECT kind,id,owner,state,summary,created,updated,context_length,detail,total_items,successful_items
            FROM (
                SELECT 'agent' AS kind,id,owner,state,task_summary AS summary,created,updated,context_length,
                    CASE WHEN state IN ('completed','failed','cancelled','preempted','interrupted') THEN COALESCE(error || CHAR(10) || CHAR(10),'') || COALESCE(result,'') ELSE COALESCE(progress,'') END AS detail,
                    NULL AS total_items,NULL AS successful_items FROM agent_jobs
                UNION ALL
                SELECT 'batch' AS kind,b.id,b.owner,b.state,'Batch inference' AS summary,b.created,b.updated,NULL AS context_length,
                    '' AS detail,COUNT(i.item_id) AS total_items,SUM(CASE WHEN i.state='succeeded' THEN 1 ELSE 0 END) AS successful_items
                FROM inference_batches b LEFT JOIN inference_batch_items i ON i.batch_id=b.id GROUP BY b.id
            ) ORDER BY created DESC,id DESC LIMIT 51 OFFSET ?
        """, (offset,)).fetchall()
        entries = []
        for row in history[:50]:
            entry = dict(row)
            entry["detail"] = str(entry["detail"] or "")[:200]
            entries.append(entry)
        token_usage = connection.execute("SELECT result FROM inference_batch_items WHERE result IS NOT NULL").fetchall()
        batch_tokens = 0
        for row in token_usage:
            try:
                batch_tokens += int((json.loads(row["result"]).get("usage") or {}).get("completion_tokens") or 0)
            except (ValueError, TypeError):
                pass
        return {
            "generated": now,
            "totals": {"agents": sum(row["total"] for row in jobs), "batches": sum(row["total"] for row in batches), "requests": requests, "batch_output_tokens": batch_tokens, "average_agent_seconds": durations, "recent_output_tokens_per_second": speed[1] / speed[2] if speed[2] else None, "recent_speed_requests": speed[0]},
            "outcomes": {"agents": {row["state"]: row["total"] for row in jobs}, "batches": {row["state"]: row["total"] for row in batches}, "batch_items": {row["state"]: row["total"] for row in items}},
            "activity": buckets,
            "history": entries,
            "has_more": len(history) > 50,
            "next_offset": offset + len(entries),
        }
    finally:
        connection.close()


def batch_detail(batch_id):
    with open(os.path.join(BROKER_ROOT, "config.json"), encoding="utf-8") as config_file:
        database = json.load(config_file)["database"]
    path = os.path.abspath(os.path.join(BROKER_ROOT, database))
    connection = sqlite3.connect("file:" + urllib.parse.quote(path.replace("\\", "/"), safe="/:") + "?mode=ro", uri=True, timeout=2)
    connection.row_factory = sqlite3.Row
    try:
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(inference_batch_items)").fetchall()}
        reasoning = "reasoning_level" if "reasoning_level" in columns else "NULL AS reasoning_level"
        rows = connection.execute(f"SELECT item_id,prompt,max_tokens,state,result,error,{reasoning} FROM inference_batch_items WHERE batch_id=? ORDER BY rowid", (batch_id,)).fetchall()
        if not rows:
            return None
        items = []
        for row in rows:
            result = json.loads(row["result"]) if row["result"] else None
            error = json.loads(row["error"]) if row["error"] else None
            items.append({"item_id": row["item_id"], "prompt": row["prompt"], "max_tokens": row["max_tokens"], "state": row["state"], "reasoning_level": row["reasoning_level"], "finish_reason": result.get("finish_reason") if result else None, "result": str(result.get("content") or "") if result else None, "error": error})
        return items
    finally:
        connection.close()


def agent_detail(agent_id):
    with open(os.path.join(BROKER_ROOT, "config.json"), encoding="utf-8") as config_file:
        database = json.load(config_file)["database"]
    path = os.path.abspath(os.path.join(BROKER_ROOT, database))
    connection = sqlite3.connect("file:" + urllib.parse.quote(path.replace("\\", "/"), safe="/:") + "?mode=ro", uri=True, timeout=2)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute("SELECT result,error,progress FROM agent_jobs WHERE id=?", (agent_id,)).fetchone()
        if row is None:
            return None
        result = row["result"] or ""
        if len(result) == 12000:
            history_path = os.path.join(BROKER_ROOT, "results", agent_id + ".history.md")
            if os.path.isfile(history_path):
                with open(history_path, encoding="utf-8", errors="replace") as history_file:
                    history = history_file.read()
                lines = history.rsplit("####", 1)[-1].splitlines()[1:] if "####" in history else []
                while lines and (not lines[0].strip() or lines[0].lstrip().startswith(">")):
                    lines.pop(0)
                full_result = "\n".join(line for line in lines[:next((index for index, line in enumerate(lines) if line.lstrip().startswith("> Tokens:")), len(lines))]).strip()
                if len(full_result) > len(result) and full_result.startswith(result):
                    result = full_result
        return {"result": result, "error": row["error"], "progress": row["progress"]}
    finally:
        connection.close()


class Handler(http.server.BaseHTTPRequestHandler):
    def send_body(self, code, value, content_type="application/json; charset=utf-8"):
        data = value.encode("utf-8") if isinstance(value, str) else json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def owner_local(self):
        try:
            client = ipaddress.ip_address(self.client_address[0])
            if client.is_loopback:
                return True
            addresses = {ipaddress.ip_address(item[4][0]) for item in socket.getaddrinfo(socket.gethostname(), None)}
            return client in addresses or owner_peer(client)
        except (ValueError, OSError):
            return False

    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path
        if path in ("/", "/index.html"):
            with open(os.path.join(ROOT, "index.html"), "rb") as page:
                html = page.read().decode("utf-8")
                revision = hashlib.sha256(html.encode("utf-8")).hexdigest()[:16]
                self.send_body(200, html.replace("__PAGE_REVISION__", revision), "text/html; charset=utf-8")
            return
        if path == "/healthz":
            self.send_body(200, {"status": "ok", "page_revision": page_revision()})
            return
        if path == "/dashboard-data":
            if not self.owner_local():
                self.send_body(403, {"error": "История доступна только с этого компьютера"})
                return
            try:
                offset = int(urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query).get("offset", ["0"])[0])
                if offset < 0 or offset > 100000:
                    raise ValueError("Invalid offset")
                self.send_body(200, dashboard_data(offset))
            except (ValueError, sqlite3.Error, OSError, KeyError) as error:
                self.send_body(503, {"error": str(error)[:300]})
            return
        if path.startswith("/dashboard-agent/"):
            if not self.owner_local():
                self.send_body(403, {"error": "Результаты доступны только с этого компьютера"})
                return
            agent_id = path.removeprefix("/dashboard-agent/")
            if len(agent_id) != 32 or not all(character in "0123456789abcdef" for character in agent_id):
                self.send_body(400, {"error": "Invalid agent id"})
                return
            try:
                detail = agent_detail(agent_id)
                self.send_body(200, detail) if detail is not None else self.send_body(404, {"error": "Agent not found"})
            except (ValueError, sqlite3.Error, OSError, KeyError) as error:
                self.send_body(503, {"error": str(error)[:300]})
            return
        if path.startswith("/dashboard-batch/"):
            if not self.owner_local():
                self.send_body(403, {"error": "Результаты доступны только с этого компьютера"})
                return
            batch_id = path.removeprefix("/dashboard-batch/")
            if len(batch_id) != 32 or not all(character in "0123456789abcdef" for character in batch_id):
                self.send_body(400, {"error": "Invalid batch id"})
                return
            try:
                detail = batch_detail(batch_id)
                self.send_body(200, {"items": detail}) if detail is not None else self.send_body(404, {"error": "Batch not found"})
            except (ValueError, sqlite3.Error, OSError, KeyError) as error:
                self.send_body(503, {"error": str(error)[:300]})
            return
        if path.startswith("/batch/"):
            if not self.owner_local():
                self.send_body(403, {"error": "Результаты доступны только с этого компьютера"})
                return
            batch_id = path.removeprefix("/batch/")
            if len(batch_id) != 32 or not all(character in "0123456789abcdef" for character in batch_id):
                self.send_body(400, {"error": "Invalid batch id"})
                return
            try:
                with urllib.request.urlopen(BROKER + "/v1/batches/" + batch_id, timeout=8) as response:
                    self.send_body(response.status, json.load(response))
            except urllib.error.HTTPError as error:
                self.send_body(error.code, {"error": error.read().decode("utf-8", "replace")[:500]})
            except Exception as error:
                self.send_body(503, {"error": str(error)[:300]})
            return
        if path != "/gpu-status":
            self.send_body(404, {"error": "Not found"})
            return
        try:
            with urllib.request.urlopen(BROKER + "/v1/status", timeout=8) as response:
                status = json.load(response)
            gpu = (status.get("capacity_estimate") or [{}])[0]
            self.send_body(200, {
                "runtime": status.get("runtime", {}),
                "models": status.get("models", []),
                "instances": status.get("instances", []),
                "leases": status.get("leases", []),
                "agents": status.get("agents", []),
                "queue": status.get("queue", []),
                "inference": status.get("inference", {}),
                "observation_available": status.get("windows_gpu", {}).get("available", False),
                "free_bytes": gpu.get("available_llm_budget_bytes"),
                "target_bytes": gpu.get("target_llm_budget_bytes"),
                "total_bytes": gpu.get("total_bytes"),
                "used_bytes": gpu.get("driver_used_bytes"),
                "external_gib": round(gpu.get("external_process_bytes", 0) / 1073741824, 1),
                "external_bytes": gpu.get("external_process_bytes"),
                "external_reserve_bytes": gpu.get("external_reserve_bytes"),
                "measurement": gpu.get("measurement", "unknown"),
                "sample_age_seconds": gpu.get("sample_age_seconds"),
                "budget_revision": status.get("budget_revision"),
                "model_state": status.get("model_state"),
                "external_pressure": status.get("external_growth_pressure", {}).get("active", False),
                "runtime_error": status.get("runtime", {}).get("error"),
                "gpu_error": status.get("runtime_gpu_observation_error"),
            })
        except Exception as error:
            self.send_body(503, {"error": str(error)[:300]})

    def do_POST(self):
        path = urllib.parse.urlsplit(self.path).path
        if path not in ("/gpu-unload", "/batch-submit", "/batch-retry"):
            self.send_body(404, {"error": "Not found"})
            return
        origin = self.headers.get("Origin")
        if not self.owner_local() or (origin and urllib.parse.urlsplit(origin).netloc.lower() != self.headers.get("Host", "").lower()):
            self.send_body(403, {"error": "Owner action is available only from this computer"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 131072:
                self.send_body(413, {"error": "Request is too large"})
                return
            body = json.loads(self.rfile.read(length) or b"{}")
            if path == "/batch-submit":
                items = body.get("items")
                if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], dict):
                    self.send_body(422, {"error": "Expected one batch item"})
                    return
                request = urllib.request.Request(BROKER + "/v1/batches", data=json.dumps({"items": items}).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(request, timeout=30) as response:
                    self.send_body(response.status, json.load(response))
                return
            if path == "/batch-retry":
                batch_id = body.get("batch_id")
                item_ids = body.get("item_ids")
                if not isinstance(batch_id, str) or len(batch_id) != 32 or not all(character in "0123456789abcdef" for character in batch_id):
                    self.send_body(400, {"error": "Invalid batch id"})
                    return
                if item_ids is not None and (not isinstance(item_ids, list) or any(not isinstance(item_id, str) for item_id in item_ids)):
                    self.send_body(422, {"error": "item_ids must be a list of strings"})
                    return
                request = urllib.request.Request(BROKER + "/v1/batches/" + batch_id + "/retry", data=json.dumps({"item_ids": item_ids}).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(request, timeout=30) as response:
                    self.send_body(response.status, json.load(response))
                return
            request = urllib.request.Request(BROKER + "/v1/model/unload", data=json.dumps({"force": False, "wait_seconds": int(body.get("wait_seconds", 300))}).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(request, timeout=3600) as response:
                self.send_body(response.status, json.load(response))
        except urllib.error.HTTPError as error:
            self.send_body(error.code, {"error": error.read().decode("utf-8", "replace")[:500]})
        except Exception as error:
            self.send_body(503, {"error": str(error)[:300]})


if __name__ == "__main__":
    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.daemon_threads = True
    server.serve_forever()
