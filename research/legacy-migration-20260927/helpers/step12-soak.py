import concurrent.futures
import json
import time
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "step12-soak.json"
BROKER = "http://127.0.0.1:8096"
SCHEDULE = [(0, "short"), (60, "paired"), (120, "cancel"), (180, "short"), (600, "short"), (720, "paired"), (900, "short"), (1080, "paired"), (1260, "cancel"), (1440, "short"), (1620, "paired"), (1790, "short")]


def save(result):
    temporary = RESULT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(RESULT)


def snapshot(client):
    status = client.get(BROKER + "/v1/status", timeout=15).json()
    model = (status.get("models") or [{}])[0]
    capacity = (status.get("capacity_estimate") or [{}])[0]
    return {
        "at": time.time(),
        "backend": status["runtime"]["backend"],
        "ready_replicas": model.get("ready_replicas", 0),
        "container_id": (model.get("container") or {}).get("Id"),
        "leases": len(status.get("leases") or []),
        "agents": len(status.get("agents") or []),
        "active_requests": (status.get("inference") or {}).get("active_requests", 0),
        "grants": status.get("context_grants"),
        "used_bytes": capacity.get("driver_used_bytes"),
        "external_pressure": (status.get("external_growth_pressure") or {}).get("active"),
    }


def lease(client, owner, mode, length=None):
    response = client.post(BROKER + "/v1/leases", json={
        "owner": owner,
        "kind": "agent",
        "model_name": "qwen3.8-27b",
        "requested_agents": 1,
        "context_mode": mode,
        "context_length": length,
        "ttl_seconds": 900,
    }, timeout=660)
    response.raise_for_status()
    return response.json()


def complete(client, lease_id, code):
    start = time.monotonic()
    response = client.post(BROKER + f"/v1/leases/{lease_id}/chat/completions", json={
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": f"Reply with exactly: {code}"}],
        "max_tokens": 48,
        "temperature": 0,
        "stream": False,
    }, timeout=180)
    response.raise_for_status()
    payload = response.json()
    return {"output": payload["choices"][0]["message"]["content"], "usage": payload.get("usage"), "elapsed_seconds": round(time.monotonic() - start, 3), "status": response.status_code}


def short(client, index):
    active = lease(client, f"NInfer soak short {index}", "fixed", 8192)
    try:
        response = complete(client, active["lease_id"], "SOAK_OK")
        return {"grant": active["context_grant_tokens"], "response": response, "correct": response["output"].strip() == "SOAK_OK"}
    finally:
        client.delete(BROKER + f"/v1/leases/{active['lease_id']}", timeout=30).raise_for_status()


def paired(client, index):
    first = lease(client, f"NInfer soak pair fixed {index}", "fixed", 8192)
    second = None
    try:
        second = lease(client, f"NInfer soak pair remaining {index}", "remaining")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            left = executor.submit(complete, client, first["lease_id"], "SOAK_A")
            right = executor.submit(complete, client, second["lease_id"], "SOAK_B")
            outputs = [left.result(), right.result()]
        return {"grants": [first["context_grant_tokens"], second["context_grant_tokens"]], "responses": outputs, "correct": outputs[0]["output"].strip() == "SOAK_A" and outputs[1]["output"].strip() == "SOAK_B"}
    finally:
        if second:
            client.delete(BROKER + f"/v1/leases/{second['lease_id']}", timeout=30).raise_for_status()
        client.delete(BROKER + f"/v1/leases/{first['lease_id']}", timeout=30).raise_for_status()


def cancelled(client, index):
    active = lease(client, f"NInfer soak cancel {index}", "fixed", 8192)
    seen_content = False
    try:
        with client.stream("POST", BROKER + f"/v1/leases/{active['lease_id']}/chat/completions", json={
            "model": "qwen3.8-27b",
            "messages": [{"role": "user", "content": "Write a numbered list from 1 through 2000 and continue until the output limit."}],
            "max_tokens": 2048,
            "temperature": 0,
            "stream": True,
        }, timeout=180) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data:") and line != "data: [DONE]":
                    payload = json.loads(line[5:])
                    choices = payload.get("choices") or []
                    if choices and (choices[0].get("delta") or {}).get("content"):
                        seen_content = True
                        break
        time.sleep(2)
        status = snapshot(client)
        return {"saw_content": seen_content, "after_cancel": status, "correct": seen_content and status["active_requests"] == 0 and status["grants"]["reserved_request_tokens"] == 0}
    finally:
        client.delete(BROKER + f"/v1/leases/{active['lease_id']}", timeout=30).raise_for_status()


def main():
    result = {"started_at": time.time(), "duration_target_seconds": 1800, "events": [], "samples": []}
    start = time.monotonic()
    with httpx.Client(timeout=httpx.Timeout(660, connect=10)) as client:
        try:
            initial = snapshot(client)
            result["initial"] = initial
            if initial["backend"] != "ninfer" or initial["leases"] or initial["agents"] or initial["external_pressure"]:
                raise RuntimeError("Broker is not free and healthy for soak")
            next_sample = 0
            for index, (scheduled, action) in enumerate(SCHEDULE):
                while time.monotonic() - start < scheduled:
                    elapsed = time.monotonic() - start
                    if elapsed >= next_sample:
                        sample = snapshot(client)
                        result["samples"].append(sample)
                        save(result)
                        if sample["external_pressure"]:
                            raise RuntimeError("External GPU pressure appeared during soak")
                        next_sample += 60
                    time.sleep(min(5, max(0.1, scheduled - (time.monotonic() - start))))
                before = snapshot(client)
                event = {"index": index, "scheduled_seconds": scheduled, "actual_seconds": round(time.monotonic() - start, 3), "action": action, "before": before}
                if action == "short":
                    event["result"] = short(client, index)
                elif action == "paired":
                    event["result"] = paired(client, index)
                else:
                    event["result"] = cancelled(client, index)
                event["after"] = snapshot(client)
                result["events"].append(event)
                save(result)
            while time.monotonic() - start < 1800:
                time.sleep(min(5, 1800 - (time.monotonic() - start)))
            result["final"] = snapshot(client)
            result["elapsed_seconds"] = round(time.monotonic() - start, 3)
            result["passed"] = all(event["result"]["correct"] and event["after"]["leases"] == 0 for event in result["events"]) and result["final"]["leases"] == 0 and result["final"]["backend"] == "ninfer" and result["elapsed_seconds"] >= 1800
        except BaseException as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            result["ended_at"] = time.time()
            save(result)
            print(json.dumps({"elapsed_seconds": round(time.monotonic() - start, 3), "events": len(result["events"]), "passed": result.get("passed"), "error": result.get("error")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
