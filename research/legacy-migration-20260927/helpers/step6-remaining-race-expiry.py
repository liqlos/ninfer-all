import concurrent.futures
import json
import time
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8096"


def status(client):
    response = client.get(URL + "/v1/status", timeout=15)
    response.raise_for_status()
    return response.json()


def acquire(client, owner, mode, ttl=300):
    response = client.post(URL + "/v1/leases", json={"owner": owner, "kind": "agent", "model_name": "qwen3.8-27b", "requested_agents": 1, "context_mode": mode, "ttl_seconds": ttl}, timeout=60)
    return response.status_code, response.json()


def release(client, lease_id):
    response = client.delete(URL + f"/v1/leases/{lease_id}", timeout=30)
    response.raise_for_status()


def main():
    result = {"started_at": time.time()}
    with httpx.Client() as client:
        before = status(client)
        result["before"] = before["context_grants"]
        if before["runtime"]["backend"] != "ninfer" or before["leases"] or before["agents"]:
            raise RuntimeError("Broker is not free for race test")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(acquire, client, "NInfer remaining race A", "remaining")
            second = pool.submit(acquire, client, "NInfer remaining race B", "remaining")
            responses = [first.result(), second.result()]
        result["race"] = responses
        live = [body for code, body in responses if code == 200]
        try:
            result["during"] = status(client)["context_grants"]
        finally:
            for body in live:
                release(client, body["lease_id"])
        result["after_race"] = status(client)["context_grants"]
        code, expiring = acquire(client, "NInfer expiry", "remaining", 15)
        if code != 200:
            raise RuntimeError(f"Expiry lease could not start: {code} {expiring}")
        result["expiry_lease"] = {"lease_id": expiring["lease_id"], "grant": expiring["context_grant_tokens"]}
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            current = status(client)
            if not any(item["id"] == expiring["lease_id"] for item in current["leases"]):
                break
            time.sleep(2)
        result["after_expiry"] = status(client)["context_grants"]
        result["expired"] = not any(item["id"] == expiring["lease_id"] for item in current["leases"])
        result["passed"] = len(live) == 1 and live[0]["context_grant_tokens"] == 122880 and result["during"]["committed_grant_tokens"] == 122880 and result["after_race"]["committed_grant_tokens"] == 0 and result["expired"] and result["after_expiry"]["committed_grant_tokens"] == 0
    result["ended_at"] = time.time()
    (ROOT / "step6-remaining-race-expiry.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "race_codes": [code for code, _ in responses], "expired": result["expired"]}))


if __name__ == "__main__":
    main()
