import json
import time
from pathlib import Path

import httpx


URL = "http://127.0.0.1:8096"
RESULT = Path(__file__).resolve().parent / "step9-agent-isolation.json"


def acquire(client, owner):
    response = client.post(URL + "/v1/leases", json={"owner": owner, "kind": "agent", "model_name": "qwen3.8-27b", "requested_agents": 1, "context_mode": "fixed", "context_length": 8192, "ttl_seconds": 300}, timeout=60)
    response.raise_for_status()
    return response.json()["lease_id"]


def complete(client, lease_id, messages):
    response = client.post(URL + f"/v1/leases/{lease_id}/chat/completions", json={"model": "qwen3.8-27b", "messages": messages, "max_tokens": 128, "temperature": 0, "stream": False}, timeout=180)
    response.raise_for_status()
    return {"text": response.json()["choices"][0]["message"]["content"], "usage": response.json().get("usage")}


def main():
    result = {"started_at": time.time()}
    with httpx.Client() as client:
        before = client.get(URL + "/v1/status", timeout=15).json()
        if before["leases"] or before["agents"] or before["external_growth_pressure"]["active"]:
            raise RuntimeError("Broker is not free")
        a = acquire(client, "Isolation A")
        b = acquire(client, "Isolation B")
        try:
            result["a"] = complete(client, a, [{"role": "user", "content": "Remember the private marker M4P8Q2. Reply with exactly: ACK_A"}])
            result["b"] = complete(client, b, [{"role": "user", "content": "If this conversation's supplied messages contain a private marker from a previous user turn, output it. Otherwise reply with exactly: NONE"}])
            result["a_followup"] = complete(client, a, [{"role": "user", "content": "Remember the private marker M4P8Q2. Reply with exactly: ACK_A"}, {"role": "assistant", "content": result["a"]["text"]}, {"role": "user", "content": "What was the private marker? Reply with the marker only."}])
            result["b_followup"] = complete(client, b, [{"role": "user", "content": "If this conversation's supplied messages contain a private marker from a previous user turn, output it. Otherwise reply with exactly: NONE"}, {"role": "assistant", "content": result["b"]["text"]}, {"role": "user", "content": "What was the private marker in this conversation? If none was supplied, reply with exactly: NONE"}])
        finally:
            for lease_id in (b, a):
                client.delete(URL + f"/v1/leases/{lease_id}", timeout=30).raise_for_status()
        after = client.get(URL + "/v1/status", timeout=15).json()
        result["final_grants"] = after["context_grants"]
        result["passed"] = result["a"]["text"].strip() == "ACK_A" and result["b"]["text"].strip() == "NONE" and result["a_followup"]["text"].strip() == "M4P8Q2" and result["b_followup"]["text"].strip() == "NONE" and result["final_grants"]["committed_grant_tokens"] == 0
    result["ended_at"] = time.time()
    RESULT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "a": result["a"]["text"], "b": result["b"]["text"], "a_followup": result["a_followup"]["text"], "b_followup": result["b_followup"]["text"]}))


if __name__ == "__main__":
    main()
