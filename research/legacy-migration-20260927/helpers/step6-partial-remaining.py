import json
from pathlib import Path

import httpx


RESULT = Path(__file__).resolve().parent / "step6-partial-remaining.json"
BROKER = "http://127.0.0.1:8096"


def acquire(client, owner, mode, length=None):
    response = client.post(BROKER + "/v1/leases", json={
        "owner": owner,
        "kind": "agent",
        "model_name": "qwen3.8-27b",
        "requested_agents": 1,
        "context_mode": mode,
        "context_length": length,
        "ttl_seconds": 300,
    })
    response.raise_for_status()
    return response.json()


def main():
    result = {}
    leases = []
    with httpx.Client(timeout=60) as client:
        try:
            before = client.get(BROKER + "/v1/status").json()
            if before.get("leases") or before.get("agents"):
                raise RuntimeError("Broker has active leases or agents")
            first = acquire(client, "NInfer fixed part", "fixed", 32768)
            leases.append(first["lease_id"])
            second = acquire(client, "NInfer remaining part", "remaining")
            leases.append(second["lease_id"])
            result["first_grant"] = first["context_grant_tokens"]
            result["second_grant"] = second["context_grant_tokens"]
            blocked = client.post(BROKER + "/v1/leases", json={
                "owner": "NInfer overcommit rejection",
                "kind": "agent",
                "model_name": "qwen3.8-27b",
                "requested_agents": 1,
                "context_mode": "fixed",
                "context_length": 2048,
                "ttl_seconds": 300,
            })
            result["third_status_when_full"] = blocked.status_code
            result["third_error_when_full"] = blocked.json()
            result["first_release_status"] = client.delete(BROKER + f"/v1/leases/{first['lease_id']}").status_code
            leases.remove(first["lease_id"])
            third = acquire(client, "NInfer freed remainder", "remaining")
            leases.append(third["lease_id"])
            result["third_grant_after_release"] = third["context_grant_tokens"]
            result["during_grants"] = client.get(BROKER + "/v1/status").json()["context_grants"]
        finally:
            result["release_statuses"] = {lease_id: client.delete(BROKER + f"/v1/leases/{lease_id}").status_code for lease_id in leases}
            result["after_grants"] = client.get(BROKER + "/v1/status").json()["context_grants"]
    result["passed"] = (
        result["first_grant"] == 32768
        and result["second_grant"] == 90112
        and result["third_status_when_full"] == 409
        and result["third_grant_after_release"] == 32768
        and result["after_grants"]["committed_grant_tokens"] == 0
    )
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "third_error_when_full"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
