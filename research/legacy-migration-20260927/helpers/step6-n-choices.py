import json
from pathlib import Path

import httpx


URL = "http://127.0.0.1:8096"
RESULT = Path(__file__).resolve().parent / "step6-n-choices.json"


def main():
    result = {}
    with httpx.Client(timeout=180) as client:
        response = client.post(URL + "/v1/leases", json={"owner": "NInfer multi-choice gate", "kind": "agent", "model_name": "qwen3.8-27b", "requested_agents": 1, "context_mode": "fixed", "context_length": 4096, "ttl_seconds": 300})
        response.raise_for_status()
        lease_id = response.json()["lease_id"]
        try:
            body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "Reply with one short word."}], "max_tokens": 32, "n": 2, "temperature": 0, "stream": False}
            response = client.post(URL + f"/v1/leases/{lease_id}/chat/completions", json=body)
            result["status"] = response.status_code
            result["body"] = response.json()
            result["during"] = client.get(URL + "/v1/status").json()["context_grants"]
        finally:
            client.delete(URL + f"/v1/leases/{lease_id}").raise_for_status()
        result["after"] = client.get(URL + "/v1/status").json()["context_grants"]
    result["passed"] = result["status"] == 422 and result["body"].get("detail", {}).get("code") == "n_not_supported" and result["during"]["committed_grant_tokens"] == 4096 and result["after"]["committed_grant_tokens"] == 0
    RESULT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "status": result["status"], "choice_count": len(result["body"].get("choices") or []), "body": result["body"] if result["status"] != 200 else None, "during": result["during"], "after": result["after"]}))


if __name__ == "__main__":
    main()
