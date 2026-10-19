import json
from pathlib import Path

import httpx


URL = "http://127.0.0.1:8096"
RESULT = Path(__file__).resolve().parent / "step5-responses-contract.json"


def main():
    result = {}
    with httpx.Client(timeout=180) as client:
        response = client.post(URL + "/v1/leases", json={"owner": "NInfer Responses contract", "kind": "agent", "model_name": "qwen3.8-27b", "requested_agents": 1, "context_mode": "fixed", "context_length": 8192, "ttl_seconds": 300})
        response.raise_for_status()
        lease_id = response.json()["lease_id"]
        try:
            body = {"model": "qwen3.8-27b", "input": "Reply with exactly: RESPONSES_OK", "max_output_tokens": 128, "reasoning": {"effort": "none"}, "stream": False}
            response = client.post(URL + f"/v1/agents/{lease_id}/responses", json=body)
            result["status"] = response.status_code
            result["raw"] = response.text
            if response.status_code == 200:
                payload = response.json()
                result["response_status"] = payload.get("status")
                result["usage"] = payload.get("usage")
                result["output"] = payload.get("output")
                result["output_text"] = payload.get("output_text")
                content = "".join(part.get("text", "") for item in payload.get("output") or [] for part in item.get("content") or [] if part.get("type") == "output_text")
                result["assembled_text"] = content
                result["passed"] = payload.get("status") == "completed" and content.strip() == "RESPONSES_OK" and (payload.get("usage") or {}).get("input_tokens", 0) > 0
            else:
                result["passed"] = False
        finally:
            client.delete(URL + f"/v1/leases/{lease_id}").raise_for_status()
        result["final_grants"] = client.get(URL + "/v1/status").json()["context_grants"]
    RESULT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "status": result["status"], "response_status": result.get("response_status"), "assembled_text": result.get("assembled_text"), "usage": result.get("usage"), "raw": result["raw"][:300] if not result["passed"] else None}))


if __name__ == "__main__":
    main()
