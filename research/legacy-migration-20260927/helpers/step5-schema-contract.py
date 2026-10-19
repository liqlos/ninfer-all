import json
import time
from pathlib import Path

import httpx


URL = "http://127.0.0.1:8096"
RESULT = Path(__file__).resolve().parent / "step5-schema-contract.json"


def main():
    result = {"started_at": time.time()}
    with httpx.Client(timeout=180) as client:
        before = client.get(URL + "/v1/status").json()
        if before["leases"] or before["agents"] or before["external_growth_pressure"]["active"]:
            raise RuntimeError("Broker is not free")
        response = client.post(URL + "/v1/leases", json={"owner": "NInfer schema contract", "kind": "agent", "model_name": "qwen3.8-27b", "requested_agents": 1, "context_mode": "fixed", "context_length": 8192, "ttl_seconds": 300})
        response.raise_for_status()
        lease_id = response.json()["lease_id"]
        try:
            body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "Return the status ready and count 3."}], "max_tokens": 128, "temperature": 0, "response_format": {"type": "json_schema", "json_schema": {"name": "status_count", "strict": True, "schema": {"type": "object", "properties": {"status": {"type": "string", "enum": ["ready"]}, "count": {"type": "integer"}}, "required": ["status", "count"], "additionalProperties": False}}}}
            response = client.post(URL + f"/v1/leases/{lease_id}/chat/completions", json=body)
            result["status"] = response.status_code
            result["raw"] = response.text
            if response.status_code == 200:
                choice = response.json()["choices"][0]
                result["content"] = choice["message"].get("content")
                result["finish_reason"] = choice.get("finish_reason")
                result["usage"] = response.json().get("usage")
                result["parsed"] = json.loads(result["content"])
                result["passed"] = result["parsed"] == {"status": "ready", "count": 3}
            else:
                result["passed"] = False
            strict_body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "Call lookup with id A"}], "max_tokens": 64, "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"], "additionalProperties": False}, "strict": True}}], "tool_choice": {"type": "function", "function": {"name": "lookup"}}}
            strict_response = client.post(URL + f"/v1/leases/{lease_id}/chat/completions", json=strict_body)
            result["strict_tool_status"] = strict_response.status_code
            result["strict_tool_body"] = strict_response.json()
            result["passed"] = result["passed"] and strict_response.status_code == 422 and strict_response.json().get("detail", {}).get("code") == "strict_tools_not_supported"
        finally:
            client.delete(URL + f"/v1/leases/{lease_id}").raise_for_status()
        result["final_grants"] = client.get(URL + "/v1/status").json()["context_grants"]
    result["ended_at"] = time.time()
    RESULT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "status": result["status"], "content": result.get("content"), "strict_tool_status": result["strict_tool_status"], "strict_tool_body": result["strict_tool_body"]}))


if __name__ == "__main__":
    main()
