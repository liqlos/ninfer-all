import hashlib
import json
import time
from pathlib import Path

import httpx


URL = "http://127.0.0.1:8096"
ROOT = Path(__file__).resolve().parent
CASES = [
    {"id": "extract_exclusion", "prompt": "Records: [id=17,duration=90,audit=no] [id=18,duration=300,audit=yes] [id=19,duration=120,audit=no]. Excluding audit=yes, return the ID of the longest-duration record as digits only.", "expected": "19", "max_tokens": 64},
    {"id": "reason_off", "prompt": "Compute 37*18-14. Return only the integer.", "expected": "652", "max_tokens": 128, "reasoning_effort": "none"},
]


def main():
    result = {"started_at": time.time(), "cases": []}
    with httpx.Client(timeout=180) as client:
        status = client.get(URL + "/v1/status").json()
        if status["leases"] or status["agents"] or status["external_growth_pressure"]["active"]:
            raise RuntimeError("Broker is not free")
        response = client.post(URL + "/v1/leases", json={"owner": "NInfer two-case recheck", "kind": "agent", "model_name": "qwen3.8-27b", "requested_agents": 1, "context_mode": "fixed", "context_length": 8192, "ttl_seconds": 300})
        response.raise_for_status()
        lease_id = response.json()["lease_id"]
        try:
            for case in CASES:
                for variant, extra in (("matched", {}), ("zero_penalty", {"presence_penalty": 0}), ("reasoning_medium", {"reasoning_effort": "medium", "max_tokens": 512})):
                    body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": case["prompt"]}], "max_tokens": case["max_tokens"], "temperature": 0, "top_p": 1, "stream": False, **extra}
                    if case.get("reasoning_effort") and variant != "reasoning_medium":
                        body["reasoning_effort"] = case["reasoning_effort"]
                    response = client.post(URL + f"/v1/leases/{lease_id}/chat/completions", json=body)
                    response.raise_for_status()
                    payload = response.json()
                    choice = payload["choices"][0]
                    result["cases"].append({"id": case["id"], "variant": variant, "prompt_sha256": hashlib.sha256(case["prompt"].encode()).hexdigest(), "request": body, "expected": case["expected"], "actual": choice["message"].get("content"), "finish_reason": choice["finish_reason"], "usage": payload.get("usage"), "passed": choice["message"].get("content", "").strip() == case["expected"]})
        finally:
            client.delete(URL + f"/v1/leases/{lease_id}").raise_for_status()
        result["final_grants"] = client.get(URL + "/v1/status").json()["context_grants"]
    result["ended_at"] = time.time()
    (ROOT / "step8-two-case-recheck.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps([{"id": item["id"], "variant": item["variant"], "actual": item["actual"], "passed": item["passed"]} for item in result["cases"]]))


if __name__ == "__main__":
    main()
