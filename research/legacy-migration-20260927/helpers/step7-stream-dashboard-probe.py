import json
import time
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parent
BROKER = "http://127.0.0.1:8096"


def main():
    result = {}
    lease_id = None
    with httpx.Client(timeout=httpx.Timeout(120, connect=10)) as client:
        try:
            lease = client.post(
                BROKER + "/v1/leases",
                json={
                    "owner": "NInfer dashboard stream probe",
                    "kind": "agent",
                    "model_name": "qwen3.8-27b",
                    "requested_agents": 1,
                    "context_mode": "fixed",
                    "context_length": 4096,
                    "ttl_seconds": 300,
                },
            )
            lease.raise_for_status()
            lease_id = lease.json()["lease_id"]
            result["lease_id"] = lease_id
            start = time.monotonic()
            chunks = []
            finished = False
            with client.stream(
                "POST",
                BROKER + f"/v1/leases/{lease_id}/chat/completions",
                json={
                    "model": "qwen3.8-27b",
                    "messages": [{"role": "user", "content": "Write a numbered list of 40 short items, one per line."}],
                    "max_tokens": 256,
                    "temperature": 0,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            ) as response:
                result["http_status"] = response.status_code
                response.raise_for_status()
                for line in response.iter_lines():
                    if line == "data: [DONE]":
                        finished = True
                    elif line.startswith("data:"):
                        event = json.loads(line[5:])
                        choices = event.get("choices") or []
                        if choices:
                            content = (choices[0].get("delta") or {}).get("content")
                            if content:
                                chunks.append(content)
                        if event.get("usage"):
                            result["usage"] = event["usage"]
            result["stream_done"] = finished
            result["output_length"] = len("".join(chunks))
            result["output_prefix"] = "".join(chunks)[:200]
            result["elapsed_seconds"] = round(time.monotonic() - start, 3)
        finally:
            if lease_id:
                release = client.delete(BROKER + f"/v1/leases/{lease_id}")
                result["release_status"] = release.status_code
            (ROOT / "step7-stream-dashboard-probe.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
