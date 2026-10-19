import asyncio
import json
import time
from pathlib import Path

import httpx


URL = "http://127.0.0.1:8096"
RESULT = Path(__file__).resolve().parent / "step9-burst4-mixed.json"


async def stream(client, lease_id, messages, max_tokens):
    start = time.monotonic()
    pieces = []
    record = {}
    async with client.stream("POST", URL + f"/v1/leases/{lease_id}/chat/completions", json={"model": "qwen3.8-27b", "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}) as response:
        record["status"] = response.status_code
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line == "data: [DONE]":
                record["done"] = True
            elif line.startswith("data:"):
                payload = json.loads(line[5:])
                if payload.get("error"):
                    raise RuntimeError(str(payload["error"]))
                for choice in payload.get("choices") or []:
                    content = (choice.get("delta") or {}).get("content")
                    if content:
                        record.setdefault("ttft_seconds", round(time.monotonic() - start, 3))
                        pieces.append(content)
                if payload.get("usage"):
                    record["usage"] = payload["usage"]
    record["elapsed_seconds"] = round(time.monotonic() - start, 3)
    record["output"] = "".join(pieces)
    record["decode_seconds"] = round(record["elapsed_seconds"] - record["ttft_seconds"], 3) if "ttft_seconds" in record else None
    record["decode_tokens_per_second"] = round(record["usage"]["completion_tokens"] / record["decode_seconds"], 3) if record["decode_seconds"] else None
    return record


async def main():
    result = {"started_at": time.time(), "leases": []}
    async with httpx.AsyncClient(timeout=httpx.Timeout(180, connect=10)) as client:
        before = (await client.get(URL + "/v1/status")).json()
        if before["leases"] or before["agents"] or before["external_growth_pressure"]["active"]:
            raise RuntimeError("Broker is not free")
        try:
            for name, grant in (("long", 32768), ("list", 8192), ("code", 16384), ("short", 8192)):
                response = await client.post(URL + "/v1/leases", json={"owner": f"NInfer burst4 {name}", "kind": "agent", "model_name": "qwen3.8-27b", "requested_agents": 1, "context_mode": "fixed", "context_length": grant, "ttl_seconds": 300})
                response.raise_for_status()
                lease = response.json()
                result["leases"].append({"name": name, "id": lease["lease_id"], "grant": lease["context_grant_tokens"]})
            source = "\n".join(Path(r"C:\AI\gpu-broker\app.py").read_text(encoding="utf-8").splitlines()[:1100])
            jobs = [
                ([{"role": "user", "content": "Review this broker source:\n" + source}, {"role": "assistant", "content": "I will trace the lifecycle."}, {"role": "user", "content": "Which method restores a stopped managed NInfer container under active lease? Reply with only the method name."}], 128),
                ([{"role": "user", "content": "Write a numbered list of 30 short daily actions, one per line."}], 256),
                ([{"role": "user", "content": "Write Python code defining square_even(items), returning squared even integers in input order."}], 128),
                ([{"role": "user", "content": "Reply with exactly: BURST4_OK"}], 64),
            ]
            start = time.monotonic()
            outputs = await asyncio.gather(*(stream(client, lease["id"], messages, max_tokens) for lease, (messages, max_tokens) in zip(result["leases"], jobs)))
            result["wall_seconds"] = round(time.monotonic() - start, 3)
            result["outputs"] = dict(zip((lease["name"] for lease in result["leases"]), outputs))
            result["aggregate_tokens_per_second"] = round(sum(item["usage"]["completion_tokens"] for item in outputs) / result["wall_seconds"], 3)
        finally:
            for lease in reversed(result["leases"]):
                (await client.delete(URL + f"/v1/leases/{lease['id']}")).raise_for_status()
        result["final_grants"] = (await client.get(URL + "/v1/status")).json()["context_grants"]
    result["passed"] = all(item["status"] == 200 and item.get("done") for item in result["outputs"].values()) and "monitor_backend_recovery" in result["outputs"]["long"]["output"] and result["outputs"]["short"]["output"].strip() == "BURST4_OK" and result["final_grants"]["committed_grant_tokens"] == 0
    result["ended_at"] = time.time()
    RESULT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "wall_seconds": result["wall_seconds"], "aggregate_tokens_per_second": result["aggregate_tokens_per_second"], "outputs": {key: {"status": value["status"], "usage": value["usage"], "ttft_seconds": value.get("ttft_seconds"), "decode_tokens_per_second": value["decode_tokens_per_second"]} for key, value in result["outputs"].items()}}))


if __name__ == "__main__":
    asyncio.run(main())
