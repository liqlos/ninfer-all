import asyncio
import json
import time
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parent
BROKER = "http://127.0.0.1:8096"


async def stream_request(client, lease_id, messages, max_tokens):
    result = {}
    start = time.monotonic()
    pieces = []
    async with client.stream(
        "POST",
        BROKER + f"/v1/leases/{lease_id}/chat/completions",
        json={
            "model": "qwen3.8-27b",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as response:
        result["http_status"] = response.status_code
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line == "data: [DONE]":
                result["stream_done"] = True
            elif line.startswith("data:"):
                event = json.loads(line[5:])
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                choices = event.get("choices") or []
                if choices:
                    content = (choices[0].get("delta") or {}).get("content")
                    if content:
                        if "ttft_seconds" not in result:
                            result["ttft_seconds"] = round(time.monotonic() - start, 3)
                        pieces.append(content)
                    if choices[0].get("finish_reason"):
                        result["finish_reason"] = choices[0]["finish_reason"]
                if event.get("usage"):
                    result["usage"] = event["usage"]
    result["elapsed_seconds"] = round(time.monotonic() - start, 3)
    result["decode_seconds"] = round(result["elapsed_seconds"] - result["ttft_seconds"], 3) if "ttft_seconds" in result else None
    result["output"] = "".join(pieces)
    result["decode_tokens_per_second"] = round(result["usage"]["completion_tokens"] / result["decode_seconds"], 3) if result["decode_seconds"] else None
    result["end_to_end_tokens_per_second"] = round(result["usage"]["completion_tokens"] / result["elapsed_seconds"], 3)
    return result


async def main():
    result = {}
    leases = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(180, connect=10)) as client:
        try:
            status = (await client.get(BROKER + "/v1/status")).json()
            if status.get("leases") or status.get("agents") or (status.get("external_growth_pressure") or {}).get("active"):
                raise RuntimeError("GPU broker is not free for mixed grants")
            for label, length in (("long", 32768), ("short", 8192)):
                response = await client.post(BROKER + "/v1/leases", json={
                    "owner": f"NInfer mixed grant {label}",
                    "kind": "agent",
                    "model_name": "qwen3.8-27b",
                    "requested_agents": 1,
                    "context_mode": "fixed",
                    "context_length": length,
                    "ttl_seconds": 300,
                })
                response.raise_for_status()
                lease = response.json()
                leases.append(lease["lease_id"])
                result[f"{label}_grant"] = lease["context_grant_tokens"]
            source_lines = (Path(r"C:\AI\gpu-broker\app.py").read_text(encoding="utf-8")).splitlines()
            long_messages = [
                {"role": "user", "content": "Review the broker code below.\n" + "\n".join(source_lines[:1100])},
                {"role": "assistant", "content": "I will trace the lifecycle."},
                {"role": "user", "content": "Which method restores a stopped managed NInfer container under active lease? Answer with the method name and two short details."},
            ]
            short_messages = [{"role": "user", "content": "Write a numbered list of 40 short daily actions, one per line."}]
            token_body = {"model": "qwen3.8-27b", "messages": long_messages, "max_tokens": 256, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
            count = await client.post("http://127.0.0.1:18032/tokenize", json=token_body)
            count.raise_for_status()
            result["measured_long_input_tokens"] = count.json()["count"]
            start = time.monotonic()
            long_result, short_result = await asyncio.gather(
                stream_request(client, leases[0], long_messages, 256),
                stream_request(client, leases[1], short_messages, 256),
            )
            result["combined_wall_seconds"] = round(time.monotonic() - start, 3)
            result["long"] = long_result
            result["short"] = short_result
            result["aggregate_tokens_per_second"] = round((long_result["usage"]["completion_tokens"] + short_result["usage"]["completion_tokens"]) / result["combined_wall_seconds"], 3)
        finally:
            result["release_statuses"] = [(await client.delete(BROKER + f"/v1/leases/{lease_id}")).status_code for lease_id in leases]
            result["final_grants"] = (await client.get(BROKER + "/v1/status")).json()["context_grants"]
    result["passed"] = (
        16000 <= result["measured_long_input_tokens"] <= 32000
        and result["long_grant"] == 32768
        and result["short_grant"] == 8192
        and result["long"]["stream_done"]
        and result["short"]["stream_done"]
        and "monitor_backend_recovery" in result["long"]["output"]
        and result["final_grants"]["committed_grant_tokens"] == 0
    )
    (ROOT / "step9-mixed-grants-throughput.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("long", "short")}, ensure_ascii=False))
    print(json.dumps({"long": {key: value for key, value in result["long"].items() if key != "output"}, "short": {key: value for key, value in result["short"].items() if key != "output"}}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
