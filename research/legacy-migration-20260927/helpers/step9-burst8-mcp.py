import asyncio
import json
import time
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


RESULT = Path(__file__).resolve().parent / "step9-burst8-mcp.json"


def unwrap(response):
    if response.isError:
        raise RuntimeError(str(response.content))
    return json.loads(response.content[0].text)


async def main():
    result = {"jobs": {}, "starts": []}
    async with streamable_http_client("http://127.0.0.1:8096/mcp/") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            before = unwrap(await session.call_tool("gpu_status"))
            if before["agents"] or before["queue"] or before["external_pressure"]:
                raise RuntimeError("GPU broker is not free for burst probe")
            for index in range(8):
                job = unwrap(await session.call_tool("gpu_agent_start", {
                    "workspace": "C:\\AI\\gpu-broker",
                    "task": f"[ask] Reply with exactly: lane {index + 1} complete.",
                    "context_length": 8192,
                    "reasoning_level": "low",
                    "idempotency_key": f"ninfer-burst8-20260927-{index + 1}",
                }))
                result["starts"].append(job)
            result["during"] = unwrap(await session.call_tool("gpu_status"))
            pending = {job["job_id"] for job in result["starts"]}
            deadline = time.monotonic() + 240
            while pending and time.monotonic() < deadline:
                for job_id in tuple(pending):
                    job = unwrap(await session.call_tool("gpu_agent_poll", {"agent_id": job_id, "wait_seconds": 0}))
                    result["jobs"][job_id] = job
                    if job["state"] in ("succeeded", "failed", "cancelled", "preempted", "interrupted"):
                        pending.remove(job_id)
                if pending:
                    await asyncio.sleep(2)
            result["pending"] = sorted(pending)
            for _ in range(30):
                after = unwrap(await session.call_tool("gpu_status"))
                if not after["agents"] and after["model"]["context_grants"]["committed_grant_tokens"] == 0:
                    break
                await asyncio.sleep(1)
            result["after"] = after
    result["passed"] = (
        not result["pending"]
        and all(result["jobs"][job["job_id"]]["state"] == "succeeded" and result["jobs"][job["job_id"]]["result"].strip() == f"lane {index + 1} complete." for index, job in enumerate(result["starts"]))
        and result["after"]["model"]["context_grants"]["committed_grant_tokens"] == 0
    )
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "states": {key: value["state"] for key, value in result["jobs"].items()}, "errors": {key: value.get("error") for key, value in result["jobs"].items() if value.get("error")}, "during_grants": result["during"]["model"]["context_grants"], "after_grants": result["after"]["model"]["context_grants"]}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
