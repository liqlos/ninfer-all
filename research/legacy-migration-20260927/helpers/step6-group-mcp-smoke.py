import asyncio
import json
import time
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


RESULT = Path(__file__).resolve().parent / "step6-group-mcp-smoke.json"


def unwrap(response):
    if response.isError:
        raise RuntimeError(str(response.content))
    return json.loads(response.content[0].text)


async def main():
    result = {}
    async with streamable_http_client("http://127.0.0.1:8096/mcp/") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            before = unwrap(await session.call_tool("gpu_status"))
            result["before_grants"] = before["model"]["context_grants"]
            result["before_agents"] = before["agents"]
            if before["agents"] or before["queue"] or before["external_pressure"]:
                raise RuntimeError("GPU broker is not free for the group probe")
            group = unwrap(await session.call_tool("gpu_agents_start", {
                "workspace": "C:\\AI\\gpu-broker",
                "reasoning_level": "low",
                "tasks": [
                    {"task": "[ask] Reply with exactly: first group lane works."},
                    {"task": "[ask] Reply with exactly: second group lane works."},
                ],
            }))
            result["group"] = group
            active = unwrap(await session.call_tool("gpu_status"))
            result["active_grants"] = active["model"]["context_grants"]
            result["active_agents"] = active["agents"]
            pending = {item["agent_id"] for item in group["agents"]}
            jobs = {}
            deadline = time.monotonic() + 240
            while pending and time.monotonic() < deadline:
                for job_id in tuple(pending):
                    job = unwrap(await session.call_tool("gpu_agent_poll", {"agent_id": job_id, "wait_seconds": 0}))
                    jobs[job_id] = job
                    if job["state"] in ("succeeded", "failed", "cancelled", "preempted", "interrupted"):
                        pending.remove(job_id)
                if pending:
                    await asyncio.sleep(2)
            result["jobs"] = jobs
            result["pending"] = sorted(pending)
            for _ in range(30):
                after = unwrap(await session.call_tool("gpu_status"))
                if not after["agents"] and after["model"]["context_grants"]["committed_grant_tokens"] == 0:
                    break
                await asyncio.sleep(1)
            result["after_grants"] = after["model"]["context_grants"]
            result["after_agents"] = after["agents"]
            result["passed"] = (
                not pending
                and group["context_grant_tokens_per_agent"] == 61440
                and group["committed_context_tokens"] == 122880
                and all(job["state"] == "succeeded" for job in jobs.values())
                and result["after_grants"]["committed_grant_tokens"] == 0
            )
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "group_id": result["group"]["group_id"], "states": {key: value["state"] for key, value in result["jobs"].items()}, "grants": result["active_grants"], "after": result["after_grants"]}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
