import asyncio
import json
import time
from pathlib import Path

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


RESULT = Path(__file__).resolve().parent / "step6-group-after-two-leases.json"
BROKER = "http://127.0.0.1:8096"


def unwrap(response):
    if response.isError:
        raise RuntimeError(str(response.content))
    return json.loads(response.content[0].text)


async def main():
    result = {}
    blockers = []
    async with httpx.AsyncClient(timeout=60) as client:
        async with streamable_http_client(BROKER + "/mcp/") as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                try:
                    before = unwrap(await session.call_tool("gpu_status"))
                    if before["agents"] or before["queue"] or before["model"]["context_grants"]["committed_grant_tokens"]:
                        raise RuntimeError("Broker is not free for group-after-two probe")
                    for index in range(2):
                        response = await client.post(BROKER + "/v1/leases", json={
                            "owner": f"NInfer group blocker {index + 1}",
                            "kind": "agent",
                            "model_name": "qwen3.8-27b",
                            "requested_agents": 1,
                            "context_mode": "fixed",
                            "context_length": 8192,
                            "ttl_seconds": 300,
                        })
                        response.raise_for_status()
                        blockers.append(response.json()["lease_id"])
                    group = unwrap(await session.call_tool("gpu_agents_start", {
                        "workspace": "C:\\AI\\gpu-broker",
                        "reasoning_level": "low",
                        "tasks": [
                            {"task": "[ask] Reply with exactly: lane one after blockers."},
                            {"task": "[ask] Reply with exactly: lane two after blockers."},
                            {"task": "[ask] Reply with exactly: lane three after blockers."},
                        ],
                    }))
                    result["group"] = group
                    during = unwrap(await session.call_tool("gpu_status"))
                    result["during_grants"] = during["model"]["context_grants"]
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
                        after_group = unwrap(await session.call_tool("gpu_status"))
                        if after_group["model"]["context_grants"]["committed_grant_tokens"] == 16384:
                            break
                        await asyncio.sleep(1)
                    result["after_group_grants"] = after_group["model"]["context_grants"]
                finally:
                    result["blocker_release_statuses"] = [(await client.delete(BROKER + f"/v1/leases/{lease_id}")).status_code for lease_id in blockers]
                    after = unwrap(await session.call_tool("gpu_status"))
                    result["final_grants"] = after["model"]["context_grants"]
    result["passed"] = (
        not result["pending"]
        and result["group"]["context_grant_tokens_per_agent"] == 35456
        and result["during_grants"]["committed_grant_tokens"] == 122752
        and all(job["state"] == "succeeded" for job in result["jobs"].values())
        and result["after_group_grants"]["committed_grant_tokens"] == 16384
        and result["final_grants"]["committed_grant_tokens"] == 0
    )
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "grant_each": result["group"]["context_grant_tokens_per_agent"], "during": result["during_grants"], "after_group": result["after_group_grants"], "final": result["final_grants"], "states": {key: value["state"] for key, value in result["jobs"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
