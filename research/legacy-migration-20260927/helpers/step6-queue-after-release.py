import asyncio
import json
import time
from pathlib import Path

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


RESULT = Path(__file__).resolve().parent / "step6-queue-after-release.json"
BROKER = "http://127.0.0.1:8096"


def unwrap(response):
    if response.isError:
        raise RuntimeError(str(response.content))
    return json.loads(response.content[0].text)


async def main():
    result = {}
    lease_id = None
    async with httpx.AsyncClient(timeout=60) as client:
        async with streamable_http_client(BROKER + "/mcp/") as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                try:
                    before = unwrap(await session.call_tool("gpu_status"))
                    if before["agents"] or before["queue"] or before["external_pressure"]:
                        raise RuntimeError("GPU broker is not free for queue probe")
                    lease_response = await client.post(BROKER + "/v1/leases", json={
                        "owner": "NInfer queue blocker",
                        "kind": "agent",
                        "model_name": "qwen3.8-27b",
                        "requested_agents": 1,
                        "context_mode": "remaining",
                        "ttl_seconds": 300,
                    })
                    lease_response.raise_for_status()
                    lease = lease_response.json()
                    lease_id = lease["lease_id"]
                    result["blocker_grant"] = lease["context_grant_tokens"]
                    job = unwrap(await session.call_tool("gpu_agent_start", {
                        "task": "[ask] Reply with exactly: queue advanced automatically.",
                        "workspace": "C:\\AI\\gpu-broker",
                        "context_length": 8192,
                        "reasoning_level": "low",
                        "idempotency_key": "ninfer-queue-after-release-20260927",
                    }))
                    result["job_id"] = job["job_id"]
                    await asyncio.sleep(3)
                    queued = unwrap(await session.call_tool("gpu_agent_poll", {"agent_id": job["job_id"], "wait_seconds": 0}))
                    result["before_release_state"] = queued["state"]
                    result["before_release_progress"] = queued.get("progress")
                    blocked_status = unwrap(await session.call_tool("gpu_status"))
                    result["before_release_grants"] = blocked_status["model"]["context_grants"]
                    release = await client.delete(BROKER + f"/v1/leases/{lease_id}")
                    result["blocker_release_status"] = release.status_code
                    lease_id = None
                    deadline = time.monotonic() + 180
                    while time.monotonic() < deadline:
                        state = unwrap(await session.call_tool("gpu_agent_poll", {"agent_id": job["job_id"], "wait_seconds": 0}))
                        if state["state"] in ("succeeded", "failed", "cancelled", "preempted", "interrupted"):
                            result["final_job"] = state
                            break
                        await asyncio.sleep(2)
                    else:
                        raise RuntimeError("Queued agent did not finish in 180 seconds")
                    final_status = unwrap(await session.call_tool("gpu_status"))
                    result["final_grants"] = final_status["model"]["context_grants"]
                    result["passed"] = (
                        result["blocker_grant"] == 122880
                        and result["before_release_state"] in ("queued", "acquiring", "waiting_capacity")
                        and result["blocker_release_status"] == 200
                        and result["final_job"]["state"] == "succeeded"
                        and result["final_job"]["result"].strip() == "queue advanced automatically."
                        and result["final_grants"]["committed_grant_tokens"] == 0
                    )
                finally:
                    if lease_id:
                        await client.delete(BROKER + f"/v1/leases/{lease_id}")
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "final_job"}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
