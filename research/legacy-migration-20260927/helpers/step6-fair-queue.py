import asyncio
import json
import time
from pathlib import Path

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


URL = "http://127.0.0.1:8096"
RESULT = Path(__file__).resolve().parent / "step6-fair-queue.json"


def unwrap(response):
    if response.isError:
        raise RuntimeError(str(response.content))
    return json.loads(response.content[0].text)


async def wait_job(session, job_id, target, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        job = unwrap(await session.call_tool("gpu_agent_poll", {"agent_id": job_id, "wait_seconds": 0}))
        if job["state"] in target:
            return job
        await asyncio.sleep(1)
    raise TimeoutError(f"Job {job_id} did not reach {target}")


async def main():
    result = {"started_at": time.time()}
    async with httpx.AsyncClient(timeout=httpx.Timeout(60, connect=10)) as client:
        before = (await client.get(URL + "/v1/status")).json()
        if before["leases"] or before["agents"] or before["external_growth_pressure"]["active"]:
            raise RuntimeError("Broker is not free")
        response = await client.post(URL + "/v1/leases", json={"owner": "NInfer fairness blocker", "kind": "agent", "model_name": "qwen3.8-27b", "requested_agents": 1, "context_mode": "fixed", "context_length": 32768, "ttl_seconds": 300})
        response.raise_for_status()
        blocker = response.json()["lease_id"]
        try:
            async with streamable_http_client(URL + "/mcp/") as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    long = unwrap(await session.call_tool("gpu_agent_start", {"workspace": "C:\\AI\\gpu-broker", "task": "[ask] Reply with exactly: LONG_DONE", "context_length": 122880, "reasoning_level": "off"}))
                    long_id = long["job_id"]
                    await wait_job(session, long_id, {"waiting_capacity"}, 30)
                    short = unwrap(await session.call_tool("gpu_agent_start", {"workspace": "C:\\AI\\gpu-broker", "task": "[ask] Reply with exactly: SHORT_DONE", "context_length": 8192, "reasoning_level": "off"}))
                    short_id = short["job_id"]
                    await asyncio.sleep(7)
                    result["while_blocked"] = {"long": unwrap(await session.call_tool("gpu_agent_poll", {"agent_id": long_id, "wait_seconds": 0}))["state"], "short": unwrap(await session.call_tool("gpu_agent_poll", {"agent_id": short_id, "wait_seconds": 0}))["state"]}
                    result["before_release_grants"] = (await client.get(URL + "/v1/status")).json()["context_grants"]
                    (await client.delete(URL + f"/v1/leases/{blocker}")).raise_for_status()
                    blocker = None
                    result["long"] = await wait_job(session, long_id, {"succeeded", "failed", "cancelled", "interrupted"}, 180)
                    result["short"] = await wait_job(session, short_id, {"succeeded", "failed", "cancelled", "interrupted"}, 180)
        finally:
            if blocker:
                await client.delete(URL + f"/v1/leases/{blocker}")
        result["final_grants"] = (await client.get(URL + "/v1/status")).json()["context_grants"]
    result["passed"] = result["while_blocked"] == {"long": "waiting_capacity", "short": "waiting_capacity"} and result["before_release_grants"]["committed_grant_tokens"] == 32768 and result["long"]["state"] == "succeeded" and result["long"]["result"].strip() == "LONG_DONE" and result["short"]["state"] == "succeeded" and result["short"]["result"].strip() == "SHORT_DONE" and result["final_grants"]["committed_grant_tokens"] == 0
    result["ended_at"] = time.time()
    RESULT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "while_blocked": result["while_blocked"], "long": result["long"]["state"], "short": result["short"]["state"], "final_grants": result["final_grants"]}))


if __name__ == "__main__":
    asyncio.run(main())
