import asyncio
import json
import time

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def main():
    async with streamable_http_client("http://127.0.0.1:8096/mcp/") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = [tool.name for tool in (await session.list_tools()).tools]
            response = await session.call_tool("gpu_status")
            if response.isError:
                raise RuntimeError(str(response.content))
            status = json.loads(response.content[0].text)
            result = {"available": "gpu_status" in names and "gpu_agent_start" in names, "backend": status["model"]["backend"], "initial_ready": status["model"]["loaded"], "context": status["model"]["max_context_tokens"], "tool_count": len(names)}
            if not status["model"]["loaded"]:
                started = await session.call_tool("gpu_agent_start", {"workspace": "C:\\AI\\gpu-broker", "task": "[ask] Reply with exactly: BOOT_WARMUP_OK", "context_length": 8192, "reasoning_level": "off"})
                if started.isError:
                    raise RuntimeError(str(started.content))
                job = json.loads(started.content[0].text)
                result["warmup_job_id"] = job["job_id"]
                deadline = time.monotonic() + 600
                while time.monotonic() < deadline:
                    polled = await session.call_tool("gpu_agent_poll", {"agent_id": job["job_id"], "wait_seconds": 2})
                    if polled.isError:
                        raise RuntimeError(str(polled.content))
                    current = json.loads(polled.content[0].text)
                    if current["state"] in ("succeeded", "failed", "cancelled", "preempted", "interrupted"):
                        result["warmup_state"] = current["state"]
                        result["warmup_result"] = current.get("result")
                        break
                if result.get("warmup_state") != "succeeded" or result.get("warmup_result", "").strip() != "BOOT_WARMUP_OK":
                    raise RuntimeError(f"Warmup did not complete: {result}")
            response = await session.call_tool("gpu_status")
            if response.isError:
                raise RuntimeError(str(response.content))
            status = json.loads(response.content[0].text)
            result["ready"] = status["model"]["loaded"]
            result["free_grants"] = status["model"]["context_grants"]["available_guaranteed_tokens"]
            print(json.dumps(result))


if __name__ == "__main__":
    asyncio.run(main())
