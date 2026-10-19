import asyncio
import importlib.util
import json
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("mixed_probe", ROOT / "step9-mixed-grants-throughput.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


async def main():
    async with httpx.AsyncClient(timeout=httpx.Timeout(180, connect=10)) as client:
        before = (await client.get(MODULE.BROKER + "/v1/status")).json()
        if before["leases"] or before["agents"] or before["external_growth_pressure"]["active"]:
            raise RuntimeError("Broker is not free")
        response = await client.post(MODULE.BROKER + "/v1/leases", json={"owner": "NInfer solo history", "kind": "agent", "model_name": "qwen3.8-27b", "requested_agents": 1, "context_mode": "fixed", "context_length": 32768, "ttl_seconds": 300})
        response.raise_for_status()
        lease_id = response.json()["lease_id"]
        try:
            source = "\n".join(Path(r"C:\AI\gpu-broker\app.py").read_text(encoding="utf-8").splitlines()[:1100])
            messages = [{"role": "user", "content": "Review the broker code below.\n" + source}, {"role": "assistant", "content": "I will trace the lifecycle."}, {"role": "user", "content": "Which method restores a stopped managed NInfer container under active lease? Answer with the method name and two short details."}]
            result = await MODULE.stream_request(client, lease_id, messages, 256)
        finally:
            (await client.delete(MODULE.BROKER + f"/v1/leases/{lease_id}")).raise_for_status()
        result["final_grants"] = (await client.get(MODULE.BROKER + "/v1/status")).json()["context_grants"]
    result["passed"] = result["stream_done"] and 16000 <= result["usage"]["prompt_tokens"] <= 32000 and "monitor_backend_recovery" in result["output"] and result["final_grants"]["committed_grant_tokens"] == 0
    (ROOT / "step9-solo-history.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("output", "final_grants")}))


if __name__ == "__main__":
    asyncio.run(main())
