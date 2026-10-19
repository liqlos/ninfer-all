import json
import subprocess
import time
from pathlib import Path

import httpx


RESULT = Path(__file__).resolve().parent / "step10-crash-recovery-explicit-error.json"
BROKER = "http://127.0.0.1:8096"
CONTAINER = "gpu-broker-ninfer"


def docker(*args):
    process = subprocess.run(
        ["wsl.exe", "-u", "root", "--", "docker", *args],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return process.stdout.strip()


def inspect():
    return json.loads(docker("inspect", CONTAINER))[0]


def main():
    result = {"broker": BROKER, "container": CONTAINER}
    lease_id = None
    start = time.monotonic()
    with httpx.Client(timeout=httpx.Timeout(90, connect=10)) as client:
        try:
            status = client.get(BROKER + "/v1/status").json()
            result["initial_leases"] = len(status.get("leases") or [])
            result["initial_agents"] = len(status.get("agents") or [])
            if result["initial_leases"] or result["initial_agents"]:
                raise RuntimeError("Refusing crash test while another lease or agent exists")
            lease_response = client.post(
                BROKER + "/v1/leases",
                json={
                    "owner": "NInfer controlled recovery probe",
                    "kind": "agent",
                    "model_name": "qwen3.8-27b",
                    "requested_agents": 1,
                    "context_mode": "fixed",
                    "context_length": 8192,
                    "ttl_seconds": 900,
                },
            )
            lease_response.raise_for_status()
            lease_id = lease_response.json()["lease_id"]
            result["lease_id"] = lease_id
            original = inspect()
            labels = original["Config"]["Labels"]
            if labels.get("io.codex.gpu-broker.managed") != "true":
                raise RuntimeError("Refusing to stop a container without the broker ownership label")
            result["original_container_id"] = original["Id"]
            body = {
                "model": "qwen3.8-27b",
                "messages": [{"role": "user", "content": "Write a numbered list from 1 through 2000. Use one line per number and continue until the output limit."}],
                "max_tokens": 2048,
                "temperature": 0,
                "stream": True,
            }
            interrupted = False
            with client.stream(
                "POST",
                BROKER + f"/v1/leases/{lease_id}/chat/completions",
                json=body,
            ) as response:
                result["stream_status"] = response.status_code
                response.raise_for_status()
                lines = response.iter_lines()
                for line in lines:
                    if line.startswith("data:") and line.strip() != "data: [DONE]":
                        payload = json.loads(line[5:])
                        choices = payload.get("choices") or []
                        content = (choices[0].get("delta") or {}).get("content") if choices else None
                        if content:
                            result["first_content"] = content[:100]
                            result["kill_output"] = docker("kill", CONTAINER)
                            result["kill_seconds"] = round(time.monotonic() - start, 3)
                            break
                if "kill_output" not in result:
                    raise RuntimeError("Stream ended before any generated content; no container was stopped")
                try:
                    for line in lines:
                        if line.startswith("event: error"):
                            result["error_event"] = True
                        if "backend_interrupted" in line:
                            result["backend_interrupted_payload"] = line[:500]
                            interrupted = True
                        if line.strip() == "data: [DONE]":
                            result["stream_done_after_kill"] = True
                except (httpx.HTTPError, httpx.StreamError) as exc:
                    result["stream_error"] = f"{type(exc).__name__}: {exc}"
                    interrupted = True
            result["stream_interrupted"] = interrupted
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                state = client.get(BROKER + "/v1/status").json()
                model = (state.get("models") or [{}])[0]
                container = model.get("container") or {}
                recovery = state.get("backend_recovery") or {}
                result["last_recovery_state"] = recovery
                if model.get("ready_replicas") and container.get("Id") != original["Id"]:
                    result["recovered_container_id"] = container.get("Id")
                    result["recovery_seconds"] = round(time.monotonic() - start, 3)
                    break
                time.sleep(5)
            else:
                raise RuntimeError("NInfer did not recover within 600 seconds")
            completion = client.post(
                BROKER + f"/v1/leases/{lease_id}/chat/completions",
                json={
                    "model": "qwen3.8-27b",
                    "messages": [{"role": "user", "content": "Reply with exactly: recovery works"}],
                    "max_tokens": 32,
                    "temperature": 0,
                    "stream": False,
                },
            )
            result["completion_status"] = completion.status_code
            result["completion_response"] = completion.json()
            completion.raise_for_status()
            result["completion_output"] = completion.json()["choices"][0]["message"]["content"]
            result["completion_ok"] = result["completion_output"].strip() == "recovery works"
        except BaseException as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if lease_id:
                release = client.delete(BROKER + f"/v1/leases/{lease_id}")
                result["release_status"] = release.status_code
            result["total_seconds"] = round(time.monotonic() - start, 3)
            RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({key: value for key, value in result.items() if key != "completion_response"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
