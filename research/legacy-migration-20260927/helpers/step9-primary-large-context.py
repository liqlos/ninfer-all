import json
import time
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "step9-primary-large-context-document.json"
BROKER = "http://127.0.0.1:8096"
TOKENIZER = "http://127.0.0.1:18032/tokenize"
MODEL = "qwen3.8-27b"
TARGET = 118744
OUTPUT_LIMIT = 4096
MARKERS = ("J7C2", "R8M5", "Z4Q9")


def body_for_words(words):
    first = words // 3
    second = words // 3
    third = words - first - second
    content = (
        "Return the three MARKER values in order, then the total units recorded in the document, separated by commas, with no other text.\n"
        + "MARKER: " + MARKERS[0] + "\n"
        + " x" * first
        + "\nMARKER: " + MARKERS[1] + "\nDocument: Ravo shipped 23 units in 2023 and 19 units in 2024.\n"
        + " x" * second
        + "\nMARKER: " + MARKERS[2] + "\n"
        + " x" * third
    )
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": OUTPUT_LIMIT,
        "temperature": 0,
        "stream": False,
    }


def count_tokens(client, words):
    response = client.post(TOKENIZER, json=body_for_words(words), timeout=60)
    response.raise_for_status()
    return int(response.json()["count"])


def main():
    result = {
        "broker": BROKER,
        "model": MODEL,
        "target_input_tokens": TARGET,
        "output_limit": OUTPUT_LIMIT,
        "markers": MARKERS,
    }
    lease_id = None
    start = time.monotonic()
    with httpx.Client(timeout=httpx.Timeout(1800, connect=10)) as client:
        try:
            base = count_tokens(client, 100)
            probe = count_tokens(client, 1000)
            if probe - base != 900:
                raise RuntimeError(f"Unexpected tokenizer slope: {base}, {probe}")
            words = TARGET - (base - 100)
            body = body_for_words(words)
            measured = count_tokens(client, words)
            if measured != TARGET:
                words += TARGET - measured
                body = body_for_words(words)
                measured = count_tokens(client, words)
            if measured != TARGET:
                raise RuntimeError(f"Input count {measured} differs from {TARGET}")
            result["measured_input_tokens"] = measured
            result["requested_word_count"] = words
            lease_response = client.post(
                BROKER + "/v1/leases",
                json={
                    "owner": "NInfer migration large context gate",
                    "kind": "agent",
                    "model_name": MODEL,
                    "requested_agents": 1,
                    "context_mode": "remaining",
                    "ttl_seconds": 1800,
                },
            )
            result["lease_status"] = lease_response.status_code
            lease_response.raise_for_status()
            lease = lease_response.json()
            lease_id = lease["lease_id"]
            result["lease_id"] = lease_id
            result["context_grant_tokens"] = lease["context_grant_tokens"]
            if result["context_grant_tokens"] < TARGET + OUTPUT_LIMIT:
                raise RuntimeError("Context grant is smaller than input plus output reserve")
            request_start = time.monotonic()
            completion = client.post(
                BROKER + f"/v1/leases/{lease_id}/chat/completions",
                json=body,
            )
            result["request_seconds"] = round(time.monotonic() - request_start, 3)
            result["completion_status"] = completion.status_code
            result["completion_response"] = completion.json()
            completion.raise_for_status()
            answer = completion.json()
            result["output"] = answer["choices"][0]["message"]["content"]
            result["finish_reason"] = answer["choices"][0]["finish_reason"]
            result["usage"] = answer.get("usage")
            result["markers_correct"] = all(marker in result["output"] for marker in MARKERS)
            result["document_answer_correct"] = "42" in result["output"]
            followup_body = {
                "model": MODEL,
                "messages": body["messages"] + [
                    {"role": "assistant", "content": result["output"]},
                    {"role": "user", "content": "Which year had fewer units? Answer with only the year."},
                ],
                "max_tokens": 128,
                "temperature": 0,
                "stream": False,
            }
            followup_start = time.monotonic()
            followup = client.post(
                BROKER + f"/v1/leases/{lease_id}/chat/completions",
                json=followup_body,
            )
            result["followup_seconds"] = round(time.monotonic() - followup_start, 3)
            result["followup_status"] = followup.status_code
            result["followup_response"] = followup.json()
            followup.raise_for_status()
            result["followup_output"] = followup.json()["choices"][0]["message"]["content"]
            result["followup_usage"] = followup.json().get("usage")
            result["followup_correct"] = "2024" in result["followup_output"]
        except BaseException as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if lease_id:
                release = client.delete(BROKER + f"/v1/leases/{lease_id}")
                result["release_status"] = release.status_code
            result["total_seconds"] = round(time.monotonic() - start, 3)
            RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({key: value for key, value in result.items() if key not in ("completion_response", "followup_response")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
