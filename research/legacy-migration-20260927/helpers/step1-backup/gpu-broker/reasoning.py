from fastapi import HTTPException


LEVEL_SHARES = {"low": 0.125, "medium": 0.375, "xhigh": 0.75}
LEVELS = ("off", "low", "medium", "high", "xhigh")


def default_output_tokens(configured, lease_context):
    return min(int(configured), max(256, int(lease_context) // 4))


def normalize_reasoning_level(value, default="xhigh"):
    level = default if value is None else value
    if not isinstance(level, str) or level not in LEVELS:
        raise HTTPException(422, {"code": "invalid_reasoning_level", "allowed": list(LEVELS)})
    return "xhigh" if level == "high" else level


def apply_reasoning_level(body, level, api="chat/completions"):
    level = normalize_reasoning_level(level)
    output_tokens = int(body.get("max_completion_tokens", body.get("max_tokens", 1024)))
    upstream_effort = "none" if level == "off" else level
    if api == "responses":
        body["reasoning"] = {"effort": upstream_effort}
        return body
    else:
        kwargs = body.get("chat_template_kwargs") or {}
        if not isinstance(kwargs, dict):
            raise HTTPException(422, {"code": "invalid_chat_template_kwargs"})
        body["chat_template_kwargs"] = {**kwargs, "enable_thinking": level != "off"}
        body["reasoning_effort"] = upstream_effort
    if level == "off":
        body.pop("thinking_token_budget", None)
    else:
        body["thinking_token_budget"] = max(1, min(output_tokens - 1, int(output_tokens * LEVEL_SHARES[level])))
    return body
