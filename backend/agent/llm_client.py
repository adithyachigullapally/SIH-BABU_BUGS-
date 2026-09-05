"""Agent brain with provider failover.

Groq, Mistral and Gemini all serve an OpenAI-compatible /v1 endpoint, so one
client and one response shape covers all three. Tries providers in
AGENT_PROVIDER_ORDER and falls through on any error (429s especially — free
tiers throttle unpredictably mid-demo).
"""
# pyrefly: ignore [missing-import]
from openai import OpenAI

from backend.config import PROVIDER_ORDER, PROVIDERS


def normalize(response):
    """OpenAI-style response -> {text, tool_calls: [{id, name, arguments}]}."""
    import json

    message = response.choices[0].message
    calls = []
    for call in message.tool_calls or []:
        raw = call.function.arguments
        try:
            args = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except json.JSONDecodeError:
            # Small models occasionally emit malformed JSON; surface it to the
            # controller as an error rather than crashing the request.
            args = {"_unparsed": raw}
        calls.append({"id": call.id, "name": call.function.name, "arguments": args})
    return {"text": message.content, "tool_calls": calls}


def call_llm_with_tools(messages, tools, tool_choice="auto"):
    """Returns (normalized_response, provider_name). Raises if all fail."""
    errors = []
    for name in PROVIDER_ORDER:
        provider = PROVIDERS[name]
        if not provider["key"]:
            errors.append(f"{name}: no API key")
            continue
        try:
            client = OpenAI(api_key=provider["key"], base_url=provider["base_url"])
            response = client.chat.completions.create(
                model=provider["model"],
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )
            return normalize(response), name
        except Exception as exc:  # noqa: BLE001 — any failure means try the next brain
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("All agent providers failed:\n  " + "\n  ".join(errors))
