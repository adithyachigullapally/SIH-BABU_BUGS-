"""Live check: every configured brain must return a normalized tool call.

Not a unit test — this hits the network and costs free-tier quota. Run it when
keys change or a provider starts misbehaving mid-build.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# pyrefly: ignore [missing-import]
from openai import OpenAI

from backend.agent.llm_client import normalize
from backend.config import PROVIDER_ORDER, PROVIDERS

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_vqa",
            "description": "Answer a question about a remote-sensing image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_id": {"type": "string"},
                    "question": {"type": "string"},
                },
                "required": ["image_id", "question"],
            },
        },
    }
]
MESSAGES = [
    {"role": "user", "content": "For image_id 'img1', ask what land cover is visible. Use the tool."}
]

ok = True
for name in PROVIDER_ORDER:
    provider = PROVIDERS[name]
    try:
        client = OpenAI(api_key=provider["key"], base_url=provider["base_url"])
        result = normalize(
            client.chat.completions.create(
                model=provider["model"], messages=MESSAGES, tools=TOOLS, tool_choice="auto"
            )
        )
        calls = result["tool_calls"]
        assert calls and calls[0]["name"] == "run_vqa", f"no run_vqa call: {result}"
        assert "image_id" in calls[0]["arguments"], f"bad args: {calls[0]}"
        print(f"{name:8} OK   {calls[0]['arguments']}")
    except Exception as exc:  # noqa: BLE001 — report every provider, don't stop at the first
        ok = False
        print(f"{name:8} FAIL {type(exc).__name__}: {str(exc)[:180]}")

print("\nall providers ok" if ok else "\nat least one provider failed (failover still works if one is up)")
