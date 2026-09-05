"""Loads backend/.env and exposes the agent-brain provider table."""
import os
from pathlib import Path

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / "backend" / ".env")

# Every brain below speaks the OpenAI chat-completions dialect, so one client
# covers all three. base_url + model are the only things that differ.
#
# Model IDs are verified against each provider's live /models list, not the
# build report — the report's `llama-3.3-70b-versatile` is gone from Groq and
# `mistral-large-latest` is not in the free tier. If a provider 404s or 403s on
# the model name, re-run tests/check_providers.py and list models again.
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key": os.getenv("GROQ_API_KEY"),
        "model": "openai/gpt-oss-120b",
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "key": os.getenv("MISTRAL_API_KEY"),
        "model": "ministral-8b-latest",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key": os.getenv("GEMINI_API_KEY"),
        "model": "gemini-2.5-flash",
    },
}

PROVIDER_ORDER = [
    p.strip()
    for p in os.getenv("AGENT_PROVIDER_ORDER", "groq,mistral,gemini").split(",")
    if p.strip() in PROVIDERS
]

HF_TOKEN = os.getenv("HF_TOKEN")


def missing_keys():
    return [name for name in PROVIDER_ORDER if not PROVIDERS[name]["key"]]


if __name__ == "__main__":
    for name in PROVIDER_ORDER:
        key = PROVIDERS[name]["key"]
        print(f"{name:8} {'ok' if key else 'MISSING'}  {PROVIDERS[name]['model']}")
    print(f"{'hf':8} {'ok' if HF_TOKEN else 'MISSING'}")
