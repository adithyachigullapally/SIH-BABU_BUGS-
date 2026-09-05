"""Checks the tool-call normalization without touching the network."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.agent.llm_client import normalize


def fake(content, calls):
    tool_calls = [
        SimpleNamespace(id=i, function=SimpleNamespace(name=n, arguments=a))
        for i, n, a in calls
    ] or None
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))]
    )


def test():
    plain = normalize(fake("hello", []))
    assert plain == {"text": "hello", "tool_calls": []}, plain

    one = normalize(fake(None, [("c1", "run_vqa", '{"image_id": "a", "question": "q"}')]))
    assert one["tool_calls"] == [
        {"id": "c1", "name": "run_vqa", "arguments": {"image_id": "a", "question": "q"}}
    ], one

    # Gemini's compat layer sometimes hands back a dict instead of a JSON string.
    dict_args = normalize(fake(None, [("c2", "run_vqa", {"image_id": "b"})]))
    assert dict_args["tool_calls"][0]["arguments"] == {"image_id": "b"}, dict_args

    bad = normalize(fake(None, [("c3", "run_vqa", "{not json")]))
    assert bad["tool_calls"][0]["arguments"] == {"_unparsed": "{not json"}, bad

    print("llm_client normalization ok")


if __name__ == "__main__":
    test()
