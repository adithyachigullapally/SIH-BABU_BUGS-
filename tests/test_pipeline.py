"""End-to-end check of the HTTP surface, offline.

Deliberately exercises the change-analysis path only, because that tool is
model-free — so this runs with no GPU, no model load and no LLM quota, and
still covers upload -> validate -> route -> tool -> JSON -> PDF. The VLM tools
have their own live check in backend/tools/vlm.py; the brains have
tests/check_providers.py.

Run: venv/Scripts/python.exe -m tests.test_pipeline
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")  # model answers contain non-cp1252 characters

from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app

ROOT = Path(__file__).resolve().parent.parent
LEVIR = ROOT / "sample_data" / "levir_cd"


def _pair(name="test_1.png"):
    return (
        ("image1", (name, (LEVIR / "A" / name).read_bytes(), "image/png")),
        ("image2", (name, (LEVIR / "B" / name).read_bytes(), "image/png")),
    )


def main():
    client = TestClient(app)

    health = client.get("/api/health").json()
    assert health["status"] == "ok", health
    assert health["vlm_weights_present"], "moondream2 weights are missing from weights/"
    print("health:", health)

    # Bi-temporal pair -> change analysis, with a mask and an overlay.
    files = list(_pair())
    response = client.post(
        "/api/analyze",
        data={"query": "What changed between these two dates?"},
        files=files,
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["task_classified"] == "run_change_analysis", result["task_classified"]
    assert result["execution_trace"], result
    assert result["visual_evidence"]["change_mask_url"], result["visual_evidence"]
    assert 0.0 < result["confidence"] <= 1.0
    print(f"analyze: {result['task_classified']} via {result['llm_provider']} "
          f"in {result['elapsed_seconds']}s, confidence {result['confidence']}")
    print("answer :", result["answer"][:200])

    # The overlay must actually be served, not just named.
    assert client.get(result["visual_evidence"]["change_mask_url"]).status_code == 200

    # PDF report for that same job.
    pdf = client.get(result["report_url"])
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF"), pdf.status_code
    print(f"report : {len(pdf.content)} byte PDF at {result['report_url']}")

    # A deliberately mismatched pair must be rejected with a reason, not a crash.
    small = ROOT / "runs" / "_mismatch.png"
    small.parent.mkdir(exist_ok=True)
    with Image.open(LEVIR / "A" / "test_1.png") as img:
        img.resize((256, 256)).save(small)
    bad = client.post(
        "/api/analyze",
        data={"query": "What changed?"},
        files=[
            ("image1", ("a.png", (LEVIR / "A" / "test_1.png").read_bytes(), "image/png")),
            ("image2", ("b.png", small.read_bytes(), "image/png")),
        ],
    )
    small.unlink()
    assert bad.status_code == 400, bad.status_code
    assert "Dimension mismatch" in bad.json()["detail"], bad.json()
    print("reject :", bad.json()["detail"][:120])

    # An unsupported file type must also be a 400, not a 500.
    junk = client.post(
        "/api/analyze",
        data={"query": "What is this?"},
        files=[("image1", ("notes.txt", b"not an image", "text/plain"))],
    )
    assert junk.status_code == 400, junk.status_code

    print("\npipeline: all checks passed")


if __name__ == "__main__":
    main()
