"""The build report's Section 13 test matrix, end to end over the real API.

Every mandatory requirement gets one row: the query goes in as a judge would
send it, and the check is that the agent picked the right tool and produced
usable evidence. This one costs GPU time and free-tier LLM quota — it loads
Moondream2 and calls a live brain per row — so it is a deliberate pre-demo
check, not something to run in a loop.

Run: venv/Scripts/python.exe -m tests.test_matrix
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")  # model answers contain non-cp1252 characters

import json

from fastapi.testclient import TestClient

from backend.main import app

ROOT = Path(__file__).resolve().parent.parent
LEVIR = ROOT / "sample_data" / "levir_cd"
BEN = ROOT / "sample_data" / "bigearthnet"


def _file(path, field):
    return (field, (Path(path).name, Path(path).read_bytes(), "application/octet-stream"))


def _rows():
    optical = LEVIR / "A" / "test_1.png"
    later = LEVIR / "B" / "test_1.png"

    rows = [
        ("Single-image VQA (mandatory)",
         "What kind of land cover is visible in this image?", [optical], "run_vqa", {}),
        ("Second single-image task - captioning",
         "Describe this scene.", [optical], "run_caption", {}),
        ("Second single-image task - grounding",
         "Highlight the buildings in this image.", [optical], "run_grounding", {}),
        ("Land-cover measurement (single image, quantitative)",
         "Describe the scene and tell me how much area of trees is present.",
         [optical], "run_land_cover", {"resolution_m": "0.5"}),
        ("Multi-image change analysis (mandatory)",
         "What changed between these two dates?", [optical, later],
         "run_change_analysis", {}),
    ]

    manifest = BEN / "manifest.json"
    if manifest.is_file():
        patch = json.loads(manifest.read_text(encoding="utf-8"))[0]
        rows.append((
            "Cross-modal optical-SAR analysis (mandatory)",
            "Using both the optical and radar views, how much of this scene is water?",
            [BEN / patch["s2"], BEN / patch["s1"]],
            "run_fusion_analysis",
            {"image1_modality": "optical", "image2_modality": "sar"},
        ))
    return rows


def main():
    client = TestClient(app)
    failures = []

    for label, query, images, expected_tool, extra in _rows():
        files = [_file(images[0], "image1")]
        if len(images) > 1:
            files.append(_file(images[1], "image2"))
        response = client.post(
            "/api/analyze", data={"query": query, **extra}, files=files
        )
        if response.status_code != 200:
            failures.append(f"{label}: HTTP {response.status_code} {response.text[:160]}")
            print(f"FAIL  {label}\n      HTTP {response.status_code}")
            continue

        result = response.json()
        chosen = result["task_classified"]
        ok = chosen == expected_tool
        pdf = client.get(result["report_url"])
        if not (pdf.status_code == 200 and pdf.content.startswith(b"%PDF")):
            ok = False
            failures.append(f"{label}: report did not render")
        if not ok and chosen != expected_tool:
            failures.append(f"{label}: expected {expected_tool}, agent chose {chosen}")

        print(f"{'ok  ' if ok else 'FAIL'}  {label}")
        print(f"      tool={chosen} brain={result['llm_provider']} "
              f"conf={result['confidence']} {result['elapsed_seconds']}s")
        print(f"      {result['answer'][:150]}")

    # Input validation: a mismatched pair must be refused with a reason.
    bad = client.post(
        "/api/analyze",
        data={"query": "What changed?"},
        files=[_file(LEVIR / "A" / "test_1.png", "image1"),
               _file(BEN / "s2" / next(p.name for p in (BEN / "s2").glob("*.tif")), "image2")]
        if (BEN / "s2").is_dir() else [_file(LEVIR / "A" / "test_1.png", "image1")],
    )
    if (BEN / "s2").is_dir():
        ok = bad.status_code == 400
        print(f"{'ok  ' if ok else 'FAIL'}  Input validation - mismatched pair rejected")
        if not ok:
            failures.append(f"mismatched pair returned {bad.status_code}, expected 400")

    adapter = ROOT / "backend" / "models" / "final_adapter"
    print(f"{'ok  ' if adapter.is_dir() else 'todo'}  RS domain adaptation - LoRA adapter present")

    print()
    if failures:
        print(f"{len(failures)} failure(s):")
        for line in failures:
            print("  -", line)
        sys.exit(1)
    print("test matrix: every row passed")


if __name__ == "__main__":
    main()
