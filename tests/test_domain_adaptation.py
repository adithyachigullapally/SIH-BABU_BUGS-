"""Measures the LoRA domain adaptation instead of asserting it.

Asks the model to name the land cover in each held-out BigEarthNet patch —
patches `training/lora_finetune.py` never trained on — and scores the CORINE
labels it names against the ground-truth label set. Run it twice to get the
before/after pair:

    set SATQUERY_DISABLE_LORA=1 && venv/Scripts/python.exe -m tests.test_domain_adaptation
    venv/Scripts/python.exe -m tests.test_domain_adaptation

The base model scores near zero not because it cannot see the scene but
because it answers in everyday words ("grass and dirt") rather than in the
CORINE vocabulary the benchmark is written in. That gap is the adaptation.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from PIL import Image

from backend.tools.vlm import answer_question
from training.lora_finetune import DATA, holdout

QUESTION = "What land cover types are present in this image?"


def _labels(text):
    """Pull CORINE class names out of a free-text answer."""
    text = text.lower()
    return {label for label in CLASSES if label in text}


def main():
    global CLASSES

    rows = [r for r in holdout() if "land cover present" in r["answer"]]
    if not rows:
        sys.exit("No holdout set. Run: python -m training.prepare_bigearthnet train 300")

    # The label vocabulary comes from the ground truth itself, longest first so
    # "broad-leaved forest" is matched before "forest" would be.
    CLASSES = sorted(
        {part.strip(" .")
         for r in rows
         for part in r["answer"].split(": ", 1)[1].replace(" and ", ", ").split(", ")
         if len(part.strip(" .")) > 3},
        key=len, reverse=True,
    )

    adapted = os.getenv("SATQUERY_DISABLE_LORA") != "1"
    scores = []
    for row in rows:
        with Image.open(DATA / row["image"]) as raw:
            image = raw.convert("RGB")
        predicted = _labels(answer_question(image_path=DATA / row["image"], question=QUESTION)["answer"])
        truth = _labels(row["answer"])
        if not truth:
            continue
        hits = len(predicted & truth)
        precision = hits / len(predicted) if predicted else 0.0
        recall = hits / len(truth)
        scores.append(0.0 if hits == 0 else 2 * precision * recall / (precision + recall))

    mean = sum(scores) / len(scores)
    print(f"{'adapted' if adapted else 'base   '} model | {len(scores)} held-out patches | "
          f"label-set F1 {mean:.3f}")
    return mean


if __name__ == "__main__":
    main()
