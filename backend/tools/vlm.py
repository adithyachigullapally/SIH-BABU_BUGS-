"""Moondream2 wrapper: VQA, captioning and grounding from one loaded model.

ponytail: one module, not the report's separate vqa_caption.py + grounding.py.
All three are the same 4GB model held in the same process — splitting them
across files would mean two modules racing to load the same singleton. Moondream
does pointing natively, so GroundingDINO is not downloaded at all; add it only
if Moondream's boxes prove unreliable on satellite imagery.
"""
import os
import threading
from pathlib import Path

from PIL import Image

WEIGHTS = Path(__file__).resolve().parents[2] / "weights" / "moondream2"
ADAPTER = Path(__file__).resolve().parents[2] / "backend" / "models" / "final_adapter"

# How hard the LoRA adapter pushes. LoRA's contribution scales as alpha/r, so
# this dials remote-sensing vocabulary against the base model's general
# captioning. Measured on 20 held-out BigEarthNet patches and one LEVIR aerial
# scene (`tests/test_domain_adaptation.py`):
#
#   alpha  holdout label F1   caption of an ordinary aerial photo
#   -----  ----------------   -----------------------------------
#     16        0.550         collapses into clipped CORINE terms
#      8        0.362         natural prose, CORINE vocabulary present
#      4        0.000         adapter effectively off
#
# 8 is the default because the demo shows both kinds of imagery. Raise it with
# SATQUERY_LORA_ALPHA=16 for a BigEarthNet-only run; the honest fix for wanting
# both at once is more varied training data, not a bigger alpha.
LORA_ALPHA = int(os.getenv("SATQUERY_LORA_ALPHA", "8"))

_model = None
# ponytail: one global re-entrant lock — a single GPU serialises inference
# anyway, so per-call locks would buy nothing. Re-entrant because the public
# helpers take it and then call load_model(), which takes it again.
_lock = threading.RLock()


def _stage_remote_code():
    """Copy every .py from the weights dir into transformers' dynamic-module cache.

    transformers walks `from .x import` chains to decide which files to copy, and
    misses Moondream's deeper ones (layers.py, rope.py, weights.py, ...), which
    then fails with FileNotFoundError on first load. Copying the lot is cheap and
    removes the whole class of problem. Harmless when transformers got it right.
    """
    import transformers.dynamic_module_utils as dmu

    target = Path(dmu.HF_MODULES_CACHE) / "transformers_modules" / WEIGHTS.name
    target.mkdir(parents=True, exist_ok=True)
    (target / "__init__.py").touch()
    for source in WEIGHTS.glob("*.py"):
        destination = target / source.name
        if not destination.is_file() or destination.stat().st_mtime < source.stat().st_mtime:
            destination.write_bytes(source.read_bytes())


def load_model():
    """Loads Moondream2 once, on GPU if available. Blocks ~20s the first time."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                import torch
                from transformers import AutoModelForCausalLM

                _stage_remote_code()
                device = "cuda" if torch.cuda.is_available() else "cpu"
                _model = AutoModelForCausalLM.from_pretrained(
                    str(WEIGHTS),
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map={"": device},
                )
                _model.eval()
                # SATQUERY_DISABLE_LORA=1 loads the base model instead, which is
                # how the before/after domain-adaptation numbers are measured.
                if ADAPTER.is_dir() and os.getenv("SATQUERY_DISABLE_LORA") != "1":
                    from training.lora_finetune import apply_adapter

                    meta = apply_adapter(_model.model, ADAPTER, alpha=LORA_ALPHA)
                    print(f"[vlm] LoRA adapter loaded: r={meta['r']} "
                          f"alpha={meta['applied_alpha']} "
                          f"on {len(meta['target_modules'])} modules, "
                          f"trained on {meta['trained_on']}")
    return _model


def _open(path):
    with Image.open(path) as img:
        return img.convert("RGB")


def answer_question(image_path, question):
    """Single-image VQA -> {'answer': str}."""
    with _lock:
        return {"answer": load_model().query(_open(image_path), question)["answer"].strip()}


def caption(image_path, length="normal"):
    """Scene description -> {'caption': str}."""
    with _lock:
        return {"caption": load_model().caption(_open(image_path), length=length)["caption"].strip()}


def ground(image_path, expression):
    """Locate a described object -> {'boxes': [[x0,y0,x1,y1] in pixels], 'count': int}.

    Moondream returns normalised coordinates; the frontend and the PDF both want
    pixels, so convert here rather than in two places downstream.
    """
    image = _open(image_path)
    width, height = image.size
    with _lock:
        objects = load_model().detect(image, expression)["objects"]
    boxes = [
        [
            int(o["x_min"] * width),
            int(o["y_min"] * height),
            int(o["x_max"] * width),
            int(o["y_max"] * height),
        ]
        for o in objects
    ]
    return {"boxes": boxes, "count": len(boxes), "expression": expression}


def draw_boxes(image_path, boxes, out_path, colour=(255, 215, 0), width=5):
    """Burn boxes into a copy of the image for the results panel / PDF."""
    from PIL import ImageDraw

    image = _open(image_path)
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box, outline=colour, width=width)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return str(out_path)


def demo():
    sample = Path(__file__).resolve().parents[2] / "sample_data" / "levir_cd" / "A" / "test_1.png"

    cap = caption(sample)
    assert isinstance(cap["caption"], str) and len(cap["caption"]) > 10, cap
    vqa = answer_question(sample, "What kind of land cover is visible in this image?")
    assert isinstance(vqa["answer"], str) and vqa["answer"], vqa
    grounded = ground(sample, "building")
    assert all(len(b) == 4 for b in grounded["boxes"]), grounded
    with Image.open(sample) as img:
        w, h = img.size
    assert all(0 <= b[0] < b[2] <= w and 0 <= b[1] < b[3] <= h for b in grounded["boxes"]), grounded

    print("caption :", cap["caption"][:160])
    print("vqa     :", vqa["answer"][:160])
    print("grounding:", grounded["count"], "boxes for 'building'", grounded["boxes"][:2])
    print("vlm: all checks passed")


if __name__ == "__main__":
    demo()
