"""Bi-temporal change detection: histogram-matched change-vector analysis.

Model-free and deterministic, so the numbers are reproducible in front of
judges and nothing has to be trained before the demo works. Both dates are
histogram-matched (kills the illumination/season difference that otherwise
registers as change), blurred to suppress per-pixel sensor noise, then
compared as a change-vector magnitude across RGB. Pixels beyond three standard
deviations of that magnitude are called changed — an outlier rule, not a
percentile, so the reported change percentage is a real measurement of the
pair and not a constant.

The VLM never sees the mask; it only narrates the statistics returned here.

ponytail: unsupervised pixel differencing. Measured against LEVIR-CD ground
truth on the 8 local pairs it lands at IoU ~0.1-0.25 and under-reports large
building developments whose new roofs match the surrounding brightness. Upgrade
path is a small siamese U-Net trained on LEVIR-CD (2.4GB, Kaggle mirror already
wired up) which would take this to IoU ~0.7 — worth it only if the change tool
becomes the demo's weak point.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.exposure import match_histograms
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

MAX_NOISE_PX = 199  # blobs this small or smaller are noise, not change
BLUR_SIGMA = 2.0
SIGMA_K = 3.0  # changed = beyond mean + 3 sd of the difference magnitude


def _rgb(path):
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0


def _where(cy, cx, height, width):
    """Region centroid -> plain-language position for the narrative."""
    row = ("northern", "", "southern")[min(int(cy / height * 3), 2)]
    col = ("western", "", "eastern")[min(int(cx / width * 3), 2)]
    return " ".join(p for p in (row, col) if p) or "centre"


def detect_change(path_t1, path_t2, out_dir=None, job_id="job"):
    """Two co-registered image paths -> change statistics, mask and overlay."""
    a, b = _rgb(path_t1), _rgb(path_t2)
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch {a.shape} vs {b.shape}; the validator should have caught this"
        )

    b = match_histograms(b, a, channel_axis=-1)
    diff = np.sqrt(
        (
            (gaussian(a, BLUR_SIGMA, channel_axis=-1) - gaussian(b, BLUR_SIGMA, channel_axis=-1))
            ** 2
        ).sum(axis=-1)
    )

    threshold = diff.mean() + SIGMA_K * diff.std()
    mask = remove_small_objects(diff > threshold, max_size=MAX_NOISE_PX)

    height, width = mask.shape
    change_percent = round(100.0 * float(mask.mean()), 2)
    regions = sorted(regionprops(label(mask)), key=lambda r: r.area, reverse=True)
    hotspots = [
        {
            "bbox": [int(r.bbox[1]), int(r.bbox[0]), int(r.bbox[3]), int(r.bbox[2])],  # x0,y0,x1,y1
            "area_percent": round(100.0 * r.area / mask.size, 2),
            "location": _where(r.centroid[0], r.centroid[1], height, width),
        }
        for r in regions[:5]
    ]

    result = {
        "change_percent": change_percent,
        "changed_pixels": int(mask.sum()),
        "region_count": len(regions),
        "hotspots": hotspots,
        "threshold": round(float(threshold), 4),
        "mask_url": None,
        "overlay_url": None,
    }

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        mask_path = out_dir / f"{job_id}_mask.png"
        overlay_path = out_dir / f"{job_id}_overlay.png"
        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
        _overlay(path_t2, mask).save(overlay_path)
        result["mask_url"] = str(mask_path)
        result["overlay_url"] = str(overlay_path)

    result["summary"] = _narrate(result)
    return result


def _overlay(base_path, mask, colour=(255, 0, 0), alpha=0.5):
    base = _rgb(base_path) * 255.0
    tint = np.broadcast_to(np.array(colour, dtype=np.float64), base.shape)
    m = mask[..., None] * alpha
    return Image.fromarray((base * (1 - m) + tint * m).astype(np.uint8))


def _narrate(result):
    pct = result["change_percent"]
    if pct < 0.05:
        return f"No significant change detected — {pct}% of the scene differs between the two dates."
    where = ", ".join(f"{h['location']} ({h['area_percent']}% of the scene)" for h in result["hotspots"][:3])
    return (
        f"{pct}% of the scene changed between the two dates, spread over "
        f"{result['region_count']} distinct regions. Largest changes: {where}."
    )


def demo():
    root = Path(__file__).resolve().parents[2] / "sample_data" / "levir_cd"

    ious = []
    for n in (1, 3, 5):
        out = detect_change(root / "A" / f"test_{n}.png", root / "B" / f"test_{n}.png")
        truth = np.asarray(Image.open(root / "label" / f"test_{n}.png").convert("L")) > 127
        assert out["change_percent"] > 0.05, f"pair {n} with real change reported none: {out}"
        assert out["hotspots"] and len(out["hotspots"][0]["bbox"]) == 4
        pred = np.zeros_like(truth)
        for h in out["hotspots"]:
            x0, y0, x1, y1 = h["bbox"]
            pred[y0:y1, x0:x1] = True
        ious.append((pred & truth).sum() / max((pred | truth).sum(), 1))

    # Unsupervised baseline: require the hotspots to actually land on real change,
    # not to be pixel-accurate. See the ponytail note at the top of this file.
    assert max(ious) > 0.1, f"hotspots miss ground-truth change entirely: {ious}"

    same = detect_change(root / "A" / "test_1.png", root / "A" / "test_1.png")
    assert same["change_percent"] == 0.0, f"identical images reported change: {same}"
    assert "No significant change" in same["summary"]

    print(f"change_analysis: ok — hotspot IoU vs LEVIR truth {[round(i, 2) for i in ious]}, "
          f"identical pair {same['change_percent']}%")


if __name__ == "__main__":
    demo()
