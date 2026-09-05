"""Single-image land-cover measurement: how much of the scene is what.

The VQA tool describes a scene; this one measures it. Same spectral indices the
fusion tool uses (reused from `fusion_analysis`, not reimplemented), but
against one image instead of an optical/SAR pair — so a plain question like
"how much of this is trees?" gets a number and a map instead of prose.

Area in hectares needs the ground sample distance — how many metres one pixel
covers. A GeoTIFF states it in its geotransform and it is read from there; a
PNG or JPEG does not, so the caller must supply it or the tool reports
percentages only and says why. Inventing a resolution would turn a real
measurement into a fabricated number, which is worse than not answering.

ponytail: index thresholds, not a segmentation network. The ceiling is honest
and stated in the output — without a near-infrared band, vegetation comes from
an RGB proxy and water detection is unreliable, because dark roofs, shadow and
asphalt all look like water in RGB. Upgrade path is a small land-cover
segmentation model if the demo needs per-class accuracy rather than magnitude.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

from backend.tools.fusion_analysis import (
    _vegetation_index,
    _water_index,
    read_optical,
)

MAX_NOISE_PX = 49  # speckle smaller than this is not a real patch of anything
BUILT_UP_BRIGHTNESS = 0.35

CLASS_COLOURS = {
    "vegetation": (0, 200, 60),
    "water": (0, 90, 255),
    "built_up_or_bare": (255, 60, 60),
}


def ground_resolution(path):
    """Metres per pixel from a GeoTIFF's geotransform, or None if unknowable."""
    path = Path(path)
    if path.suffix.lower() not in {".tif", ".tiff"}:
        return None
    try:
        import rasterio

        with rasterio.open(path) as src:
            if not src.crs:
                return None
            metres = abs(src.transform.a)
            # A geographic CRS states pixel size in degrees; convert at the equator.
            if src.crs.is_geographic:
                metres *= 111_320.0
            return round(float(metres), 4)
    except Exception:  # noqa: BLE001 — a missing geotransform is not an error here
        return None


def analyze_land_cover(image_path, resolution_m=None, out_dir=None, job_id="job"):
    """One optical image -> per-class coverage, area, and a classified overlay."""
    bands, band_map = read_optical(image_path)
    has_nir = "nir" in bands

    veg_index, veg_name, veg_threshold = _vegetation_index(bands)
    water_index, water_name, water_threshold = _water_index(bands)
    rgb = np.stack([bands["red"], bands["green"], bands["blue"]], axis=-1)
    brightness = rgb.mean(axis=-1)

    vegetation = veg_index > veg_threshold
    # Water must also be dark: in RGB the water index alone flags any blue-ish
    # bright surface, which is how you end up calling a metal roof a lake.
    water = (water_index > water_threshold) & (brightness < 0.35) & ~vegetation
    built_up = (brightness > BUILT_UP_BRIGHTNESS) & ~vegetation & ~water

    classes = {
        "vegetation": remove_small_objects(vegetation, max_size=MAX_NOISE_PX),
        "water": remove_small_objects(water, max_size=MAX_NOISE_PX),
        "built_up_or_bare": remove_small_objects(built_up, max_size=MAX_NOISE_PX),
    }

    resolution_m = resolution_m or ground_resolution(image_path)
    total_px = int(brightness.size)
    height, width = brightness.shape

    coverage = {}
    for name, mask in classes.items():
        pixels = int(mask.sum())
        entry = {
            "percent": round(100.0 * pixels / total_px, 2),
            "pixels": pixels,
            "hectares": None,
            "largest_patch": None,
        }
        if resolution_m:
            entry["hectares"] = round(pixels * resolution_m**2 / 10_000.0, 3)
        entry["largest_patch"] = _largest_patch(mask, resolution_m)
        coverage[name] = entry

    other = total_px - sum(c["pixels"] for c in coverage.values())
    coverage["unclassified"] = {
        "percent": round(100.0 * other / total_px, 2),
        "pixels": other,
        "hectares": round(other * resolution_m**2 / 10_000.0, 3) if resolution_m else None,
        "largest_patch": None,
    }

    result = {
        "vegetation_index": veg_name,
        "water_index": water_name,
        "has_nir": bool(has_nir),
        "optical_bands": len(band_map),
        "resolution_m": resolution_m,
        "resolution_source": (
            None if not resolution_m
            else "geotiff" if ground_resolution(image_path) == resolution_m else "supplied"
        ),
        "image_size": [width, height],
        "scene_hectares": (
            round(total_px * resolution_m**2 / 10_000.0, 3) if resolution_m else None
        ),
        "coverage": coverage,
        "overlay_url": None,
    }

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = out_dir / f"{job_id}_landcover.png"
        _overlay(rgb, classes).save(overlay_path)
        result["overlay_url"] = str(overlay_path)

    result["summary"] = _narrate(result)
    return result


def _largest_patch(mask, resolution_m):
    """Biggest contiguous region: how far it runs, not just how much there is."""
    regions = regionprops(label(mask))
    if not regions:
        return None
    region = max(regions, key=lambda r: r.area)
    y0, x0, y1, x1 = region.bbox
    patch = {
        "bbox": [int(x0), int(y0), int(x1), int(y1)],
        "pixels": int(region.area),
        "extent_px": [int(x1 - x0), int(y1 - y0)],
    }
    if resolution_m:
        patch["extent_m"] = [round((x1 - x0) * resolution_m, 1),
                             round((y1 - y0) * resolution_m, 1)]
        patch["hectares"] = round(region.area * resolution_m**2 / 10_000.0, 3)
    return patch


def _overlay(rgb, classes, alpha=0.45):
    stretched = np.clip(rgb / max(np.percentile(rgb, 98), 1e-3), 0, 1)
    base = stretched * 255.0
    tint = np.zeros_like(base)
    covered = np.zeros(base.shape[:2], dtype=bool)
    for name, mask in classes.items():
        tint[mask] = CLASS_COLOURS[name]
        covered |= mask
    m = covered[..., None] * alpha
    return Image.fromarray((base * (1 - m) + tint * m).astype(np.uint8))


def _area(entry, resolution_m):
    """'12.4% (3.05 ha)' when area is knowable, '12.4%' when it is not."""
    if resolution_m and entry["hectares"] is not None:
        return f"{entry['percent']}% ({entry['hectares']} ha)"
    return f"{entry['percent']}%"


def _narrate(result):
    cover, resolution = result["coverage"], result["resolution_m"]
    parts = [
        f"Vegetation covers {_area(cover['vegetation'], resolution)} of the scene, "
        f"water {_area(cover['water'], resolution)}, and built-up or bare ground "
        f"{_area(cover['built_up_or_bare'], resolution)}."
    ]
    # Never let the classified figures stand alone: if half the scene fell into
    # no class, saying so is the difference between a measurement and a claim.
    unclassified = cover["unclassified"]["percent"]
    if unclassified > 5:
        parts.append(
            f"The remaining {_area(cover['unclassified'], resolution)} matched none of "
            "the three classes cleanly — mixed or intermediate surfaces that the index "
            "thresholds leave undecided rather than force into a class."
        )
    if resolution:
        parts.append(
            f"The scene is {result['scene_hectares']} ha at {resolution} m per pixel."
        )
    else:
        parts.append(
            "Only percentages are available: the image carries no ground resolution, "
            "so pixel counts cannot be converted to hectares. Upload a GeoTIFF, or "
            "state the metres-per-pixel, to get areas."
        )

    biggest = cover["vegetation"]["largest_patch"]
    if biggest and "extent_m" in biggest:
        parts.append(
            f"The largest single stand of vegetation is {biggest['hectares']} ha and runs "
            f"about {biggest['extent_m'][0]} m by {biggest['extent_m'][1]} m."
        )
    elif biggest:
        parts.append(
            f"The largest single stand of vegetation spans {biggest['extent_px'][0]} by "
            f"{biggest['extent_px'][1]} pixels."
        )

    if not result["has_nir"]:
        parts.append(
            f"This image has no near-infrared band, so vegetation is estimated with "
            f"{result['vegetation_index']} rather than NDVI, and the water figure is "
            "unreliable — in RGB alone, shadow and dark roofs resemble water."
        )
    return " ".join(parts)


def demo():
    """Checks the arithmetic on a scene whose true composition is known exactly."""
    import tempfile

    size = 200  # 40,000 px; at 0.5 m/px the scene is exactly 1 hectare
    scene = np.zeros((size, size, 3), dtype=np.uint8)
    scene[:, :] = (150, 150, 150)   # bright grey: built-up
    scene[:100, :] = (40, 180, 60)  # top half: vegetation
    scene[150:, :] = (10, 20, 70)   # bottom quarter: dark blue water

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "scene.png"
        Image.fromarray(scene).save(path)

        # Without a resolution: percentages only, and it must say so.
        bare = analyze_land_cover(path)
        assert bare["resolution_m"] is None
        assert bare["coverage"]["vegetation"]["hectares"] is None
        assert "no ground resolution" in bare["summary"], bare["summary"]
        # Everything is classified in this scene, so no leftover clause is needed.
        assert "matched none" not in bare["summary"], bare["summary"]
        assert 45 < bare["coverage"]["vegetation"]["percent"] < 55, bare["coverage"]
        assert 20 < bare["coverage"]["water"]["percent"] < 30, bare["coverage"]

        # With one: the whole 200x200 scene at 0.5 m/px is 1.0 ha, and the
        # half-scene of vegetation is 0.5 ha.
        scaled = analyze_land_cover(path, resolution_m=0.5, out_dir=tmp, job_id="demo")
        assert scaled["scene_hectares"] == 1.0, scaled["scene_hectares"]
        veg = scaled["coverage"]["vegetation"]
        assert abs(veg["hectares"] - 0.5) < 0.02, veg
        # The vegetation block is 200 px wide by 100 px tall -> 100 m by 50 m.
        assert veg["largest_patch"]["extent_m"] == [100.0, 50.0], veg["largest_patch"]
        assert "ha" in scaled["summary"]
        assert Path(scaled["overlay_url"]).is_file()

        # Percentages must account for the whole image, once.
        total = sum(c["percent"] for c in scaled["coverage"].values())
        assert abs(total - 100.0) < 0.5, total

    print("land_cover: ok —",
          {k: v["percent"] for k, v in scaled["coverage"].items()},
          f"| scene {scaled['scene_hectares']} ha")


if __name__ == "__main__":
    demo()
