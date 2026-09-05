"""Input validation and profiling.

Runs before the agent sees anything: rejects unreadable/unsupported files,
profiles each image, and works out whether a submitted pair is bi-temporal
(same modality, two dates) or cross-modal (optical + SAR). The controller gets
these profiles in its first message so it classifies the task from facts
instead of guessing from the query wording.
"""
from pathlib import Path

from PIL import Image

SUPPORTED_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
MAX_PIXELS = 8192 * 8192


class ValidationError(ValueError):
    """Bad input from the user — surfaced as a 400, never a stack trace."""


def validate_and_profile(path, modality_hint="auto"):
    """path -> {band_count, width, height, crs, modality, format}.

    modality_hint of "optical"/"sar" overrides the band-count guess, which is
    only a heuristic (single-band can equally be SAR or a panchromatic scene).
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise ValidationError(
            f"Unsupported format '{ext or path.name}'. Supported: "
            + ", ".join(sorted(SUPPORTED_EXT))
        )
    if not path.is_file():
        raise ValidationError(f"File not found: {path}")

    if ext in {".tif", ".tiff"}:
        profile = _profile_geotiff(path)
    else:
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                profile = {
                    "band_count": len(img.getbands()),
                    "width": img.width,
                    "height": img.height,
                    "crs": None,
                }
        except ValidationError:
            raise
        except Exception as exc:  # noqa: BLE001 — PIL raises a zoo of types
            raise ValidationError(f"Could not read image {path.name}: {exc}") from exc

    if profile["width"] * profile["height"] > MAX_PIXELS:
        raise ValidationError(
            f"Image too large ({profile['width']}x{profile['height']}); "
            "tile or downsample it before uploading."
        )

    profile["format"] = ext.lstrip(".")
    profile["name"] = path.name
    profile["path"] = str(path)
    # 1 band = panchromatic or single-pol SAR; 2 bands = Sentinel-1 VV+VH. Optical
    # imagery is 3+ bands in practice, so this splits cleanly for real inputs.
    guess = "sar" if profile["band_count"] <= 2 else "optical"
    profile["modality"] = guess if modality_hint in ("auto", None, "") else modality_hint
    profile["modality_inferred"] = modality_hint in ("auto", None, "")
    return profile


def _profile_geotiff(path):
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - install issue, not user input
        raise ValidationError(
            "GeoTIFF support needs rasterio; upload PNG/JPEG instead."
        ) from exc
    try:
        with rasterio.open(path) as src:
            return {
                "band_count": src.count,
                "width": src.width,
                "height": src.height,
                "crs": str(src.crs) if src.crs else None,
            }
    except Exception as exc:  # noqa: BLE001 — rasterio/GDAL error types vary
        raise ValidationError(f"Could not read GeoTIFF {path.name}: {exc}") from exc


def check_pair_compatibility(profile1, profile2):
    """Two profiles -> {compatible, reason, pair_type}."""
    if profile1["width"] != profile2["width"] or profile1["height"] != profile2["height"]:
        return {
            "compatible": False,
            "pair_type": None,
            "reason": (
                f"Dimension mismatch: {profile1['width']}x{profile1['height']} vs "
                f"{profile2['width']}x{profile2['height']} — resample to a common "
                "grid before comparing."
            ),
        }
    if profile1["crs"] and profile2["crs"] and profile1["crs"] != profile2["crs"]:
        return {
            "compatible": False,
            "pair_type": None,
            "reason": (
                f"CRS mismatch: {profile1['crs']} vs {profile2['crs']} — the images "
                "are not co-registered."
            ),
        }
    cross_modal = profile1["modality"] != profile2["modality"]
    return {
        "compatible": True,
        "reason": None,
        "pair_type": "cross_modal" if cross_modal else "bi_temporal",
    }


def profile_inputs(path1, path2=None, modality1="auto", modality2="auto"):
    """The one entry point main.py calls. Never raises for a valid single image."""
    p1 = validate_and_profile(path1, modality1)
    if path2 is None:
        return {"profile1": p1, "profile2": None,
                "pair": {"compatible": True, "reason": None, "pair_type": "none"}}
    p2 = validate_and_profile(path2, modality2)
    return {"profile1": p1, "profile2": p2, "pair": check_pair_compatibility(p1, p2)}


def demo():
    root = Path(__file__).resolve().parent.parent / "sample_data" / "levir_cd"
    a, b = root / "A" / "test_1.png", root / "B" / "test_1.png"

    single = profile_inputs(a)
    assert single["profile1"]["width"] == 1024, single
    assert single["profile1"]["modality"] == "optical", single
    assert single["pair"]["pair_type"] == "none"

    matched = profile_inputs(a, b)
    assert matched["pair"] == {"compatible": True, "reason": None,
                               "pair_type": "bi_temporal"}, matched

    cross = profile_inputs(a, b, modality2="sar")
    assert cross["pair"]["pair_type"] == "cross_modal", cross

    # Mismatched size must be rejected with a reason, not a crash.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        small = Path(tmp) / "small.png"
        Image.open(a).resize((256, 256)).save(small)
        bad = profile_inputs(a, small)
        assert bad["pair"]["compatible"] is False
        assert "Dimension mismatch" in bad["pair"]["reason"], bad

        junk = Path(tmp) / "notes.txt"
        junk.write_text("nope")
        try:
            validate_and_profile(junk)
        except ValidationError as exc:
            assert "Unsupported format" in str(exc)
        else:
            raise AssertionError("bad extension was accepted")

        corrupt = Path(tmp) / "corrupt.png"
        corrupt.write_bytes(b"\x89PNG\r\n\x1a\n garbage")
        try:
            validate_and_profile(corrupt)
        except ValidationError as exc:
            assert "Could not read" in str(exc)
        else:
            raise AssertionError("corrupt PNG was accepted")

    print("validator: all checks passed")


if __name__ == "__main__":
    demo()
