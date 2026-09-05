"""Pull a handful of co-registered Sentinel-1/Sentinel-2 patches from BigEarthNet.

The build report streams `GFM-Bench/BigEarthNet`, which is a script-based
dataset and no longer loadable (`datasets` 4.x dropped dataset scripts). This
instead range-reads the zip in
`ranjeetgupta/Cross-Modal_Retrieval_BigEarthNet_14K_S1_and_S2` straight off the
Hub: `HfFileSystem` gives a seekable file object, so `zipfile` reads the central
directory and then only the members we ask for. A few megabytes instead of the
3.1GB archive, and no 110GB full BigEarthNet.

Each patch yields an S1 GeoTIFF (VV/VH backscatter), the matching S2 GeoTIFF
(12 optical bands including NIR) and its land-cover labels, which is exactly
what run_fusion_analysis needs to be tested against real data — and what the
LoRA caption pairs get built from later.

Run: venv/Scripts/python.exe -m training.prepare_bigearthnet [n_patches]
"""
import collections
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "sample_data" / "bigearthnet"
REPO = "datasets/ranjeetgupta/Cross-Modal_Retrieval_BigEarthNet_14K_S1_and_S2"
ARCHIVE = f"{REPO}/BigEarthNet_14K.zip"

# Scenes worth demoing the fusion tool on: water and built-up are where optical
# and SAR disagree most usefully.
PREFERRED_LABELS = ("Inland waters", "Urban fabric", "Water bodies",
                    "Industrial or commercial units")


def open_archive():
    from dotenv import load_dotenv
    from huggingface_hub import HfFileSystem

    load_dotenv(ROOT / "backend" / ".env")
    return zipfile.ZipFile(HfFileSystem().open(ARCHIVE, "rb"))


def fetch(n_patches=12):
    import pandas as pd

    archive = open_archive()
    members = archive.namelist()
    by_stem = {Path(name).stem: name for name in members if name.endswith(".tif")}

    meta_name = next(n for n in members if n.endswith("metadata.parquet"))
    meta = pd.read_parquet(io.BytesIO(archive.read(meta_name)))
    # The parquet describes all 480k BigEarthNet patches; this zip carries 13,683
    # of them, so intersect before filtering or every pick misses.
    meta = meta[meta.patch_id.isin(by_stem) & meta.s1_name.isin(by_stem)]
    meta = meta[~meta.contains_cloud_or_shadow & ~meta.contains_seasonal_snow]

    interesting = meta[meta.labels.apply(lambda ls: any(l in PREFERRED_LABELS for l in ls))]
    chosen = pd.concat([interesting.head(n_patches - n_patches // 3),
                        meta.head(n_patches // 3)]).drop_duplicates("patch_id").head(n_patches)

    (OUT / "s1").mkdir(parents=True, exist_ok=True)
    (OUT / "s2").mkdir(parents=True, exist_ok=True)
    manifest = []
    for row in chosen.itertuples():
        s1_member, s2_member = by_stem.get(row.s1_name), by_stem.get(row.patch_id)
        if not (s1_member and s2_member):
            continue
        (OUT / "s1" / f"{row.patch_id}.tif").write_bytes(archive.read(s1_member))
        (OUT / "s2" / f"{row.patch_id}.tif").write_bytes(archive.read(s2_member))
        manifest.append({
            "patch_id": row.patch_id,
            "labels": list(row.labels),
            "country": row.country,
            "s1": f"s1/{row.patch_id}.tif",
            "s2": f"s2/{row.patch_id}.tif",
        })

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total_mb = sum(p.stat().st_size for p in OUT.rglob("*.tif")) / 1e6
    print(f"{len(manifest)} S1/S2 pairs -> {OUT}  ({total_mb:.1f} MB)")
    for entry in manifest[:5]:
        print(f"  {entry['patch_id']}  {', '.join(entry['labels'][:3])}")
    return manifest




# --- LoRA training set -------------------------------------------------------
# Moondream sees RGB, so the 10-band Sentinel-2 patches are rendered to PNG once
# here rather than on every training step. The land-cover labels BigEarthNet
# already carries become the captions — that is the whole domain adaptation
# signal: teach the model to name CORINE land-cover classes from a satellite
# patch instead of describing it as a generic aerial photo.

QUESTIONS = [
    ("Describe this satellite image.",
     lambda ls: f"A Sentinel-2 satellite patch showing {_join(ls)}."),
    ("What land cover types are present in this image?",
     lambda ls: f"The land cover present is: {_join(ls)}."),
    ("Is there water in this scene?",
     lambda ls: "Yes, the scene contains water."
     if any("water" in l.lower() for l in ls) else "No, there is no water in this scene."),
    ("Is this scene urban or rural?",
     lambda ls: "This is an urban scene."
     if any(w in " ".join(ls).lower() for w in ("urban", "industrial", "construction"))
     else "This is a rural scene."),
]


def _join(labels):
    labels = [l.lower() for l in labels]
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def _render_rgb(tif_bytes, out_path):
    """10-band Sentinel-2 -> an 8-bit RGB PNG with a 2-98% contrast stretch."""
    import numpy as np
    import rasterio
    from PIL import Image

    with rasterio.MemoryFile(tif_bytes) as memfile, memfile.open() as src:
        # BigEarthNet S2 band order is B02,B03,B04,... so RGB = bands 3,2,1.
        rgb = np.stack([src.read(3), src.read(2), src.read(1)], axis=-1).astype(np.float32)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    Image.fromarray(rgb.astype("uint8")).resize((378, 378), Image.BICUBIC).save(out_path)


# A random draw follows BigEarthNet's own skew, and the first 300-patch set
# showed it: 164 patches of arable land, 28 with water, 7 industrial. The
# adapter learned farmland vocabulary well and water barely. Fill a per-class
# quota instead so every class the model is expected to name is actually taught.
CLASS_QUOTA = 60


def _stratified(meta, n_patches, quota=CLASS_QUOTA):
    """Pick patches filling a per-class quota, rarest class first, then top up.

    Rarest first matters: satisfying 'Arable land' would incidentally cover most
    agriculture classes and leave inland waters short, because BigEarthNet
    patches carry several labels each and the common ones ride along free.
    """
    counts = collections.Counter(l for ls in meta.labels for l in ls)
    picked, have = [], collections.Counter()
    for cls, _ in sorted(counts.items(), key=lambda kv: kv[1]):
        if have[cls] >= quota or len(picked) >= n_patches:
            continue
        pool = meta[meta.labels.apply(lambda ls: cls in ls) & ~meta.patch_id.isin(picked)]
        take = pool.sample(n=min(quota - have[cls], len(pool), n_patches - len(picked)),
                           random_state=0)
        for row in take.itertuples():
            picked.append(row.patch_id)
            have.update(row.labels)
    if len(picked) < n_patches:
        rest = meta[~meta.patch_id.isin(picked)]
        picked += list(rest.sample(n=min(n_patches - len(picked), len(rest)),
                                   random_state=0).patch_id)
    print("  patches per class: " + ", ".join(
        f"{c}={have[c]}" for c, _ in counts.most_common()))
    return meta[meta.patch_id.isin(picked)]


def build_training_set(n_patches=300, out_dir=None):
    """Render N Sentinel-2 patches to PNG and write question/answer pairs."""
    import pandas as pd

    out_dir = Path(out_dir or OUT / "train")
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    archive = open_archive()
    members = archive.namelist()
    by_stem = {Path(name).stem: name for name in members if name.endswith(".tif")}
    meta = pd.read_parquet(io.BytesIO(archive.read(
        next(n for n in members if n.endswith("metadata.parquet")))))
    meta = meta[meta.patch_id.isin(by_stem)]
    meta = meta[~meta.contains_cloud_or_shadow & ~meta.contains_seasonal_snow]
    meta = _stratified(meta, min(n_patches, len(meta)))

    rows = []
    for i, row in enumerate(meta.itertuples(), 1):
        png = out_dir / "images" / f"{row.patch_id}.png"
        if not png.is_file():
            _render_rgb(archive.read(by_stem[row.patch_id]), png)
        for question, answer_for in QUESTIONS:
            rows.append({"image": f"images/{png.name}",
                         "question": question,
                         "answer": answer_for(list(row.labels))})
        if i % 50 == 0:
            print(f"  {i}/{len(meta)} patches", flush=True)

    (out_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    print(f"{len(rows)} QA pairs over {len(meta)} patches -> {out_dir}")
    return rows


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        build_training_set(int(sys.argv[2]) if len(sys.argv) > 2 else 300)
    else:
        fetch(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
