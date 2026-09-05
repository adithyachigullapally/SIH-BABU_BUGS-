"""FastAPI surface: /api/health, /api/analyze, /api/report/{job_id}.

Each analyze call gets a job directory under runs/<job_id>/ holding the
uploaded images, any overlay or mask the tools produced, the result JSON and
the PDF. That directory is also served read-only at /files/<job_id>/..., so
the frontend can show the visual evidence without a second endpoint.
"""
import json
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.agent.controller import analyze
from backend.config import PROVIDERS, PROVIDER_ORDER
from backend.report import build_report
from backend.validator import ValidationError, profile_inputs

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
RUNS.mkdir(exist_ok=True)

app = FastAPI(title="SatQuery AI", version="1.0")
# Hackathon setting: the frontend is served from wherever a judge opens it.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
app.mount("/files", StaticFiles(directory=RUNS), name="files")



def _to_url(path, job_id):
    """Filesystem path inside the job dir -> the URL the frontend can fetch."""
    if not path:
        return None
    return f"/files/{job_id}/{Path(path).name}"


@app.get("/api/health")
def health():
    import torch

    adapter = ROOT / "backend" / "models" / "final_adapter"
    return {
        "status": "ok",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "providers_ok": {name: bool(PROVIDERS[name]["key"]) for name in PROVIDER_ORDER},
        "vlm_weights_present": (ROOT / "weights" / "moondream2" / "model.safetensors").is_file(),
        "lora_adapter_loaded": adapter.is_dir(),
    }


@app.post("/api/analyze")
async def analyze_endpoint(
    query: str = Form(...),
    image1: UploadFile = File(...),
    image2: UploadFile | None = File(None),
    image1_modality: str = Form("auto"),
    image2_modality: str = Form("auto"),
    resolution_m: float | None = Form(None),
):
    job_id = uuid.uuid4().hex[:12]
    job_dir = RUNS / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for key, upload in (("image1", image1), ("image2", image2)):
        if upload is None or not upload.filename:
            paths[key] = None
            continue
        dest = job_dir / f"{key}{Path(upload.filename).suffix.lower()}"
        with dest.open("wb") as fh:
            shutil.copyfileobj(upload.file, fh)
        paths[key] = dest

    try:
        profiles = profile_inputs(
            paths["image1"], paths["image2"], image1_modality, image2_modality
        )
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not profiles["pair"]["compatible"]:
        # Not a crash and not a silent pass — the requirement is a clear rejection.
        raise HTTPException(status_code=400, detail=profiles["pair"]["reason"])

    result = analyze(query, paths, profiles, out_dir=job_dir, job_id=job_id,
                     resolution_m=resolution_m)
    result["visual_evidence"] = {
        "bbox": result["visual_evidence"].get("bbox"),
        "overlay_png_url": _to_url(result["visual_evidence"].get("overlay_png_url"), job_id),
        "change_mask_url": _to_url(result["visual_evidence"].get("change_mask_url"), job_id),
    }
    result["input_profiles"] = profiles
    result["report_url"] = f"/api/report/{job_id}"

    (job_dir / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


@app.get("/api/report/{job_id}")
def report(job_id: str):
    job_dir = RUNS / job_id
    result_file = job_dir / "result.json"
    if not result_file.is_file():
        raise HTTPException(status_code=404, detail=f"No such job: {job_id}")

    pdf_path = job_dir / f"{job_id}.pdf"
    if not pdf_path.is_file():
        result = json.loads(result_file.read_text(encoding="utf-8"))
        # result.json stores web URLs; the PDF needs the files on disk.
        evidence = result.get("visual_evidence") or {}
        for key in ("overlay_png_url", "change_mask_url"):
            if evidence.get(key):
                evidence[key] = str(job_dir / Path(evidence[key]).name)
        build_report(result, pdf_path)
    return FileResponse(pdf_path, media_type="application/pdf", filename=f"satquery_{job_id}.pdf")


FRONTEND = ROOT / "frontend"
if FRONTEND.is_dir():
    # Registered last: a StaticFiles mount at "/" would shadow any route declared
    # after it, so every /api route above must already exist.
    app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
