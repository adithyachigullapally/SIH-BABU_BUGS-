# SatQuery AI

Agentic remote-sensing analysis. You upload one or two satellite images and ask
a question in plain English; an LLM controller picks the right tool from five,
runs it, and answers from what the tool measured — with the execution trace,
the visual evidence and a PDF report attached.

## Run it

```bash
venv/Scripts/python.exe -m uvicorn backend.main:app --reload
```

Then open <http://127.0.0.1:8000/>. The API docs are at `/docs`.

First analyze call loads Moondream2 onto the GPU (~8s, 4.4GB VRAM); every call
after that is fast.

## What answers what

| Query shape | Tool | How it answers |
|---|---|---|
| "What land cover is visible?" | `run_vqa` | Moondream2 |
| "Describe this scene." | `run_caption` | Moondream2 |
| "How much of this is trees?" | `run_land_cover` | Excess Green / NDVI + water index, % and hectares |
| "Highlight the buildings." | `run_grounding` | Moondream2 pointing, boxes drawn on the image |
| "What changed between these dates?" | `run_change_analysis` | histogram-matched change-vector analysis, 3σ outlier mask |
| "How much of this is water?" (optical + SAR) | `run_fusion_analysis` | NDVI/NDWI vs Sentinel-1 VV backscatter, agreement and disagreement |

Moondream2 carries a LoRA adapter trained on BigEarthNet land-cover captions
(`backend/models/final_adapter`, loaded automatically). On 20 held-out
Sentinel-2 patches it lifts the CORINE label-set F1 from **0.000 to 0.387** —
the base model sees the scene fine but answers "grass and dirt" instead of
"broad-leaved forest, complex cultivation patterns". `LORA_ALPHA` in
`backend/tools/vlm.py` dials how hard the adapter pushes; the measured
trade-off against general captioning is documented there.

Areas in hectares need the ground sample distance. A GeoTIFF states it and it
is read automatically; for a PNG or JPEG, fill in the "metres per pixel" field
(LEVIR-CD is 0.5). Without it the tool reports percentages and says why, rather
than inventing a scale.

The three measurement tools are deterministic — no model in the measurement path,
so the numbers reproduce exactly. The LLM classifies the query and narrates the
result; it never invents a figure.

## Layout

```
backend/
  main.py          FastAPI: /api/health, /api/analyze, /api/report/{job_id}
  validator.py     format/size/CRS checks, modality and pair-type inference
  report.py        one-page PDF per job
  config.py        .env + the provider table
  agent/
    llm_client.py  three brains behind one OpenAI-compatible client, with failover
    tool_schemas.py  the five tools
    controller.py  the agentic loop, execution trace, confidence
  tools/
    vlm.py             Moondream2: VQA, captioning, grounding
    change_analysis.py bi-temporal change detection
    land_cover.py      single-image land-cover area measurement
    fusion_analysis.py optical + SAR fusion
frontend/index.html    single-file UI, served at /
training/prepare_bigearthnet.py   pulls S1/S2 pairs out of the Hub zip by range read
sample_data/           LEVIR-CD pairs + BigEarthNet S1/S2 patches
runs/<job_id>/         uploads, overlays, result.json, report PDF
weights/moondream2/    model weights (not committed)
```

## Checks

```bash
venv/Scripts/python.exe backend/validator.py            # validation rules
venv/Scripts/python.exe -m backend.tools.change_analysis  # vs LEVIR-CD ground truth
venv/Scripts/python.exe -m backend.tools.fusion_analysis  # synthetic + real S1/S2
venv/Scripts/python.exe -m backend.tools.land_cover     # area arithmetic on a known scene
venv/Scripts/python.exe -m backend.tools.vlm            # loads the GPU model
venv/Scripts/python.exe -m backend.agent.controller     # routing, offline
venv/Scripts/python.exe -m backend.report               # PDF renders
venv/Scripts/python.exe -m tests.test_pipeline          # HTTP end to end, no GPU
venv/Scripts/python.exe -m tests.test_matrix            # all 5 task types, live
venv/Scripts/python.exe -m tests.test_domain_adaptation   # LoRA label F1 on held-out patches
venv/Scripts/python.exe -m training.lora_finetune --smoke # training loop, loss must fall
venv/Scripts/python.exe tests/check_providers.py        # the three brains
```

`tests/test_matrix.py` is the pre-demo check: it costs GPU time and free-tier
quota, and it is the one that proves every mandatory requirement.

## Setup notes

- `backend/.env` holds `GROQ_API_KEY`, `MISTRAL_API_KEY`, `GEMINI_API_KEY`,
  `HF_TOKEN`, `AGENT_PROVIDER_ORDER`. Never committed.
- torch comes from the CUDA index, separately from `requirements.txt`:
  `pip install torch --index-url https://download.pytorch.org/whl/cu124`
- Model weights: `snapshot_download('vikhyatk/moondream2', revision='2025-06-21',
  local_dir='weights/moondream2')` — `local_dir` avoids the HF cache's symlinks,
  which Windows refuses without Developer Mode.
- Sample data: LEVIR-CD pairs are already in `sample_data/levir_cd/`;
  `python -m training.prepare_bigearthnet 12` fetches the S1/S2 patches and
  `python -m training.prepare_bigearthnet train 300` builds the LoRA training set.
- Re-running the fine-tune: `python -m training.lora_finetune --epochs 2`
  (~12 minutes, 5.7GB peak on the 4060 — Kaggle is not needed).
