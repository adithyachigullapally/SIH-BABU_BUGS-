# SatQuery AI — Complete Build Report (Accounts, Datasets, Models, and a No-Gap Build Sequence)

This is the full companion to the earlier 20-hour build guide. It exists so a
single pass through this document — handed to Claude Code section by
section — gets you from an empty repo to a working, demoable system with no
missing dependency, no ambiguous step, and no undiscovered dead end.

Read Section 0 first — it's the short version. Everything after it is the
extensive version.

---

## 0. Quick reference — everything external you need

**Accounts to create (all free, do this in the first 15 minutes, in parallel across your team):**

| # | Account | URL | Why |
|---|---|---|---|
| 1 | Groq | console.groq.com | Free, fast, function-calling LLM — this is your agent's "brain" |
| 2 | Mistral La Plateforme | console.mistral.ai | Backup agent-brain API if Groq rate-limits or goes down mid-demo |
| 3 | Google AI Studio | aistudio.google.com | Second backup agent-brain API (Gemini) |
| 4 | Hugging Face | huggingface.co | Download model weights + stream dataset subsets |
| 5 | Kaggle | kaggle.com | Free GPU (≈30 hrs/week, T4×2 or P100) to run the LoRA fine-tune off your laptop |
| 6 (optional) | Copernicus Data Space Ecosystem | dataspace.copernicus.eu | Only if you want extra real Sentinel-1/2 scenes beyond BigEarthNet |
| 7 (optional, apply now, don't depend on it) | Anthropic Claude Campus Program | anthropic.com (student builder program) | Free Claude API credits if approved in time — treat as a bonus, not a dependency |

**Datasets to pull:**

| Dataset | Use | Source | Size (pull a subset, not all) |
|---|---|---|---|
| BigEarthNet v2.0 (S1+S2) | LoRA domain adaptation + real co-registered optical–SAR pairs for the fusion tool | `huggingface.co/datasets/GFM-Bench/BigEarthNet` or `bigearth.net` (Zenodo) | Full set ≈110GB — stream 5,000–10,000 patches instead |
| VRSBench | Validate VQA / captioning / grounding | `huggingface.co/datasets/xiang709/VRSBench` | 29,614 images, stream a slice |
| RSVQA-LR / RSVQA-HR | Validate VQA against the named benchmark | Zenodo DOI `10.5281/zenodo.6344333` | Small, download directly |
| CDVQA | Validate change-VQA | `github.com/YZHJessica/CDVQA` | JSON QA files; confirm base imagery in the repo's linked paper before use |
| LEVIR-CD (recommended supplement) | Clean bi-temporal building-change pairs for testing/demoing the change tool | `justchenhao.github.io/LEVIR/` | 637 pairs, 1024×1024 |

**Pretrained model weights (not benchmark data, but also external downloads, all free/Apache-2.0):**

| Model | Role | Source |
|---|---|---|
| `vikhyatk/moondream2` (≈2B) | VQA, captioning, AND built-in object pointing/grounding | Hugging Face |
| `IDEA-Research/grounding-dino-tiny` | Backup/secondary grounding (dedicated open-vocab detector) | Hugging Face |
| `Qwen/Qwen2-VL-2B-Instruct` (optional alternative to Moondream2) | VQA/captioning | Hugging Face |

Nothing else is required. No paid API is needed anywhere in this build — the
plan below uses only free tiers, with redundancy so a single provider going
down doesn't take out your demo.

---

## 1. Account & API key setup (do this first, ~15 minutes, in parallel)

### 1.1 Groq (primary agent brain)
1. Go to console.groq.com → sign up (Google/GitHub login works) → no credit card required.
2. Create an API key under **API Keys**.
3. Confirm tool-use-capable models by name — use these exact IDs:
   - `llama-3.3-70b-versatile` (recommended default — strong tool use)
   - `openai/gpt-oss-120b` (backup, also tool-use capable)
   - `llama-3.1-8b-instant` (fastest, use if you need low latency over reasoning quality)
4. Free tier: ~14,400 requests/day; token-per-minute is the real ceiling (a few thousand TPM) — keep controller prompts short (Section 7 gives compact schemas) so you don't hit it mid-demo.

### 1.2 Mistral La Plateforme (backup #1)
1. console.mistral.ai → sign up → free tier API key (no card).
2. Use model `mistral-large-latest` or `mistral-small-latest` for function calling.
3. Higher TPM ceiling than Groq (~50,000 TPM) — good overflow if Groq throttles.

### 1.3 Google AI Studio / Gemini (backup #2)
1. aistudio.google.com → sign in with Google account → **Get API key**.
2. Use `gemini-2.5-flash` or `gemini-2.0-flash` — both support function calling (`tools` parameter) on the free tier.
3. Free tier quotas vary by model and can be uneven ("Flash works, Pro is often quota-zero") — treat this strictly as your third fallback, not primary.

### 1.4 Hugging Face
1. huggingface.co → sign up → create a **read** access token under Settings → Access Tokens.
2. `huggingface-cli login` locally (or set `HF_TOKEN` env var) before downloading gated/rate-limited resources.

### 1.5 Kaggle
1. kaggle.com → sign up → verify phone number (required to unlock GPU quota).
2. New Notebook → Settings → Accelerator → GPU T4×2 (or P100 if offered).
3. This is where the LoRA fine-tune job runs — not on your laptop.

### 1.6 `.env` template — create this in `backend/.env` immediately

```env
GROQ_API_KEY=
MISTRAL_API_KEY=
GEMINI_API_KEY=
HF_TOKEN=
AGENT_PROVIDER_ORDER=groq,mistral,gemini
```

Never commit this file — add `.env` to `.gitignore` in step 1 of the repo scaffold.

---

## 2. Multi-provider agent-brain failover (do this — it's cheap insurance)

Free tiers rate-limit unpredictably, especially during a live demo when
judges are hammering refresh. Build the controller's LLM call as a thin
wrapper that tries providers in order and falls through on error/429:

```python
# backend/agent/llm_client.py
import os
from groq import Groq
from mistralai import Mistral
import google.generativeai as genai

PROVIDERS = os.getenv("AGENT_PROVIDER_ORDER", "groq,mistral,gemini").split(",")

def call_llm_with_tools(messages, tools, max_retries_per_provider=1):
    last_err = None
    for provider in PROVIDERS:
        try:
            if provider == "groq":
                client = Groq(api_key=os.environ["GROQ_API_KEY"])
                return client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages, tools=tools, tool_choice="auto",
                ), "groq"
            if provider == "mistral":
                client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
                return client.chat.complete(
                    model="mistral-large-latest",
                    messages=messages, tools=tools, tool_choice="auto",
                ), "mistral"
            if provider == "gemini":
                genai.configure(api_key=os.environ["GEMINI_API_KEY"])
                model = genai.GenerativeModel("gemini-2.5-flash", tools=tools)
                return model.generate_content(messages), "gemini"
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"All providers failed: {last_err}")
```

Log which provider actually answered into your execution trace (`"llm_provider": "groq"`)
— this is a nice, honest thing to show judges too: the system is resilient by
design, not just by luck.

Note: each provider's tool-calling response shape differs slightly (Groq/Mistral
are OpenAI-style `tool_calls`; Gemini uses `function_call` parts). Normalize
these into one internal shape (`{name, arguments, id}`) in `llm_client.py`
before the controller loop touches them — this normalization is the single
most bug-prone spot in this whole build, so write a unit test for it with one
fixture response from each provider before moving on.

---

## 3. Dataset acquisition — exact commands

Run these once, early (Track D, hour 0–1), while everyone else works.

### 3.1 BigEarthNet subset (streaming, no 110GB download)

```python
# training/prepare_bigearthnet.py
from datasets import load_dataset

ds = load_dataset("GFM-Bench/BigEarthNet", split="train", streaming=True)
subset = []
for i, example in enumerate(ds):
    subset.append(example)
    if i >= 8000:  # tune this — 5k-10k is enough for a LoRA pass
        break
# Save subset locally as a compact format (parquet/webdataset) for fast re-use
```

This pulls Sentinel-2 optical + Sentinel-1 SAR + multi-label land-cover tags
per patch — one dataset covers both your LoRA fine-tuning need AND gives you
real co-registered optical–SAR pairs to test/demo the fusion tool with,
without needing a separate Copernicus download.

### 3.2 VRSBench

```python
from datasets import load_dataset
vrsbench = load_dataset("xiang709/VRSBench", streaming=True)
```
Pull ~200–500 samples across captioning/VQA/grounding splits for validation —
you don't need the full 29,614 images for a hackathon sanity check.

### 3.3 RSVQA-LR

```bash
wget -O rsvqa_lr.zip "https://zenodo.org/record/6344333/files/RSVQA_LR.zip"
unzip rsvqa_lr.zip -d sample_data/rsvqa_lr
```
(Confirm the exact file name on the Zenodo record page before running — Zenodo
file names occasionally differ from the DOI landing page's displayed title.)

### 3.4 CDVQA

```bash
git clone https://github.com/YZHJessica/CDVQA.git sample_data/cdvqa
```
Open `sample_data/cdvqa/README.md` yourself and confirm the base imagery
source/download link stated there before building against it — this repo
ships the QA JSON files but the underlying image files may need a separate
pull described in that README or the linked paper.

### 3.5 LEVIR-CD (recommended — reliable, self-contained, well-documented)

Download from `justchenhao.github.io/LEVIR/` (links to Google Drive/Baidu).
637 pre-cropped 1024×1024 bi-temporal pairs with binary change masks — use
this as your primary test/demo data for the change-detection tool; it's
cleaner and easier to obtain than chasing CDVQA's base imagery under time
pressure.

---

## 4. Model weight acquisition

```bash
pip install -U huggingface_hub transformers accelerate bitsandbytes peft
huggingface-cli login  # paste your HF token

python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('vikhyatk/moondream2', trust_remote_code=True)
"

python -c "
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny')
AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny')
"
```

Both are Apache-2.0, free for commercial use, no gating/approval wait. Run
these downloads in hour 0 — they total a few GB and you don't want to
discover a slow connection at hour 10.

Moondream2 already supports pointing/object detection/grounded reasoning
natively — you can use it alone for VQA + captioning + grounding and treat
GroundingDINO purely as a fallback if Moondream's box outputs are unreliable
on satellite imagery specifically (it was trained mostly on natural images,
so validate this early, hour 1–3, before committing).

---

## 5. Local environment setup

```bash
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install fastapi uvicorn python-multipart pillow numpy scikit-image rasterio \
            torch --index-url https://download.pytorch.org/whl/cu121 \
            transformers accelerate bitsandbytes peft \
            groq mistralai google-generativeai \
            jinja2 weasyprint datasets
```

Verify CUDA is visible before doing anything else:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
If this prints `False`, stop and fix your PyTorch/CUDA install before writing
a single line of model code — this is the single most common hackathon
time-sink and it's a 5-minute fix now vs. a 2-hour mystery at hour 12.

---

## 6. Full repo structure (create every file up front, even empty, so nothing is missed)

```
satquery-ai/
├── .env                          # never commit
├── .gitignore
├── requirements.txt
├── backend/
│   ├── main.py
│   ├── config.py                 # loads .env, provider order, model paths
│   ├── validator.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── llm_client.py         # Section 2
│   │   ├── controller.py
│   │   └── tool_schemas.py       # Section 7
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── vqa_caption.py
│   │   ├── grounding.py
│   │   ├── change_analysis.py
│   │   └── fusion_analysis.py
│   ├── report.py
│   └── models/                   # LoRA adapter weights land here after training
├── training/
│   ├── prepare_bigearthnet.py    # Section 3.1
│   └── lora_finetune.py          # Section 11
├── frontend/                     # Stitch export lands here
├── sample_data/
│   ├── bigearthnet_subset/
│   ├── vrsbench_subset/
│   ├── rsvqa_lr/
│   ├── cdvqa/
│   └── levir_cd/
├── tests/
│   ├── test_validator.py
│   ├── test_tools.py
│   └── test_controller.py
└── docs/
    └── execution_trace_schema.json
```

Tell Claude Code to scaffold this entire tree in its first action, with every
file stubbed (even a one-line `pass`), before writing real logic into any of
them — that way nothing gets forgotten mid-build.

---

## 7. Agent tool schemas — complete, ready to paste

```json
[
  {
    "name": "run_vqa",
    "description": "Answer a natural-language question about a single remote-sensing image.",
    "input_schema": {
      "type": "object",
      "properties": {
        "image_id": {"type": "string"},
        "question": {"type": "string"}
      },
      "required": ["image_id", "question"]
    }
  },
  {
    "name": "run_caption",
    "description": "Generate a scene-description caption for a single remote-sensing image.",
    "input_schema": {
      "type": "object",
      "properties": {"image_id": {"type": "string"}},
      "required": ["image_id"]
    }
  },
  {
    "name": "run_grounding",
    "description": "Locate the region matching a text description in a single image and return a bounding box.",
    "input_schema": {
      "type": "object",
      "properties": {
        "image_id": {"type": "string"},
        "referring_expression": {"type": "string"}
      },
      "required": ["image_id", "referring_expression"]
    }
  },
  {
    "name": "run_change_analysis",
    "description": "Compare two images of the same location taken at different times and answer a question about what changed.",
    "input_schema": {
      "type": "object",
      "properties": {
        "image_id_t1": {"type": "string"},
        "image_id_t2": {"type": "string"},
        "question": {"type": "string"}
      },
      "required": ["image_id_t1", "image_id_t2", "question"]
    }
  },
  {
    "name": "run_fusion_analysis",
    "description": "Jointly analyze a co-registered optical and SAR image pair to answer a question requiring both modalities.",
    "input_schema": {
      "type": "object",
      "properties": {
        "optical_id": {"type": "string"},
        "sar_id": {"type": "string"},
        "question": {"type": "string"}
      },
      "required": ["optical_id", "sar_id", "question"]
    }
  }
]
```

---

## 8. API contract

**`POST /api/analyze`** — multipart form:
- `image1`: file (required)
- `image2`: file (optional — presence implies a pair; modality/timestamp metadata below disambiguates cross-modal vs bi-temporal)
- `image1_modality` / `image2_modality`: `"optical" | "sar" | "auto"`
- `pair_type`: `"cross_modal" | "bi_temporal" | "none"` (auto-filled by the validator if omitted)
- `query`: string

**Response:**
```json
{
  "answer": "string",
  "confidence": 0.0,
  "task_classified": "run_vqa | run_caption | run_grounding | run_change_analysis | run_fusion_analysis",
  "llm_provider": "groq | mistral | gemini",
  "execution_trace": [
    {"step": 1, "tool": "run_vqa", "parameters": {}, "output_summary": "string"}
  ],
  "visual_evidence": {
    "bbox": [0, 0, 0, 0],
    "overlay_png_url": "string or null",
    "change_mask_url": "string or null"
  },
  "report_url": "/api/report/{job_id}"
}
```

**`GET /api/report/{job_id}`** — returns the rendered PDF (Section 12).

**`GET /api/health`** — returns `{"status": "ok", "gpu_available": true, "providers_ok": {"groq": true, "mistral": true, "gemini": true}}` — build this first, right after the FastAPI skeleton, and check it before wiring anything else. It's your fastest signal that an API key or CUDA setup is broken.

---

## 9. Input validator — concrete checks

```python
# backend/validator.py
import rasterio
from PIL import Image

SUPPORTED_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

def validate_and_profile(path):
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise ValueError(f"Unsupported format: {ext}")
    if ext in {".tif", ".tiff"}:
        with rasterio.open(path) as src:
            return {
                "band_count": src.count,
                "width": src.width,
                "height": src.height,
                "crs": str(src.crs) if src.crs else None,
                "modality_guess": "sar" if src.count == 1 else "optical",
            }
    else:
        img = Image.open(path)
        return {"band_count": len(img.getbands()), "width": img.width,
                "height": img.height, "crs": None, "modality_guess": "optical"}

def check_pair_compatibility(profile1, profile2):
    if profile1["width"] != profile2["width"] or profile1["height"] != profile2["height"]:
        return {"compatible": False, "reason": "dimension mismatch — resample before proceeding"}
    if profile1["crs"] and profile2["crs"] and profile1["crs"] != profile2["crs"]:
        return {"compatible": False, "reason": "CRS mismatch — images not co-registered"}
    return {"compatible": True, "reason": None}
```

Wire this so the agent controller receives `profile1`/`profile2`/`pair_compat`
as part of its first message — this is what lets it correctly classify
cross-modal vs bi-temporal vs single-image without guessing.

---

## 10. Build sequence — literal, ordered, testable at every step

Do these in order. Do not skip the test at the end of a step even if it feels
obvious — this is exactly how "one bug at hour 18" happens.

1. **Scaffold the repo** (Section 6). Test: `tree satquery-ai` shows every file.
2. **Write `requirements.txt`, install, verify CUDA** (Section 5). Test: the CUDA check prints `True`.
3. **Set up `.env` and `config.py`** to load all three provider keys. Test: `python -c "from backend.config import settings; print(settings.groq_key[:4])"` prints something, not `None`.
4. **Build `llm_client.py`** (Section 2) with the three-provider failover. Test: call each provider individually with a trivial tool schema and confirm you get a normalized `{name, arguments}` back from all three.
5. **Build `validator.py`** (Section 9). Test: run it against one LEVIR-CD pair and one mismatched-size pair; confirm it accepts the first and rejects the second with a clear reason.
6. **Build `vqa_caption.py`** wrapping Moondream2. Test: ask "What land cover is visible?" on one VRSBench sample and eyeball the answer for sanity.
7. **Build `grounding.py`.** Test: "Highlight the water body" on a VRSBench grounding sample; compare the returned box against the ground-truth box (VRSBench gives you this for free — use it to sanity-check IoU, doesn't need to be perfect).
8. **Build `change_analysis.py`** (SSIM diff + Otsu threshold + narrative). Test: run on one LEVIR-CD pair; confirm the change % is non-zero and roughly matches the visible change area, and the mask overlay looks right when viewed as an image.
9. **Build `fusion_analysis.py`** (NDVI/NDWI + SAR backscatter stats + narrative). Test: run on one BigEarthNet S1/S2 pair; confirm built-up/water percentages are plausible for the scene (spot-check visually).
10. **Build `tool_schemas.py` and `controller.py`** (Sections 7, plus the loop from the earlier build guide's Section 7). Test: send a single-image VQA query end-to-end through the controller (not the API yet, just the Python function) and confirm the trace has exactly one step calling `run_vqa`.
11. **Build `main.py`** (`/api/health`, then `/api/analyze`). Test: `curl` a real image + query at `/api/analyze` and get back well-formed JSON.
12. **Build `report.py`** (Jinja2 → PDF). Test: hit `/api/report/{job_id}` after a real analyze call and confirm a valid PDF downloads and opens.
13. **Wire the Stitch frontend** to `/api/analyze` and `/api/report`. Test: full click-through from upload to results to PDF download, in the browser, for one sample of each of the five task types.
14. **Kick off / check the LoRA job** (Section 11) — this should already be running in the background since hour 1–2; swap the adapter in now and re-run steps 6–9's tests to confirm no regression.
15. **Run the full test matrix** (Section 13) against all five mandatory requirements before declaring done.

---

## 11. LoRA fine-tuning on BigEarthNet (runs on Kaggle, not the laptop)

```python
# training/lora_finetune.py  — run this in a Kaggle notebook (GPU: T4x2)
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2", trust_remote_code=True,
    load_in_4bit=True, device_map="auto",
)
lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],  # confirm exact module names via model.named_modules()
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# dataset: caption/QA pairs generated from BigEarthNet multi-label tags, e.g.
# "This Sentinel-2 patch contains: agricultural land, coniferous forest, pastures."
# built in prepare_bigearthnet.py

training_args = TrainingArguments(
    output_dir="./lora_out", per_device_train_batch_size=2,
    gradient_accumulation_steps=4, num_train_epochs=1,
    learning_rate=2e-4, fp16=True, logging_steps=20,
    save_strategy="epoch",
)
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
trainer.train()
model.save_pretrained("./lora_out/final_adapter")
```

Download `final_adapter/` from Kaggle and drop it into `backend/models/` when
done. Confirm the exact `target_modules` names for whichever base model you
picked by running `for n, _ in model.named_modules(): print(n)` first —
guessing wrong here is the single most common LoRA setup failure.

---

## 12. Report generation

```python
# backend/report.py
from jinja2 import Template
from weasyprint import HTML

TEMPLATE = Template("""
<h1>SatQuery AI — Analysis Report</h1>
<p><b>Query:</b> {{ query }}</p>
<p><b>Answer:</b> {{ answer }}</p>
<p><b>Confidence:</b> {{ confidence }}</p>
<h2>Execution Trace</h2>
<ol>
{% for step in trace %}
  <li>{{ step.tool }} — {{ step.output_summary }}</li>
{% endfor %}
</ol>
""")

def build_report(job_id, query, answer, confidence, trace):
    html = TEMPLATE.render(query=query, answer=answer, confidence=confidence, trace=trace)
    HTML(string=html).write_pdf(f"/tmp/report_{job_id}.pdf")
```

---

## 13. Test matrix — map every mandatory requirement to a concrete check

| Requirement | Test |
|---|---|
| Single-image VQA (mandatory) | Query a VRSBench image with a factual question, confirm a coherent answer |
| Second single-image task (captioning or grounding) | Run both `run_caption` and `run_grounding` on a VRSBench sample |
| Multi-image change analysis (mandatory) | Run `run_change_analysis` on a LEVIR-CD pair, confirm change % and mask are sane |
| Cross-modal optical–SAR analysis (mandatory) | Run `run_fusion_analysis` on a BigEarthNet S1/S2 pair |
| Agentic orchestration (mandatory) | Confirm the execution trace correctly names the tool actually invoked for 5 different query phrasings |
| RS domain adaptation (mandatory) | Confirm `backend/models/final_adapter` exists and is loaded (log it at startup) |
| Input validation | Feed a deliberately mismatched pair and confirm a clear rejection message, not a crash |
| GUI end-to-end | Full click-through for all 5 task types with visible confidence + trace + PDF download |

---

## 14. Troubleshooting — common failures and fixes

- **CUDA OOM on Moondream2/GroundingDINO**: load in 4-bit (`load_in_4bit=True`), drop batch size to 1, process images sequentially not in parallel.
- **Groq 429 mid-demo**: confirm the failover in Section 2 actually triggers — test this deliberately by using an invalid Groq key temporarily and confirming Mistral picks up the call.
- **Gemini/Mistral/Groq tool-call response shapes differ**: this is the #1 integration bug — write the unit test mentioned in Section 2 before building the controller loop on top of it.
- **Rasterio can't open a GeoTIFF**: usually a missing GDAL system dependency — `pip install rasterio` alone sometimes isn't enough on Windows/WSL; use `conda install -c conda-forge rasterio` if pip fails.
- **BigEarthNet streaming is slow**: cap your subset size explicitly (Section 3.1's `break` at 8000) — don't let a background download job silently run past your patience and eat bandwidth needed elsewhere.
- **LoRA `target_modules` mismatch error**: print `model.named_modules()` first; don't guess module names from a tutorial written for a different base model.
- **Frontend CORS errors calling FastAPI**: add `CORSMiddleware` with `allow_origins=["*"]` for the hackathon (tighten later, never in production).

---

## 15. Day-of execution checklist (condensed)

- [ ] All 3 LLM API keys created and tested individually (Section 1)
- [ ] `.env` populated, `.gitignore` includes it
- [ ] CUDA verified working on the laptop
- [ ] BigEarthNet subset streamed and cached locally
- [ ] VRSBench, RSVQA-LR, LEVIR-CD sample data pulled
- [ ] Moondream2 + GroundingDINO weights downloaded
- [ ] LoRA fine-tune kicked off on Kaggle by hour 1–2
- [ ] Repo fully scaffolded (Section 6) before any real logic is written
- [ ] Each build-sequence step (Section 10) tested before moving to the next
- [ ] Full test matrix (Section 13) passing before hour 19
- [ ] Backup demo video recorded in case live inference stalls
- [ ] Multi-provider failover deliberately tested, not just assumed to work
