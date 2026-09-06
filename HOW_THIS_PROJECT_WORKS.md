# How This Project Works — SatQuery AI, explained in plain words

This file explains everything: what the project is, what each file does, and
exactly what maths each tool runs on the pixels. No jargon without an
explanation right next to it.

Read it top to bottom the first time. After that, jump to the file you care
about using the table of contents.

---

## Table of contents

1. What the project actually is
2. The one-minute version of how a request flows
3. Vocabulary you need (short)
4. The folder layout
5. `backend/main.py` — the web server
6. `backend/validator.py` — the bouncer at the door
7. `backend/config.py` — keys and provider table
8. `backend/agent/llm_client.py` — talking to the LLM
9. `backend/agent/tool_schemas.py` — the menu of tools
10. `backend/agent/controller.py` — the brain loop
11. `backend/tools/change_analysis.py` — pixel subtraction, in full detail
12. `backend/tools/land_cover.py` — measuring vegetation/water/built-up
13. `backend/tools/fusion_analysis.py` — optical + radar together
14. `backend/tools/vlm.py` — the vision model (Moondream2)
15. `backend/report.py` — the PDF
16. `frontend/index.html` — the UI
17. `training/prepare_bigearthnet.py` — getting real satellite data
18. `training/lora_finetune.py` — teaching the model satellite vocabulary
19. `tests/` — every check and what it proves
20. Config files, requirements, environment
21. The honest limitations

---

## 1. What the project actually is

You upload one or two satellite images and type a question in normal English,
like "what changed between these two dates?" or "how much of this is trees?".

The system:

1. Checks the images are readable and that the two of them actually match.
2. Sends the *facts about the images* (not the pixels) plus your question to a
   large language model (LLM) running in the cloud, for free.
3. The LLM does not answer the question. Its only job is to pick which tool
   should run. That is deliberate — see below.
4. The chosen tool runs locally on your machine and produces real, measured
   numbers from the actual pixels.
5. The LLM then writes 2–4 sentences of English around those numbers.
6. You get: an answer, a confidence score with its arithmetic shown, the exact
   list of steps that ran, a picture with the findings drawn on it, and a PDF.

**The key design decision of the whole project:** the language model never
sees the pixels and never produces a number. Language models make up numbers
confidently — that is their most famous failure. So every number in the final
answer comes from deterministic code (fixed maths, same input → same output
every time). The LLM only *routes* and *narrates*. If a judge asks "how do we
know the model didn't invent that 4.6%?", the answer is: it structurally
cannot, because the code that computed 4.6% has no model in it.

---

## 2. The one-minute version of how a request flows

```
Browser (frontend/index.html)
   |  POST /api/analyze  with image1, image2, query text, resolution
   v
backend/main.py
   |  saves uploads to runs/<job_id>/
   v
backend/validator.py
   |  Can I open these files? Same size? Same coordinate system?
   |  Is this one optical + one radar, or the same place at two dates?
   |  --> if bad: HTTP 400 with a plain-English reason. Stop here.
   v
backend/agent/controller.py
   |  builds a short JSON summary of the images ("1024x1024, 3 bands, optical,
   |  pair_type = bi_temporal") and sends it + your question to the LLM
   v
backend/agent/llm_client.py --> Groq, else Mistral, else Gemini
   |  LLM replies: "call run_change_analysis with image1 and image2"
   v
backend/tools/change_analysis.py   (or land_cover / fusion / vlm)
   |  actual pixel maths --> numbers + a PNG mask + a PNG overlay
   v
back to controller.py
   |  tool output goes back to the LLM, which writes the English answer
   |  controller computes the confidence score and the execution trace
   v
backend/main.py --> JSON response --> frontend draws it
                --> runs/<job_id>/result.json saved
                --> /api/report/<job_id> builds the PDF on demand
```

---

## 3. Vocabulary you need

- **Pixel** — one dot in the image. A 1024x1024 image has 1,048,576 of them.
- **Band / channel** — one measurement per pixel. A normal photo has 3 bands
  (Red, Green, Blue). A satellite may have 10–13 bands, including light your
  eye cannot see.
- **NIR (near-infrared)** — an invisible band. It matters enormously because
  living plants reflect NIR very strongly while roads, roofs and water do not.
  With NIR you can tell a green-painted roof from a green field. Without it,
  you are guessing from colour alone.
- **SAR (radar)** — a satellite that sends its own radio pulse down and
  measures the echo. It works at night and through cloud. Smooth water bounces
  the pulse away so it looks almost black; buildings bounce it straight back so
  they look very bright.
- **dB (decibels)** — the unit radar strength is reported in. Negative numbers.
  Water is around −18 dB or lower; buildings around −5 dB or higher.
- **Co-registered** — two images lined up so that pixel (100, 200) in one is the
  exact same patch of ground as pixel (100, 200) in the other. Comparing images
  that are not co-registered is meaningless.
- **GSD / ground resolution / metres per pixel** — how much real ground one
  pixel covers. 0.5 means one pixel = half a metre across. You cannot convert
  a pixel count into hectares without this number.
- **Index** — a formula combining bands into one number per pixel that
  highlights one thing. NDVI highlights plants, NDWI highlights water.
- **Threshold** — the cut-off. "Index above 0.2 = vegetation."
- **Mask** — a black-and-white image the same size as the input, where white
  means "this pixel qualifies" and black means it does not.
- **LLM** — the cloud text model (Groq / Mistral / Gemini) that routes and writes.
- **VLM** — the local vision model (Moondream2) that can look at an image and
  answer in words or point at objects.
- **LoRA** — a small add-on file that adjusts a big model's behaviour without
  retraining the whole thing. Ours is 4.7 MB attached to a 4 GB model.

---

## 4. The folder layout

```
backend/
  main.py                    the web server, 3 endpoints
  validator.py               input checks and image profiling
  config.py                  reads .env, holds the LLM provider table
  report.py                  builds the one-page PDF
  agent/
    llm_client.py            calls the LLM, with automatic failover
    tool_schemas.py          the descriptions of the tools the LLM can pick
    controller.py            the loop: route -> run tool -> narrate -> score
  tools/
    change_analysis.py       compare two dates
    land_cover.py            measure one image
    fusion_analysis.py       optical + radar together
    vlm.py                   Moondream2: describe, answer, point at things
  models/final_adapter/      the trained LoRA file (4.7 MB, committed)
frontend/index.html          the entire UI, one file
training/
  prepare_bigearthnet.py     downloads real Sentinel-1/2 patches
  lora_finetune.py           trains the LoRA adapter
tests/                       five checks, described in section 19
sample_data/                 LEVIR-CD pairs + BigEarthNet patches (not committed)
weights/moondream2/          the 4 GB vision model (not committed, 7.1 GB folder)
runs/<job_id>/               everything one analysis produced
requirements.txt             python packages
.env.example                 template for the secret keys
README.md                    short version of this document
```

Anything in `runs/`, `weights/`, `sample_data/`, `venv/`, and
`backend/.env` is deliberately **not** in git. `.env` holds real API keys —
committing it once puts it in the git history forever, even if you delete it.

---

## 5. `backend/main.py` — the web server

Uses FastAPI. It has exactly three endpoints plus two static file mounts.

**`GET /api/health`** — a status check. Returns whether a GPU is available and
its name, which LLM providers have keys configured, whether the Moondream2
weights file exists, and whether the LoRA adapter folder exists. The frontend
calls this on page load so it can display "GPU: RTX 4060, brains: groq,
mistral, gemini" before you do anything.

**`POST /api/analyze`** — the main one. It accepts:

- `query` — your question, text
- `image1` — required file
- `image2` — optional second file
- `image1_modality` / `image2_modality` — "auto", "optical", or "sar". Auto
  guesses from band count.
- `resolution_m` — optional metres per pixel

What it does, step by step:

1. Makes a random 12-character job ID and a folder `runs/<job_id>/`.
2. Streams each uploaded file to disk as `image1.png`, `image2.tif`, etc. It
   uses `shutil.copyfileobj`, which copies in chunks, so a large upload does
   not get loaded into memory all at once.
3. Calls the validator. If the validator raises, it returns **HTTP 400 with the
   reason as text** — never a stack trace. If the two images are incompatible
   (different sizes, different coordinate systems), also HTTP 400 with the
   reason. This is a requirement: a mismatched pair must be *refused clearly*,
   not silently analysed into garbage.
4. Calls `controller.analyze(...)` which does all the real work.
5. Rewrites the file paths in the result into URLs. Internally the tools return
   `C:\...\runs\abc123\abc123_mask.png`; the browser needs
   `/files/abc123/abc123_mask.png`. There is one subtlety: the browser cannot
   display a GeoTIFF, so if you uploaded a `.tif` the code sets `image1_url` to
   `None` and the frontend falls back to showing the rendered overlay instead.
6. Writes the whole result to `runs/<job_id>/result.json`.
7. Returns the JSON.

**`GET /api/report/{job_id}`** — reads `result.json` back off disk, converts the
web URLs back into file paths (the PDF needs actual files, not URLs), builds
the PDF if it does not exist yet, and returns it. Building lazily means you
only pay for a PDF you asked for.

**Two mounts:**

- `/files` serves `runs/` read-only, so overlays and masks are viewable.
- `/` serves `frontend/`. This is mounted **last**, on purpose, with a comment
  explaining why: a static mount at `/` swallows every route declared after it,
  so all `/api/...` routes must already be registered before this line runs.

CORS is wide open (`allow_origins=["*"]`). That is a hackathon setting so a
judge can open the frontend from anywhere. It would not be right for production.

---

## 6. `backend/validator.py` — the bouncer at the door

Runs before the agent sees anything at all. Three jobs.

**Job 1: is this file usable?**

- Extension must be one of `.tif .tiff .png .jpg .jpeg`. Anything else →
  `ValidationError("Unsupported format ...")` → HTTP 400.
- For PNG/JPEG it opens with Pillow and calls `.verify()`, which detects a
  corrupt file. (Note: `verify()` closes the file, so the code reopens it to
  read the dimensions — that is why the file is opened twice.)
- For GeoTIFF it opens with rasterio, which also gives the CRS (coordinate
  reference system — which map projection the image is in).
- Anything over 8192×8192 pixels (67 million) is rejected with "tile or
  downsample it". This stops a giant upload from freezing the demo machine.

**Job 2: profile the image.** Returns band count, width, height, CRS, format,
name, path, and a guessed modality.

The modality guess is a heuristic and the code says so:

> 1 band = panchromatic or single-polarisation SAR; 2 bands = Sentinel-1 VV+VH;
> 3 or more bands = optical.

The user can override it with the dropdown in the UI, and the profile records
`modality_inferred: true/false` so you can tell whether it was guessed or told.

**Job 3: are these two images comparable?** `check_pair_compatibility` returns
one of three outcomes:

- Different width or height → **incompatible**, "Dimension mismatch: 1024x1024
  vs 256x256 — resample to a common grid before comparing."
- Both have a CRS and they differ → **incompatible**, "CRS mismatch ... the
  images are not co-registered."
- Otherwise compatible, and the pair type is:
  - `cross_modal` if one is optical and the other is SAR
  - `bi_temporal` if both are the same modality (same place, two dates)
  - `none` if there is only one image

That `pair_type` is the single most important fact the LLM gets. The system
prompt says outright: `run_change_analysis` only when bi_temporal,
`run_fusion_analysis` only when cross_modal. So the routing is grounded in a
measured fact about the files, not in whether your question happened to contain
the word "changed".

**Self-check:** running `python backend/validator.py` runs `demo()`, which
asserts a real LEVIR image profiles as 1024 wide and optical, a matched pair is
bi_temporal, a forced SAR override makes it cross_modal, a resized copy is
rejected with "Dimension mismatch", a `.txt` file is rejected, and a
deliberately corrupted PNG is rejected. If any of that breaks, the script fails
loudly.

---

## 7. `backend/config.py` — keys and provider table

Loads `backend/.env` with python-dotenv, then defines `PROVIDERS`, a dictionary
of three cloud LLM providers:

| Name | Base URL | Model |
|---|---|---|
| groq | api.groq.com/openai/v1 | openai/gpt-oss-120b |
| mistral | api.mistral.ai/v1 | ministral-8b-latest |
| gemini | generativelanguage.googleapis.com/v1beta/openai/ | gemini-2.5-flash |

All three of these speak the same request format (the "OpenAI chat-completions
dialect"), so one client library talks to all three and only the URL and the
model name change. That is why there is one `llm_client.py` and not three.

There is a comment worth reading: the model IDs were checked against each
provider's live model list, not copied from the build report. The report's
`llama-3.3-70b-versatile` no longer exists on Groq and `mistral-large-latest`
is not in the free tier. If a provider starts returning 404 or 403 on the model
name, re-run `tests/check_providers.py`.

`PROVIDER_ORDER` comes from the `AGENT_PROVIDER_ORDER` env var, default
`groq,mistral,gemini`. That is the failover order.

Running `python backend/config.py` prints an ok/MISSING table for each key.

---

## 8. `backend/agent/llm_client.py` — talking to the LLM

Two functions, about 50 lines total.

**`call_llm_with_tools(messages, tools, tool_choice)`** loops over
`PROVIDER_ORDER`. For each provider: skip if no key; otherwise build an
`OpenAI` client pointed at that provider's base URL and send the request. On
success, return immediately. On **any** exception, record the error and try the
next one. If all three fail, raise `RuntimeError` listing every failure.

Why catch every exception rather than just network errors? Because free tiers
fail in many ways: 429 rate limit, 503 overloaded, an expired key, a model that
was silently renamed. All of them mean the same thing operationally — try the
next brain. The hard constraint on this project is that no single API outage
can kill the demo.

**`normalize(response)`** flattens whatever came back into one shape:
`{text, tool_calls: [{id, name, arguments}]}`. Two real-world quirks are
handled here:

- Gemini's compatibility layer sometimes returns the arguments as a dict
  already instead of a JSON string, so the code checks the type before parsing.
- Small models sometimes emit malformed JSON. Rather than crashing the whole
  request, it stores `{"_unparsed": "<the raw text>"}` and lets the controller
  turn that into a tool error the LLM can see and correct.

---

## 9. `backend/agent/tool_schemas.py` — the menu of tools

This file is just data: the tool definitions in the standard function-calling
format that all three providers accept. Each has a name, a description, and a
JSON schema of its parameters.

The descriptions are written defensively, because tool choice is where small
backup models get sloppy. Examples of the deliberate wording:

- `run_vqa`'s description says outright: *"Returns prose, never a verified
  number: it cannot count and cannot measure, and any figure it states is
  invented rather than computed."* This is there to stop the LLM routing "how
  many buildings" to the model that would happily make up "about 200".
- `run_grounding` says the referring expression must be a **singular** noun —
  "house", not "houses" — because Moondream draws one box around a whole
  cluster when given a plural.
- `run_land_cover` says "run_vqa only describes; this one returns numbers."

The tools:

| Tool | Inputs | Used for |
|---|---|---|
| `run_vqa` | image_id, question | descriptive question, no quantity |
| `run_caption` | image_id | "describe this" |
| `run_land_cover` | image_id | how much / what area / what percentage |
| `run_grounding` | image_id, referring_expression | locate / highlight / how many |
| `run_change_analysis` | image_id_t1, image_id_t2, question | two dates |
| `run_fusion_analysis` | optical_id, sar_id, question | optical + radar |

---

## 10. `backend/agent/controller.py` — the brain loop

The biggest file in the backend, and the one worth understanding properly.

### The system prompt

A block of instructions given to the LLM. The important rules in it:

- change analysis only when bi_temporal; fusion only when cross_modal.
- With one image: land_cover for "how much / what area / percentage /
  hectares"; grounding for "how many of a countable thing" or "locate /
  highlight"; caption for "what is this"; vqa **only** for a descriptive
  question with no quantity in it at all.
- "run_vqa cannot count and cannot measure. Never route a 'how many' or 'how
  much' question to run_vqa."
- A query asking several things gets one tool call per thing, up to three, then
  one combined answer.
- "Never invent measurements: every number in your answer must appear in a
  tool's output, and if no tool measured something, say plainly that it was
  not measured rather than estimating."

### What the LLM is actually shown

`_inventory()` builds a tiny JSON blob like:

```json
{"images":{"image1":{"size":"1024x1024","bands":3,"modality":"optical"},
           "image2":{"size":"1024x1024","bands":3,"modality":"optical"}},
 "pair_type":"bi_temporal"}
```

Compact separators are used on purpose — the free tiers are limited by tokens
per minute, so every wasted character costs you demo headroom. Note there are
no pixels here. The LLM is choosing a tool from metadata and your sentence.

### The loop

Up to `MAX_STEPS = 3` rounds:

1. Ask the LLM.
2. If it returned no tool calls, its text **is** the answer. Done.
3. If it returned tool calls, run each one via `run_tool()`, append the
   one-line summary of each result to the conversation as a `tool` message, and
   ask again.

If a tool throws, the exception is caught and the string
`"ERROR: ValueError: No image called 'image3'"` is fed back to the LLM as that
tool's output. The LLM then usually corrects itself and calls the tool properly.
This is why a bad tool call does not become a 500 error.

### The "ran out of steps" case

There is a specific, well-commented fix here. If all three rounds got used up
on tool calls, the LLM never got a turn to write prose. The naive thing would
be to hand back the last tool's raw summary — but if the LLM ran a count *and*
an area measurement, returning only the last one throws away half the work. So
the code asks the LLM one more time with `tool_choice="none"`, which forces it
to write prose instead of calling anything. Only if that also fails does it
fall back to the last tool summary.

### `run_tool()` — the dispatcher

A plain if-chain mapping tool name → actual Python function. It also decides
what visual evidence each tool produces:

- `run_land_cover` → a classified overlay PNG
- `run_grounding` → a bounding-box PNG, the first box, **every** box, the label,
  and the image size. Passing every box (not just the first) is deliberate: the
  frontend places a label at each real detection coordinate, so what you see is
  the model's actual output rather than a decorative rectangle.
- `run_change_analysis` → a mask PNG, an overlay PNG, and the largest hotspot's box
- `run_fusion_analysis` → a classified overlay PNG

`_resolve()` turns `"image1"` into a real path and raises a clear error if the
LLM asks for an image that was not uploaded.

### The keyword fallback router

`_fallback_tool(query, profiles)` is what runs when **all three** LLM providers
are down. Pure keyword matching, no network:

- bi_temporal pair → change analysis, always.
- cross_modal pair → fusion analysis, and it works out which image is the
  optical one from the profiles.
- contains "locate / where / highlight / find / show me / bounding box" →
  grounding, and it strips the lead-in phrase to get the object name
  ("highlight the water body" → "water body").
- Then a two-part test for land cover. A quantity word alone is **not** enough:
  "how many buildings" is a counting job for the vision model, while "how much
  forest" is a land-cover measurement. So it requires *both* a quantity word
  (how much, how many, what area, percent, hectare, coverage, how large, area
  of) *and* a land-cover subject (tree, forest, vegetation, green, water, lake,
  river, built, urban, land cover, crop, field, bare, soil).
- "describe / caption / what is this" → caption.
- Everything else → VQA.

This is not as good as the LLM and the confidence score is docked 0.15 when it
is used. It exists purely so an outage cannot end the demo.

### The confidence score

Starts at `CONFIDENCE_BASE = 0.85` and subtracts named amounts:

| Deduction | Reason |
|---|---|
| −0.15 | tool chosen by keyword fallback, no LLM available |
| −0.35 | input pair failed compatibility checks |
| −0.15 | grounding hit its 50-object limit, so the count is a floor |
| −0.35 | grounding found nothing matching the description |
| −0.10 | change is at or below the noise floor (<0.05%) |
| −0.10 | no ground resolution, so percentages only and no areas |
| −0.15 | no near-infrared band, vegetation index is a proxy |
| −0.05 | free-text model output (vqa/caption), not a measurement |

Floor of 0.1. Every deduction is returned **twice**: once as a reason string
and once as `{amount, reason}`, so the UI and the PDF can literally show
`0.85 − 0.15 − 0.10 = 0.60`. The controller's own self-check asserts that the
listed deductions add up to the gap — otherwise the breakdown shown to a judge
would be decoration rather than arithmetic.

### What `analyze()` returns

```
job_id, answer, confidence, confidence_reasons, confidence_base,
confidence_deductions, measurements, thresholds, task_classified,
llm_provider, execution_trace, visual_evidence, elapsed_seconds, query
```

`measurements` is the headline numbers structured as `{label, value, unit}` so
a UI can display them without parsing a sentence. `thresholds` states which
index and which cut-off decided each class, so a reader can audit the numbers.

**Self-check:** `python -m backend.agent.controller` runs the fallback router
against real profiles, runs a genuine change analysis offline, and verifies the
confidence arithmetic. No network and no GPU needed.

---

## 11. `backend/tools/change_analysis.py` — pixel subtraction, in full detail

**Question it answers:** "what changed between these two dates?"

**Method name:** histogram-matched change-vector analysis with a 3-sigma
outlier threshold. Model-free, so the numbers reproduce exactly.

### The maths, step by step

Call the two images A (time 1) and B (time 2). Both are the same size — the
validator guaranteed that — and both get read as RGB floats in the range 0 to 1.

**Step 1 — histogram matching.**

```python
b = match_histograms(b, a, channel_axis=-1)
```

This is the step people miss and it is the reason naive change detection fails.
Two photos of the same place taken months apart differ in overall brightness
and colour because of sun angle, season, atmosphere and sensor gain. If you
subtract them raw, *the entire image* comes out as "changed". Histogram
matching reshapes B's brightness distribution to match A's, so the illumination
difference cancels and only real ground change survives. It is done per channel.

**Step 2 — blur both images.**

```python
gaussian(a, sigma=2.0), gaussian(b, sigma=2.0)
```

A Gaussian blur with sigma 2 averages each pixel with its neighbours. This
kills per-pixel sensor noise, which would otherwise show up as thousands of
single-pixel "changes". It costs a little spatial precision at the edges of
real changes, which is an accepted trade.

**Step 3 — the change vector, per pixel.**

```
diff = sqrt( (Ar-Br)^2 + (Ag-Bg)^2 + (Ab-Bb)^2 )
```

For every pixel, take the difference in Red, in Green, and in Blue; square each;
add them; square-root. This is the straight-line distance between the two
pixels' colours in 3D colour space. Small number = the pixel looks the same at
both dates. Big number = it looks different. This produces one grey "difference
magnitude" image the same size as the inputs.

**Step 4 — the threshold.**

```python
threshold = diff.mean() + 3.0 * diff.std()
mask = diff > threshold
```

This is the important choice. It does **not** use a fixed cut-off and it does
**not** use a percentile. It computes the average difference across this
specific pair and its standard deviation (how spread out those differences are)
and calls a pixel "changed" only if it is more than three standard deviations
above the average.

Why that matters: a percentile rule like "the top 2% of pixels are changed"
would report exactly 2% change on every pair, including two identical images.
That would be a constant dressed up as a measurement. The 3-sigma rule is an
*outlier* rule — if nothing unusual happened, almost nothing crosses the line,
and identical images correctly report 0.00%.

**Step 5 — remove small blobs.**

```python
remove_small_objects(mask, max_size=199)
```

Any connected white blob of 199 pixels or fewer is deleted. A real new building
or a cleared field is hundreds or thousands of pixels; 12 scattered pixels is
noise that survived the blur.

**Step 6 — measure and describe.**

- `change_percent` = what fraction of the whole image is white in the mask.
- `changed_pixels` = the raw white pixel count.
- `region_count` = how many separate connected blobs there are (`label()` gives
  every blob its own number, `regionprops()` measures each).
- `hotspots` = the five biggest blobs, each with its bounding box in
  `[x0, y0, x1, y1]` pixels, its share of the scene, and a plain-language
  location.

The location comes from `_where()`, which splits the image into a 3×3 grid and
names the cell: "northern eastern", "southern western", or "centre" for the
middle. It is deliberately simple — the point is a human-readable sentence, not
a coordinate.

**Step 7 — the pictures.**

- `<job>_mask.png` — pure black and white, the mask itself.
- `<job>_overlay.png` — image B with the changed pixels tinted red at 50%
  opacity, computed as `base*(1-m) + tint*m` where m is the mask times alpha.

**Step 8 — the sentence.** `_narrate()` writes either "No significant change
detected — 0.0% of the scene differs" (below 0.05%) or "4.6% of the scene
changed between the two dates, spread over 37 distinct regions. Largest
changes: northern eastern (0.9% of the scene), ..."

### The honest limitation, stated in the file

There is a `ponytail:` comment at the top saying: measured against LEVIR-CD
ground truth on 8 local pairs, this lands at IoU roughly 0.1–0.25 and
under-reports large building developments whose new roofs happen to match the
surrounding brightness. (IoU = intersection over union: how much your detected
area overlaps the true answer, 1.0 being perfect.) The stated upgrade path is a
small siamese U-Net trained on LEVIR-CD, which would reach roughly IoU 0.7 —
worth doing only if change detection becomes the demo's weak point.

Writing the ceiling down is deliberate. Discovering it live in front of a judge
is worse than declaring it.

**Self-check:** `python -m backend.tools.change_analysis` runs three real LEVIR
pairs, checks each reports more than 0.05% change, computes hotspot IoU against
the ground-truth mask and requires the best to exceed 0.1, and — the important
one — runs an image against **itself** and asserts the result is exactly 0.0%.

---

## 12. `backend/tools/land_cover.py` — measuring one image

**Question it answers:** "how much of this is trees / water / built-up?"

This is the tool that turns a picture into numbers. It reuses the index
functions from `fusion_analysis.py` rather than duplicating them.

### The maths

**Step 1 — read the bands.** `read_optical()` returns red, green, blue and, if
the file has it, near-infrared, each as a 2D array of floats 0–1.

**Step 2 — vegetation index.**

If NIR is present, use **NDVI**:

```
NDVI = (NIR - Red) / (NIR + Red)          threshold: > 0.2 is vegetation
```

Living leaves reflect NIR strongly and absorb red (chlorophyll eats red light),
so healthy plants score high. Bare soil scores near 0, water scores negative.

If NIR is absent (a normal PNG or JPEG), use **Excess Green** on
chromatic-normalised RGB:

```
total = R + G + B
ExG   = 2*(G/total) - (R/total) - (B/total)      threshold: > 0.08
```

Dividing by the total first removes overall brightness, so a shaded lawn and a
sunlit lawn score similarly.

There is a documented reason ExG is used instead of the more common VARI:
VARI's denominator is `green + red - blue`, which passes through zero on
ordinary imagery, so its values explode without bound. Measured on the LEVIR
pair, VARI's 95th percentile was 0.22 on one date and 3.43 on the other — no
fixed threshold can survive that. ExG is bounded and stable across the same
pair.

**Step 3 — water index.**

With NIR, use **NDWI**:

```
NDWI = (Green - NIR) / (Green + NIR)      threshold: > 0.0 is water
```

Water absorbs NIR almost completely, so water goes strongly positive while
everything else goes negative. It is a very clean separator.

Without NIR, the fallback is `(Blue - Red) / (Blue + Red)`, and this is where
the tool makes its most important honest decision — see step 5.

**Step 4 — build the three masks.**

```python
vegetation = veg_index > veg_threshold
water      = (water_index > water_threshold) & (brightness < 0.35) & ~vegetation
built_up   = (brightness > 0.35) & ~vegetation & ~water
```

Note the extra conditions. Water must also be *dark*, because the RGB water
index alone flags any bluish bright surface — which is how you end up calling a
metal roof a lake. The `~` means "not": the classes are made mutually exclusive
in priority order, vegetation first.

`brightness` is just the mean of R, G and B per pixel.

**Step 5 — the RGB water decision (read this one).**

If the image has no NIR band, the tool **refuses to report a water percentage
at all**. It sets `water: {percent: None, measurable: false}` and those pixels
fall through to "unclassified".

The measurement behind that decision is written in the code: on LEVIR test_8,
the RGB water index reads −0.066 on the actual pond, −0.081 on house roofs, and
−0.073 on forest canopy. Dark water, dark shingle and tree shadow are one
statistical population in visible light — they cannot be separated. The old
threshold returned "1.04% water" that was actually 106 roof specks and almost
none of the pond.

The reasoning in the comment is worth quoting: *"A caveat in prose does not
undo a number in a table."* If you print 1.04% and add a footnote, people
remember 1.04%. So the number is not printed at all. A near-infrared band
separates them cleanly, and the narration says exactly that.

**Step 6 — clean up.** `remove_small_objects(..., max_size=49)` drops any patch
of 49 pixels or fewer as speckle.

**Step 7 — pixels to hectares.** This needs the ground resolution.

```
hectares = pixels * resolution_m^2 / 10000
```

(One hectare is 10,000 square metres.) The resolution comes from one of two
places:

- A GeoTIFF states it in its geotransform. `ground_resolution()` reads
  `abs(src.transform.a)`. If the file is in a *geographic* CRS the pixel size is
  in degrees, not metres, so it multiplies by 111,320 (metres per degree at the
  equator) to convert.
- Otherwise the user types it in the UI ("metres per pixel"). LEVIR-CD is 0.5.

If neither is available, the tool reports percentages only and the narration
says: *"Only percentages are available: the image carries no ground resolution,
so pixel counts cannot be converted to hectares. Upload a GeoTIFF, or state the
metres-per-pixel, to get areas."* Inventing a resolution would turn a real
measurement into a fabricated number.

**Step 8 — the largest patch.** For each class it finds the single biggest
connected region and reports its bounding box, pixel count, extent in pixels
and — with a resolution — extent in metres and area in hectares. This answers
"is that 30% vegetation one forest or a thousand scattered bushes?"

**Step 9 — unclassified.** Whatever matched none of the three classes is
counted and reported. If it is over 5% the narration says so explicitly:
*"The remaining 24% matched none of the three classes cleanly — mixed or
intermediate surfaces that the index thresholds leave undecided rather than
force into a class."* Percentages that do not add to 100 are a red flag, so the
tool makes the remainder visible instead of hiding it.

**Step 10 — the overlay.** Green tint = vegetation, blue = water, red =
built-up, 45% opacity over a contrast-stretched version of the original.

### The stated ceiling

Also written at the top of the file: these are index thresholds, not a trained
segmentation network. Without NIR, vegetation comes from an RGB proxy and water
is not measurable at all. The upgrade path is a small land-cover segmentation
model if the demo needs per-class accuracy rather than magnitude.

**Self-check:** `python -m backend.tools.land_cover` builds a synthetic 200×200
image whose composition is known exactly — top half vegetation, bottom quarter
dark blue, the rest grey. At 0.5 m/px the whole scene is exactly 1.0 hectare and
the vegetation half is exactly 0.5 ha, and the test asserts both to within 0.02.
It also asserts the vegetation block measures exactly 100 m × 50 m, that water
is reported as unmeasurable on this RGB image, that the blue quarter lands in
unclassified, and that all percentages sum to 100 ± 0.5.

---

## 13. `backend/tools/fusion_analysis.py` — optical + radar together

**Question it answers:** "using both the optical and radar views, how much of
this is water?"

The point of fusion is not running two tools next to each other. It is saying
**where the two modalities agree and where they disagree**, because that is
information neither image holds alone.

### The physics, in one table

| Surface | Optical looks like | Radar looks like |
|---|---|---|
| Open water | dark, bluish | very dark (−18 dB or lower) — the pulse bounces away |
| Buildings | bright grey | very bright (−5 dB or higher) — walls bounce it straight back |
| Vegetation | high NDVI | middling — the canopy scatters in all directions |

### The rules

```python
water      = optical_water  AND radar_dark      # both agree
builtup    = optical_bright AND radar_bright    # both agree
vegetation = optical_veg    AND NOT radar_dark
flooded    = optical_veg    AND radar_dark      # DISAGREEMENT
```

That last one is the interesting output. A pixel that looks vegetated in the
photo but is radar-dark is usually **flooded vegetation** — water sitting under
a tree canopy — or terrain shadow. Optical imagery alone cannot see that, since
the canopy hides the water. This is exactly the finding a judge asks about, so
it is reported rather than hidden.

Two more disagreement figures are reported:

- `optical_only_water` — looks like water in the photo but not in radar. More
  likely cloud shadow or a dark surface than actual water.
- `radar_only_dark` — radar-dark but not optically water. Smooth surfaces like
  tarmac, or radar shadow behind a hill.

### Reading the files correctly

Three details in this file exist because getting them wrong fails *silently*,
which is worse than crashing:

**1. Sentinel-2 scaling.** Sentinel-2 L2A GeoTIFFs store reflectance multiplied
by 10,000 as 16-bit integers. An 8-bit photo stores 0–255. The code decides
which by checking `arr.max() > 300` rather than assuming. Divide by the wrong
one and every index is nonsense.

**2. Which SAR band.** BigEarthNet's Sentinel-1 files store VH first, then VV
(two different radar polarisations). The thresholds in this file are **VV**
thresholds. VH runs about 6 dB below VV, so reading band 1 instead of band 2
would silently label roughly a fifth of every forest scene as water. The
`SAR_VV_BAND = {1: 1, 2: 2}` mapping exists for exactly that.

**3. dB or amplitude.** Sentinel-1 GRD arrives already in decibels (negative
floats). An ordinary image file carries amplitude and has to be converted with
`10*log10(amplitude^2)`. The code decides from the data — `if arr.min() < 0` it
is already dB — rather than assuming, because getting this backwards turns every
threshold into nonsense with no error message.

### The sign-preserving divide

`_safe_divide()` deserves a mention. All these indices are fractions, and you
must guard against dividing by zero. The obvious guard — clamp the denominator
to a small **positive** number — is a bug here. VARI's denominator
(`green + red - blue`) goes negative over deep water and shadow, and clamping
those to a small positive value flips them to a huge positive index, i.e. it
calls open water dense vegetation. So the guard preserves the sign it was given:

```python
floor = np.where(denominator < 0, -1e-6, 1e-6)
```

### The tuning knobs

`SAR_WATER_DB = -18.0` and `SAR_BUILTUP_DB = -5.0` are tuned to Sentinel-1 GRD
VV over land. They are module-level constants specifically so a different sensor
can be recalibrated without touching any logic. Real hardware always needs
tuning that a clean model does not anticipate.

Also note there are separate thresholds per index — `NDVI_VEG_THRESHOLD = 0.2`
and `EXG_VEG_THRESHOLD = 0.08` — with a comment explaining why: NDVI and Excess
Green do not live on the same numeric scale, and one shared constant is how a
woodland scene gets reported as 2% trees.

**Self-check:** `python -m backend.tools.fusion_analysis` builds a synthetic
256×256 scene (blue water strip that is radar-dark, grey block that is
radar-bright, vegetation between) and asserts each class is found above 20%.
Then, if the BigEarthNet sample data exists, it runs a **real** Sentinel-1/2
pair labelled "Inland waters" and asserts NIR is present, NDVI is the chosen
index, the mean backscatter is in a physically sane range, and vegetation
exceeds 5%.

---

## 14. `backend/tools/vlm.py` — the vision model

Wraps **Moondream2**, a ~4 GB vision-language model that runs locally on your
GPU. One module handles all three of its uses, because they are all the same
loaded model in the same process — splitting them across files would mean two
modules racing to load the same 4 GB singleton.

### Loading

`load_model()` uses the double-checked locking pattern: check `_model is None`,
take the lock, check again, then load. It loads in float16 on GPU (half the
memory, plenty of precision for inference) or float32 on CPU. First call takes
about 8 seconds and 4.4 GB of VRAM; every call after that is fast.

There is one global re-entrant lock, `_lock`. Re-entrant because the public
helpers take the lock and then call `load_model()`, which takes it again — a
plain lock would deadlock. It is a single global lock because a single GPU
serialises inference anyway, so per-call locks would buy nothing.

`_stage_remote_code()` fixes a real, annoying bug. Moondream ships its own Python
code alongside its weights (that is what `trust_remote_code=True` means).
transformers copies that code into a cache directory by following
`from .x import` chains — and it misses Moondream's deeper files (layers.py,
rope.py, weights.py), which then fails with FileNotFoundError on first load. The
fix copies **every** `.py` from the weights folder into the cache. Cheap, and it
removes the whole class of problem.

### The three functions

**`answer_question(path, question)`** → `{"answer": "..."}`. Free-text VQA.

**`caption(path)`** → `{"caption": "..."}`. Whole-scene description.

**`ground(path, expression)`** → `{boxes, count, expression, saturated}`. This
is the one that actually counts things. Moondream's `detect()` returns
normalised coordinates (0–1); the code converts to pixels here rather than in
two places downstream.

Two safeguards in `ground()`:

- **Plural handling.** Moondream's detect wants a singular noun. Given "houses"
  it draws one box around the whole cluster and returns 1 object; given "house"
  it finds 50. So if the result is 0 or 1 objects and the expression ends in "s"
  (but not "ss"), it retries with the "s" stripped and keeps whichever found
  more. This guard lives in the tool, not in the router, so it covers every
  caller — the LLM and the keyword fallback both.
- **Saturation.** Moondream's detect stops at 50 objects. So a result of exactly
  50 means "at least 50", not "50". Saying "50 houses" for a scene holding 200 is
  the same failure as inventing a number. The result carries
  `saturated: true`, the summary reads "at least 50 region(s) matching
  'house'", and the confidence drops 0.15.

**`draw_boxes()`** burns gold rectangles into a copy of the image for the
results panel and the PDF.

### The LoRA adapter

If `backend/models/final_adapter/` exists and `SATQUERY_DISABLE_LORA` is not
set to 1, the adapter is attached at load time. `LORA_ALPHA` (env var
`SATQUERY_LORA_ALPHA`, default 8) controls how hard it pushes — LoRA's
contribution scales as alpha divided by rank, so this is a live dial that needs
no retraining. The measured trade-off is written into the file:

| alpha | holdout label F1 | caption of an ordinary aerial photo |
|---|---|---|
| 16 | 0.550 | collapses into clipped CORINE terms |
| 8 | 0.362 | natural prose, CORINE vocabulary present |
| 4 | 0.000 | adapter effectively off |

8 is the default because the demo shows both kinds of imagery. The comment is
blunt about the real fix: *"the honest fix for wanting both at once is more
varied training data, not a bigger alpha."*

### Why GroundingDINO is not used

The build report specified a separate GroundingDINO model for object detection.
Moondream does pointing natively, so it is not downloaded at all — that is
another gigabyte of weights and another dependency avoided. The note says to add
it only if Moondream's boxes prove unreliable on satellite imagery.

**Self-check:** `python -m backend.tools.vlm` loads the model, captions a real
LEVIR image, asks a VQA question, grounds "building", and asserts every returned
box is 4 numbers and lies inside the image bounds with x0 < x1 and y0 < y1.

---

## 15. `backend/report.py` — the PDF

One page per job, built with **fpdf2**, drawn directly rather than through the
build report's Jinja2 → HTML → weasyprint chain. The reason is written down:
weasyprint needs GTK system libraries on Windows, and the report is a title, a
few fields, a numbered trace and an image — no CSS layout worth two extra
dependencies and a Windows install headache. Swap back only if the layout gets
genuinely complex.

Contents: the title, a grey line with job ID / elapsed seconds / which brain
answered, then Query, Task classified, Answer, Confidence (with its reasons in
brackets), then the numbered execution trace with each step's tool, parameters
and output summary, then up to two evidence images side by side.

`_text()` exists because fpdf2's built-in fonts are latin-1 only. Em-dashes and
curly quotes — which the LLM produces constantly — would crash the encoder, so
they are replaced with plain ASCII equivalents first.

**Self-check:** `python -m backend.report` builds a PDF from a fake result,
asserts the file starts with the bytes `%PDF` and is over 800 bytes, then builds
a second one with a random image attached and asserts it is **larger** — which
proves the image was actually embedded rather than silently skipped.

---

## 16. `frontend/index.html` — the UI

One file, 813 lines: HTML, CSS and JavaScript together. No build step, no
framework, no npm. It is served by FastAPI at `/`.

**Layout:** a left rail with the form (two image drop zones with modality
dropdowns, a metres-per-pixel field, the query box, and the Analyze button) and
a right work area that shows the results.

**On load** it calls `/api/health` and displays the GPU name and which brains
have keys.

**While analysing** it shows the uploaded image with a scanning sweep animation
and cycles phase labels: "Dispatching to the agent" → "Running analysis tools"
→ "Measuring pixels" → "Composing the answer".

**When the result arrives** `paint()` renders:

- The image viewer, with pan and zoom (mouse wheel zooms toward the cursor,
  drag pans). If both the original and an overlay exist, there is a **compare**
  slider that wipes between them and a **sweep** button that animates the wipe.
- Bounding boxes drawn at their real scaled coordinates with labels, using
  `image_size` from the backend to scale from model pixels to screen pixels.
- A legend of which colour means which class, and the rule that decided it
  (pulled from the `thresholds` field).
- The measurement tiles — label, value, unit.
- The answer text.
- The confidence ledger, shown as visible arithmetic: base score 0.85, then
  each deduction on its own line with its amount and reason, then the net.
- The execution trace, one row per step: tool name, parameters, output summary.
- The validated-input facts: size, bands, modality, CRS, pair type.
- A link to the PDF.

`esc()` escapes every piece of text before it goes into `innerHTML`, so a model
answer containing `<script>` cannot execute.

It respects `prefers-reduced-motion` and disables the animations if the user has
that set.

---

## 17. `training/prepare_bigearthnet.py` — getting real satellite data

**Problem:** BigEarthNet is about 110 GB. You cannot download it for a
hackathon, and the build report's approach (`datasets` streaming a
script-based dataset) stopped working when `datasets` 4.x dropped dataset
scripts.

**Solution, and it is a nice one:** a 3.1 GB zip lives on the Hugging Face Hub.
`HfFileSystem().open(...)` gives a **seekable** file object over HTTP. Python's
`zipfile` can therefore read the zip's central directory (a small index at the
end of the file) and then range-request only the specific members you ask for.
So you download a few megabytes instead of 3.1 GB, and never touch the 110 GB
original.

**Two modes:**

`python -m training.prepare_bigearthnet 12` — fetches 12 patches for testing
fusion. Each patch is a Sentinel-1 GeoTIFF (VV/VH radar), the matching
Sentinel-2 GeoTIFF (12 optical bands including NIR), and its land-cover labels.
It filters out cloudy and snowy patches, and prefers scenes labelled "Inland
waters", "Urban fabric", "Water bodies" or "Industrial or commercial units" —
because water and built-up are where optical and radar disagree most usefully.
It writes a `manifest.json` listing each patch.

One detail: the parquet metadata file describes all 480,000 BigEarthNet
patches, but this zip only carries 13,683 of them. The code intersects the two
sets **before** filtering, or every pick would miss.

`python -m training.prepare_bigearthnet train 300` — builds the LoRA training
set. It renders each 10-band Sentinel-2 patch to an RGB PNG (Moondream sees RGB;
rendering once here beats re-rendering on every training step), applies a 2–98
percentile contrast stretch, resizes to 378×378, and writes four
question/answer pairs per patch:

1. "Describe this satellite image." → "A Sentinel-2 satellite patch showing
   arable land, broad-leaved forest and inland waters."
2. "What land cover types are present in this image?" → "The land cover present
   is: ..."
3. "Is there water in this scene?" → yes/no from the labels
4. "Is this scene urban or rural?" → urban/rural from the labels

**The stratified sampling — an important fix.** A random draw follows
BigEarthNet's own skew, and the first 300-patch set showed it clearly: 164
patches of arable land, 28 with water, 7 industrial. The adapter learned
farmland vocabulary well and water barely.

So `_stratified()` fills a per-class quota of 60, **rarest class first**. Order
matters: satisfying "Arable land" first would incidentally cover most
agriculture classes and leave inland waters short, because BigEarthNet patches
carry several labels each and the common classes ride along free. Starting with
the rarest guarantees every class the model is expected to name is actually
taught.

---

## 18. `training/lora_finetune.py` — teaching the model satellite vocabulary

### What problem this solves

The base Moondream2 looks at a Sentinel-2 patch and says "an aerial photo of
grass and dirt". That is not wrong, but it is not remote-sensing language. The
domain adaptation teaches it to name **CORINE land-cover classes** — "arable
land, broad-leaved forest and inland waters" — which is the vocabulary the
field actually uses.

### Why the build report's recipe could not work

The report said use `get_peft_model` plus HuggingFace `Trainer`. That fails
here: `HfMoondream` is an inference-only wrapper with no trainable `forward()`,
so `Trainer` has nothing to call. Also `model.text` is a bare `nn.ModuleDict`,
not a `PreTrainedModel`, so the standard PEFT wrapper does not apply.

What Moondream *does* expose is its uncached full-sequence path —
`text._produce_hidden` plus `lm_head` — which is exactly a training forward
pass. So the loop is written directly against that, and
`peft.inject_adapter_in_model` adds the LoRA layers to the text tower in place.

### The sequence layout

```
[bos] [729 image patch embeddings] [query prefix] [question] [query suffix] [answer] [eos]
<-------------- loss masked out (-100) -----------------------------------> <-- loss -->
```

Labels are set to `-100` for everything except the answer tokens. `-100` is
PyTorch's "ignore this position" marker in cross-entropy. So the model is only
graded on producing the right answer, never on reproducing the image or the
question. The vision encoder runs under `torch.no_grad()` — only the text LoRA
trains.

### Three bugs that were found and fixed, all documented

1. **The attention mask.** Moondream's own `_produce_hidden` builds its mask
   with `torch.zeros(...)` on the CPU, which crashes
   `scaled_dot_product_attention` the moment the model is on a GPU. This file
   rebuilds the same mask on the right device. The mask is bidirectional across
   the 729 image tokens (every patch can see every other patch — an image has
   no left-to-right order) and causal after that (each text token sees only
   what came before it).

2. **The wrong lm_head.** `lm_head` keeps only the last position, because it is
   the generation path. Training needs logits for the whole sequence, which is
   `_lm_head`. Using the wrong one produces a loss that never moves, with no
   error.

3. **Where the LoRA goes.** The target modules were confirmed against
   `model.named_modules()`, not guessed. Moondream's text blocks are
   `blocks.N.attn.qkv` and `blocks.N.attn.proj` — *not* the `q_proj`/`v_proj`
   you would target on a Llama. Guessing gives you zero injected adapters and a
   run that trains nothing.

### Settings

Rank 8, alpha 16, dropout 0.05, learning rate 2e-4, gradient accumulation 8
(process 8 samples, then take one optimiser step — this simulates a batch of 8
on a GPU that only fits 1), gradient clipping at norm 1.0. LoRA parameters are
kept in float32 even though the model is float16, so small updates are not lost
to rounding.

It fits an RTX 4060 8 GB at batch size 1, peaks around 5.7 GB, and takes about
12 minutes for 2 epochs. Kaggle's T4×2 is the fallback and was not needed.

### The holdout split

The last 20 patch names (sorted) are held out and never trained on. That is what
makes the before/after comparison honest — otherwise you would be testing on
data the model memorised.

### `apply_adapter()`

Used at inference time by `vlm.py`. Rebuilds the LoRA config from the saved
JSON, injects it, loads the saved weights, and — the useful part — accepts an
`alpha` override. Since LoRA's effect scales as alpha/rank, this dials the
adaptation up or down at load time without retraining.

**Self-check:** `python -m training.lora_finetune --smoke` runs 4 training steps
on a single sample and asserts the loss goes **down**. That single assertion
catches both silent failures above: a wrong `lm_head` and a detached graph from
freezing the wrong parameters both produce a flat loss.

---

## 19. `tests/` — every check and what it proves

| File | Costs | Proves |
|---|---|---|
| `test_pipeline.py` | nothing — no GPU, no LLM quota | The whole HTTP path: health, upload → validate → route → tool → JSON → PDF. It deliberately uses only change analysis because that tool is model-free. Also asserts a mismatched pair returns 400 with "Dimension mismatch", and a `.txt` upload returns 400 not 500. |
| `test_matrix.py` | GPU time + LLM quota | The build report's Section 13 matrix, one row per mandatory requirement, sent exactly as a judge would send it. Checks the agent picked the **right tool** for each and that the PDF renders. This is the pre-demo check. |
| `test_domain_adaptation.py` | GPU time | Measures the LoRA rather than asserting it. Asks the model to name land cover in each of the 20 held-out patches and scores the CORINE labels it names against the truth, as F1. Run twice — once with `SATQUERY_DISABLE_LORA=1`, once without — to get the before/after pair. |
| `test_llm_client.py` | nothing | Tool-call normalisation, including Gemini returning a dict instead of a JSON string, and malformed JSON becoming `{_unparsed: ...}` instead of a crash. |
| `check_providers.py` | a little LLM quota | Each of the three brains actually returns a well-formed tool call. Run when keys change or a provider starts misbehaving. It reports **all** providers rather than stopping at the first failure. |

Plus every source file has a `demo()` under `if __name__ == "__main__"`, listed
in the README's Checks section. That is the project's testing convention: one
runnable check per file, using plain `assert`, no test framework.

**The measured domain adaptation result:** label-set F1 goes from **0.000 to
0.387** on 20 held-out patches. The base model scores zero not because it cannot
see the scene but because it answers in everyday words ("grass and dirt") rather
than in the CORINE vocabulary the benchmark is written in. That gap *is* the
adaptation.

---

## 20. Config files, requirements, environment

**`requirements.txt`** — fastapi, uvicorn, python-multipart, pillow, numpy,
scikit-image, rasterio, transformers (pinned to 4.53.2), accelerate,
bitsandbytes, peft, openai, python-dotenv, datasets, pandas, huggingface_hub,
fpdf2.

Three pins with reasons written in as comments:

- `transformers==4.53.2` — Moondream2's remote code was written against the 4.x
  API and dies on transformers 5 (`all_tied_weights_keys`). Unpin only when
  Moondream ships 5.x-compatible code.
- one `openai` client for all three brains — one response shape to normalise
  instead of three SDKs.
- `fpdf2` instead of weasyprint — no GTK system libraries needed on Windows.

`torch` is **not** in requirements.txt. It is installed separately from the CUDA
index: `pip install torch --index-url https://download.pytorch.org/whl/cu124`.

**`backend/.env`** (never committed) holds `GROQ_API_KEY`, `MISTRAL_API_KEY`,
`GEMINI_API_KEY`, `HF_TOKEN`, `AGENT_PROVIDER_ORDER`. `.env.example` is the
template.

**Model weights:**

```python
snapshot_download('vikhyatk/moondream2', revision='2025-06-21',
                  local_dir='weights/moondream2')
```

`local_dir` matters — the default HF cache uses symlinks, which Windows refuses
without Developer Mode enabled.

**Running it:**

```bash
venv/Scripts/python.exe -m uvicorn backend.main:app --reload
```

Then open http://127.0.0.1:8000/ . API docs at `/docs`.

---

## 21. The honest limitations

Know these before a judge asks. Every one of them is written into the source
where it applies.

1. **Change detection is unsupervised.** IoU roughly 0.1–0.25 against LEVIR-CD
   ground truth. It under-reports large developments whose new roofs match the
   surrounding brightness. Upgrade: a siamese U-Net trained on LEVIR-CD would
   reach roughly IoU 0.7.

2. **Land cover uses index thresholds, not a segmentation network.** Without a
   near-infrared band, vegetation is an RGB proxy and water is **not reported at
   all** — dark water, dark roofs and tree shadow are indistinguishable in
   visible light.

3. **No ground resolution means no hectares.** The tool says so rather than
   inventing a scale.

4. **The fusion thresholds are fixed constants**, tuned to Sentinel-1/2 ranges.
   A different sensor needs recalibration. They are module-level constants
   precisely so that is a one-line change.

5. **Grounding saturates at 50 objects.** A count of exactly 50 means "at least
   50" and the system says so.

6. **The LoRA alpha is a trade-off.** Higher alpha gives better CORINE
   vocabulary and worse general captioning. 8 is the compromise. The real fix
   is more varied training data.

7. **CORS is fully open** and there is no authentication. Hackathon setting,
   not a production one.

8. **Free-tier LLM quota is finite.** The three-provider failover plus the
   keyword fallback router exists so an outage or a rate limit cannot end the
   demo, but the fallback router is noticeably dumber and docks the confidence
   score accordingly.

---

## The one thing to remember

Every number this system reports comes from deterministic pixel maths that a
person can re-run and audit. The language model chooses which maths to run and
writes the sentence around the result. It never touches a pixel and never
produces a figure. That separation is the whole design, and it is what makes
the output defensible.
