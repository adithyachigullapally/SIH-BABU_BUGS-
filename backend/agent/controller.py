"""The agentic loop: classify the query into a tool, run it, narrate the result.

The brain never sees pixels. It sees the validator's profiles (sizes, band
counts, modality, whether the pair is bi-temporal or cross-modal) and the
user's query, picks one of the five tools, and then turns that tool's
measured output into an answer. Keeping the measurement deterministic and the
language model purely on classification + narration is what makes the numbers
in the report defensible.

Every step lands in an execution trace, including which provider answered.
"""
import json
import time
import uuid
from pathlib import Path

from backend.agent.llm_client import call_llm_with_tools
from backend.agent.tool_schemas import OPENAI_TOOLS, TOOL_NAMES

MAX_STEPS = 3

SYSTEM_PROMPT = """You are SatQuery AI, an orchestrator for remote-sensing image analysis.

Pick the tool(s) that answer the user's query, call them, then write the final
answer from their output. Rules:
- run_change_analysis only when pair_type is bi_temporal.
- run_fusion_analysis only when pair_type is cross_modal.
- With one image: run_land_cover when the user asks HOW MUCH / what area / what
  percentage / how many hectares; run_grounding when the user asks HOW MANY of
  a countable object (houses, buildings, ponds, vehicles) or asks to locate or
  highlight something; run_caption for "what is this / describe it"; run_vqa
  only for a descriptive question containing no quantity at all.
- run_vqa cannot count and cannot measure. If an answer needs a number, a
  measuring tool has to produce it. Never route a "how many" or "how much"
  question to run_vqa.
- A query that asks for several different things (a count AND an area, or a
  description AND a quantity) needs one tool call per thing, up to three. Call
  them one after another, then write a single answer from all their outputs.
- Use the exact image ids given. Never invent measurements or counts: every
  number in your answer must appear in a tool's output, and if no tool measured
  something, say plainly that it was not measured rather than estimating.
  Answer in 2-4 sentences, no preamble."""


def _inventory(profiles):
    """The facts the brain classifies from, as compact JSON (free tiers are TPM-bound)."""
    facts = {"images": {}, "pair_type": profiles["pair"]["pair_type"]}
    for key, prof in (("image1", profiles["profile1"]), ("image2", profiles["profile2"])):
        if prof:
            facts["images"][key] = {
                "size": f"{prof['width']}x{prof['height']}",
                "bands": prof["band_count"],
                "modality": prof["modality"],
            }
    if not profiles["pair"]["compatible"]:
        facts["pair_problem"] = profiles["pair"]["reason"]
    return json.dumps(facts, separators=(",", ":"))


def _resolve(image_id, images):
    if image_id not in images or images[image_id] is None:
        raise ValueError(f"No image called '{image_id}'. Available: {sorted(k for k, v in images.items() if v)}")
    return images[image_id]


def run_tool(name, args, images, out_dir, job_id, resolution_m=None):
    """Dispatch a tool call. Returns (result_dict, evidence_dict)."""
    from backend.tools import change_analysis, fusion_analysis, land_cover, vlm

    if name == "run_land_cover":
        result = land_cover.analyze_land_cover(
            _resolve(args["image_id"], images),
            resolution_m=resolution_m,
            out_dir=out_dir,
            job_id=job_id,
        )
        return result, {"overlay_png_url": result["overlay_url"]}

    if name == "run_vqa":
        result = vlm.answer_question(_resolve(args["image_id"], images), args["question"])
        return result, {}

    if name == "run_caption":
        return vlm.caption(_resolve(args["image_id"], images)), {}

    if name == "run_grounding":
        path = _resolve(args["image_id"], images)
        result = vlm.ground(path, args["referring_expression"])
        evidence = {}
        if result["boxes"]:
            evidence["bbox"] = result["boxes"][0]
            evidence["overlay_png_url"] = vlm.draw_boxes(
                path, result["boxes"], Path(out_dir) / f"{job_id}_boxes.png"
            )
        return result, evidence

    if name == "run_change_analysis":
        result = change_analysis.detect_change(
            _resolve(args["image_id_t1"], images),
            _resolve(args["image_id_t2"], images),
            out_dir=out_dir,
            job_id=job_id,
        )
        return result, {
            "change_mask_url": result["mask_url"],
            "overlay_png_url": result["overlay_url"],
            "bbox": result["hotspots"][0]["bbox"] if result["hotspots"] else None,
        }

    if name == "run_fusion_analysis":
        result = fusion_analysis.analyze_fusion(
            _resolve(args["optical_id"], images),
            _resolve(args["sar_id"], images),
            out_dir=out_dir,
            job_id=job_id,
        )
        return result, {"overlay_png_url": result["overlay_url"]}

    raise ValueError(f"Unknown tool '{name}'. Known: {TOOL_NAMES}")


def _summarize(name, result):
    """One line per tool for the execution trace and the PDF."""
    if "summary" in result:
        return result["summary"]
    if name == "run_caption":
        return result["caption"]
    if name == "run_vqa":
        return result["answer"]
    if name == "run_grounding":
        at_least = "at least " if result.get("saturated") else ""
        return f"{at_least}{result['count']} region(s) matching '{result['expression']}'"
    return json.dumps(result)[:300]


def _confidence(tool, result, profiles, used_fallback):
    """An explainable number, not a vibe: named signals, each with a stated weight."""
    score, reasons = 0.85, []
    if used_fallback:
        score -= 0.15
        reasons.append("tool chosen by keyword fallback, no LLM available")
    if not profiles["pair"]["compatible"]:
        score -= 0.35
        reasons.append("input pair failed compatibility checks")
    if tool == "run_grounding" and result.get("saturated"):
        score -= 0.15
        reasons.append("detection hit its 50-object limit, so the count is a floor")
    if tool == "run_grounding" and not result.get("boxes"):
        score -= 0.35
        reasons.append("no region matched the description")
    if tool == "run_change_analysis" and result.get("change_percent", 0) < 0.05:
        score -= 0.10
        reasons.append("change is at or below the noise floor")
    if tool == "run_land_cover" and not result.get("resolution_m"):
        score -= 0.10
        reasons.append("no ground resolution, so percentages only and no areas")
    if tool in ("run_land_cover", "run_fusion_analysis") and not result.get("has_nir"):
        score -= 0.15
        reasons.append("no near-infrared band, vegetation index is a proxy")
    if tool in ("run_vqa", "run_caption"):
        score -= 0.05
        reasons.append("free-text model output, not a measurement")
    return round(max(score, 0.1), 2), reasons


def _fallback_tool(query, profiles):
    """Keyword routing for when every LLM provider is down mid-demo.

    Not a substitute for the brain — it exists because the build's own
    constraint is that no single API outage can take out the demo.
    """
    q = (query or "").lower()
    pair_type = profiles["pair"]["pair_type"]
    if pair_type == "bi_temporal":
        return "run_change_analysis", {
            "image_id_t1": "image1", "image_id_t2": "image2", "question": query
        }
    if pair_type == "cross_modal":
        optical = "image1" if profiles["profile1"]["modality"] == "optical" else "image2"
        sar = "image2" if optical == "image1" else "image1"
        return "run_fusion_analysis", {"optical_id": optical, "sar_id": sar, "question": query}
    if any(w in q for w in ("locate", "where", "highlight", "find", "show me", "bounding box")):
        expression = q
        for lead in ("highlight the ", "locate the ", "find the ", "where is the ", "where is "):
            if lead in q:
                expression = q.split(lead, 1)[1].strip(" ?.")
                break
        return "run_grounding", {"image_id": "image1", "referring_expression": expression}
    # A quantity word alone is not enough: "how many buildings" is a counting
    # question for the VLM, while "how much forest" is a land-cover measurement.
    # Both a quantity word and a land-cover subject have to be present.
    asks_quantity = any(w in q for w in ("how much", "how many", "what area", "percent",
                                         "hectare", "coverage", "how large", "area of"))
    about_land_cover = any(w in q for w in ("tree", "forest", "vegetation", "green",
                                            "water", "lake", "river", "built", "urban",
                                            "land cover", "landcover", "crop", "field",
                                            "bare", "soil"))
    if asks_quantity and about_land_cover:
        return "run_land_cover", {"image_id": "image1"}
    if any(w in q for w in ("describe", "caption", "what is this", "what does this show")):
        return "run_caption", {"image_id": "image1"}
    return "run_vqa", {"image_id": "image1", "question": query}


def analyze(query, images, profiles, out_dir, job_id=None, resolution_m=None):
    """Run one query end to end. `images` maps 'image1'/'image2' -> path.

    `resolution_m` is metres per pixel, when the user knows it — the only way a
    PNG or JPEG can be turned into real areas rather than percentages.
    """
    job_id = job_id or uuid.uuid4().hex[:12]
    started = time.time()
    trace = []
    evidence = {"bbox": None, "overlay_png_url": None, "change_mask_url": None}

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Inputs: {_inventory(profiles)}\nQuery: {query}"},
    ]

    provider, used_fallback, tool_name, tool_result = None, False, None, None
    answer = None

    try:
        response, provider = call_llm_with_tools(messages, OPENAI_TOOLS)
    except RuntimeError as exc:
        used_fallback, provider = True, "keyword-fallback"
        trace.append({"step": len(trace) + 1, "tool": "llm_failover",
                      "parameters": {}, "output_summary": f"all providers failed: {exc}"[:300]})
        response = None

    if used_fallback:
        tool_name, tool_args = _fallback_tool(query, profiles)
        tool_result, tool_evidence = run_tool(
            tool_name, tool_args, images, out_dir, job_id, resolution_m
        )
        evidence.update({k: v for k, v in tool_evidence.items() if v})
        trace.append({"step": len(trace) + 1, "tool": tool_name, "parameters": tool_args,
                      "output_summary": _summarize(tool_name, tool_result)})
        answer = _summarize(tool_name, tool_result)
    else:
        for _ in range(MAX_STEPS):
            calls = response["tool_calls"]
            if not calls:
                answer = response["text"]
                break
            messages.append({
                "role": "assistant",
                "tool_calls": [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                    for c in calls
                ],
            })
            for call in calls:
                try:
                    tool_result, tool_evidence = run_tool(
                        call["name"], call["arguments"], images, out_dir, job_id, resolution_m
                    )
                    tool_name = call["name"]
                    evidence.update({k: v for k, v in tool_evidence.items() if v})
                    summary = _summarize(call["name"], tool_result)
                except Exception as exc:  # noqa: BLE001 — hand the error back to the brain
                    summary = f"ERROR: {type(exc).__name__}: {exc}"
                trace.append({"step": len(trace) + 1, "tool": call["name"],
                              "parameters": call["arguments"], "output_summary": summary})
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": summary})
            try:
                response, provider = call_llm_with_tools(messages, OPENAI_TOOLS)
            except RuntimeError:
                answer = _summarize(tool_name, tool_result) if tool_result else None
                break
        if answer is None and response is not None:
            # Every round went on tool calls, so the brain never reached a turn
            # to write prose. Handing back the last tool's raw summary ("5
            # region(s) matching 'water body'") loses the other tools' findings
            # entirely, so ask once more with tools switched off.
            if not response["tool_calls"] and response["text"]:
                answer = response["text"]
            else:
                try:
                    final, provider = call_llm_with_tools(
                        messages, OPENAI_TOOLS, tool_choice="none")
                    answer = final["text"] or None
                except RuntimeError:
                    answer = None
        if answer is None:
            answer = _summarize(tool_name, tool_result) if tool_result else "No answer produced."

    confidence, reasons = _confidence(tool_name, tool_result or {}, profiles, used_fallback)
    return {
        "job_id": job_id,
        "answer": answer,
        "confidence": confidence,
        "confidence_reasons": reasons,
        "task_classified": tool_name,
        "llm_provider": provider,
        "execution_trace": trace,
        "visual_evidence": evidence,
        "elapsed_seconds": round(time.time() - started, 2),
        "query": query,
    }


def demo():
    """Offline check of routing and trace shape — no network, no GPU."""
    from backend.validator import profile_inputs

    root = Path(__file__).resolve().parents[2] / "sample_data" / "levir_cd"
    a, b = root / "A" / "test_1.png", root / "B" / "test_1.png"
    images = {"image1": a, "image2": b}

    bi = profile_inputs(a, b)
    assert _fallback_tool("what changed here?", bi)[0] == "run_change_analysis"

    cross = profile_inputs(a, b, modality2="sar")
    tool, args = _fallback_tool("is there water?", cross)
    assert tool == "run_fusion_analysis" and args["sar_id"] == "image2", (tool, args)

    single = profile_inputs(a)
    assert _fallback_tool("describe this scene", single)[0] == "run_caption"
    assert _fallback_tool("how many buildings are there?", single)[0] == "run_vqa"
    assert _fallback_tool("how much of this is trees?", single)[0] == "run_land_cover"
    assert _fallback_tool("what area of water is present?", single)[0] == "run_land_cover"
    # Describing and quantifying at once is still a measurement.
    assert _fallback_tool(
        "describe the scene and how much area of trees is present", single
    )[0] == "run_land_cover"
    tool, args = _fallback_tool("highlight the water body", single)
    assert tool == "run_grounding" and args["referring_expression"] == "water body", args

    # Full offline path: no LLM needed, change tool runs for real.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tool, args = _fallback_tool("what changed?", bi)
        result, ev = run_tool(tool, args, images, tmp, "demo")
        assert result["change_percent"] >= 0 and Path(ev["change_mask_url"]).is_file()

    conf, reasons = _confidence("run_grounding", {"boxes": []}, single, used_fallback=True)
    assert conf < 0.5 and len(reasons) == 2, (conf, reasons)

    print("controller: routing, dispatch and confidence checks passed")


if __name__ == "__main__":
    demo()
