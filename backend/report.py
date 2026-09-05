"""One-page PDF analysis report per job.

ponytail: fpdf2 draws the page directly instead of the report's Jinja2 -> HTML
-> weasyprint chain. weasyprint needs GTK system libraries on Windows and the
report is a title, a few fields, a numbered trace and an image — no CSS layout
worth two extra dependencies. Swap back only if the layout gets real.
"""
from pathlib import Path

from fpdf import FPDF

MARGIN = 15


def _text(value):
    """fpdf2's core fonts are latin-1 only; keep the em-dashes out of the PDF."""
    return (
        str(value)
        .replace("—", "-").replace("–", "-")
        .replace("‘", "'").replace("’", "'")
        .replace("“", '"').replace("”", '"')
        .encode("latin-1", "replace").decode("latin-1")
    )


def build_report(result, out_path):
    """`result` is the controller's dict. Returns the written path."""
    pdf = FPDF()
    pdf.set_margins(MARGIN, MARGIN, MARGIN)
    pdf.set_auto_page_break(True, margin=MARGIN)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "SatQuery AI - Analysis Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(110, 110, 110)
    pdf.cell(0, 5, _text(f"Job {result['job_id']}  |  {result.get('elapsed_seconds', 0)}s  |  "
                         f"brain: {result.get('llm_provider')}"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    for label, value in (
        ("Query", result.get("query")),
        ("Task classified", result.get("task_classified")),
        ("Answer", result.get("answer")),
        ("Confidence", f"{result.get('confidence')}"
         + (f"  ({'; '.join(result['confidence_reasons'])})"
            if result.get("confidence_reasons") else "")),
    ):
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, _text(label), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 5.5, _text(value if value is not None else "-"),
                       new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Execution trace", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for step in result.get("execution_trace", []):
        pdf.multi_cell(0, 5, _text(f"{step['step']}. {step['tool']}  {step.get('parameters', {})}"),
                       new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(70, 70, 70)
        pdf.multi_cell(0, 5, _text(f"    -> {step['output_summary']}"),
                       new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(1)

    evidence = result.get("visual_evidence") or {}
    images = [p for p in (evidence.get("overlay_png_url"), evidence.get("change_mask_url"))
              if p and Path(p).is_file()]
    if images:
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Visual evidence", new_x="LMARGIN", new_y="NEXT")
        width = (pdf.w - 2 * MARGIN - 5) / 2 if len(images) > 1 else pdf.w - 2 * MARGIN
        x = MARGIN
        top = pdf.get_y()
        for path in images[:2]:
            pdf.image(path, x=x, y=top, w=width)
            x += width + 5

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out_path))
    return str(out_path)


def demo():
    import tempfile

    result = {
        "job_id": "demo123",
        "query": "What changed between these two dates?",
        "answer": "About 1.4% of the scene changed — new buildings in the northern eastern quadrant.",
        "confidence": 0.85,
        "confidence_reasons": [],
        "task_classified": "run_change_analysis",
        "llm_provider": "groq",
        "elapsed_seconds": 3.1,
        "execution_trace": [
            {"step": 1, "tool": "run_change_analysis",
             "parameters": {"image_id_t1": "image1", "image_id_t2": "image2"},
             "output_summary": "1.4% of the scene changed across 12 regions."},
        ],
        "visual_evidence": {"overlay_png_url": None, "change_mask_url": None, "bbox": None},
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(build_report(result, Path(tmp) / "report.pdf"))
        assert path.is_file() and path.stat().st_size > 800, path.stat().st_size
        assert path.read_bytes().startswith(b"%PDF"), "not a PDF"

        # With an image attached, the PDF must be meaningfully larger.
        from PIL import Image
        import numpy as np

        png = Path(tmp) / "overlay.png"
        Image.fromarray((np.random.rand(128, 128, 3) * 255).astype("uint8")).save(png)
        result["visual_evidence"]["overlay_png_url"] = str(png)
        with_image = Path(build_report(result, Path(tmp) / "report2.pdf"))
        assert with_image.stat().st_size > path.stat().st_size, "image not embedded"

    print("report: valid PDF written, with and without visual evidence")


if __name__ == "__main__":
    demo()
