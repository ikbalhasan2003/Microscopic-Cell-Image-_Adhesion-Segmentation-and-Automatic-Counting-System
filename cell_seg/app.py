"""Gradio UI entry point."""
import dataclasses
import html
from datetime import datetime
from typing import List, Tuple

import gradio as gr

from config import OUTPUTS_DIR, SAMPLES_DIR, Params
from core.pipeline import CellCountingPipeline
from utils.device import device_label
from utils.logger import get_logger

TITLE = "Cell Segmentation & Counting"
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
NO_IMAGE_MSG = "Upload an image or pick a sample, then run segmentation."
NO_CELLS_MSG = "No cells found. Lower the probability threshold or the minimum cell area."
DEFAULTS = Params()
# Slider groups in UI order: (group title, open?, [(field, label, min, max, step, help)]).
# blur_k stays at its default (not in mockup).
SLIDER_GROUPS = [
    ("Detection", False, [
        ("thresh", "Probability threshold τ", 0.05, 0.95, 0.01,
         "Pixels above τ count as cell. Lower it for faint cells."),
        ("min_area", "Minimum cell area (px²)", 0, 1000, 10,
         "Objects smaller than this are discarded as debris."),
    ]),
    ("Splitting touching cells", False, [
        ("peak_ratio", "Peak ratio r", 0.1, 0.9, 0.05,
         "Lower values split clumps into more cells."),
        ("min_dist", "Minimum peak distance (px)", 1, 50, 1,
         "Roughly the radius of the smallest cell."),
    ]),
    ("Image cleanup", False, [
        ("clip", "Contrast boost (CLAHE clip)", 0.5, 8.0, 0.1,
         "Raise for dim or low-contrast images."),
        ("kernel", "Smoothing kernel size", 1, 15, 2,
         "Larger values fill holes and remove specks in the mask."),
    ]),
]
SLIDERS = [s for _, _, group in SLIDER_GROUPS for s in group]
INT_FIELDS = {"kernel", "min_dist", "min_area"}
N_OUTPUTS = 9  # 5 images, plot, summary, csv, status
OUT_H = 300  # px; equal panel height keeps the stage grid aligned

_logger = get_logger("app")

THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#F3F1FA", c100="#E4DFF4", c200="#C9BFE9", c300="#A898D9", c400="#8571C6",
        c500="#6650B0", c600="#4B3B8F", c700="#3C2F73", c800="#2E2458", c900="#21193F",
        c950="#160F2C"),
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_md,
    font=[gr.themes.GoogleFont("Instrument Sans"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#F4F5F9",
    body_background_fill_dark="#14151C",
    block_border_width="1px",
    block_shadow="none",
    block_label_text_weight="500",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_primary_text_color="#FFFFFF",
    slider_color="*primary_600",
)

CSS = """
:root { --hx: #4B3B8F; --eo: #D9467A; --ink: #1E2233; --muted: #5C6378; --line: #DCDFEA; }
.dark { --hx: #A898D9; --eo: #F07AA3; --ink: #E8E9F0; --muted: #9CA3B8; --line: #2C2F3D; }
.gradio-container { max-width: 1440px !important; margin: 0 auto !important; }
#app-header { display: flex; align-items: flex-end; justify-content: space-between; gap: 16px;
  flex-wrap: wrap; padding: 8px 4px 18px; border-bottom: 1px solid var(--line); margin-bottom: 8px; }
#app-header h1 { margin: 0; font-size: 1.75rem; font-weight: 600; letter-spacing: -0.01em;
  color: var(--ink); }
#app-header p { margin: 4px 0 0; color: var(--muted); font-size: 0.95rem; max-width: 64ch; }
.device-chip { display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px;
  border: 1px solid var(--line); border-radius: 999px; font-size: 0.85rem; color: var(--muted); }
.device-chip::before { content: ""; width: 8px; height: 8px; border-radius: 50%;
  background: var(--hx); }
.step-title { display: flex; align-items: center; gap: 10px; margin: 6px 0 2px;
  font-weight: 600; font-size: 1rem; color: var(--ink); }
.step-title span { display: inline-grid; place-items: center; width: 24px; height: 24px;
  border-radius: 50%; background: var(--hx); color: #fff; font-size: 0.8rem; }
.dark .step-title span { color: #14151C; }
#sidebar { gap: 10px; }
#summary { display: grid; grid-template-columns: 1.4fr 1fr 1fr; gap: 0;
  border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
#summary .cell { padding: 14px 18px; border-left: 1px solid var(--line); }
#summary .cell:first-child { border-left: 0; }
#summary .k { font-size: 0.85rem; color: var(--muted); }
#summary .v { font-size: 1.5rem; font-weight: 600; color: var(--ink);
  font-variant-numeric: tabular-nums; margin-top: 2px; }
#summary .cell:first-child .v { font-size: 2.6rem; line-height: 1.05; color: var(--hx); }
#summary .v small { font-size: 0.85rem; font-weight: 400; color: var(--muted); margin-left: 4px; }
#summary.empty .v { color: var(--muted); }
.status { padding: 10px 14px; border-radius: 8px; border-left: 4px solid var(--hx);
  background: color-mix(in srgb, var(--hx) 8%, transparent); color: var(--ink);
  font-size: 0.92rem; }
.status.warn { border-left-color: #C98A12; background: color-mix(in srgb, #C98A12 10%, transparent); }
.status.error { border-left-color: var(--eo); background: color-mix(in srgb, var(--eo) 10%, transparent); }
.status ul { margin: 6px 0 0; padding-left: 18px; color: var(--muted); }
#run-btn { font-size: 1rem; }
.hint { margin: -4px 0 4px; color: var(--muted); font-size: 0.88rem; }
@media (max-width: 720px) {
  #summary { grid-template-columns: 1fr 1fr; }
  #summary .cell:first-child { grid-column: 1 / -1; border-bottom: 1px solid var(--line); }
  #summary .cell:nth-child(2) { border-left: 0; }
}
"""


def _sample_images() -> List[List[str]]:
    if not SAMPLES_DIR.is_dir():
        return []
    return [[str(p)] for p in sorted(SAMPLES_DIR.iterdir()) if p.suffix.lower() in IMAGE_EXT]


def _out_image(label: str, height: int = OUT_H) -> gr.Image:
    return gr.Image(label=label, format="png", interactive=False, height=height,
                    buttons=["download", "fullscreen"])


def _step(n: int, text: str) -> gr.HTML:
    return gr.HTML(f'<div class="step-title"><span>{n}</span>{text}</div>')


def _summary_html(count=None, mean=None, median=None) -> str:
    if count is None:
        vals, cls = ("–", "–", "–"), ' class="empty"'
    else:
        vals = (str(count), f"{mean:.1f}<small>px²</small>", f"{median:.1f}<small>px²</small>")
        cls = ""
    keys = ("Cells detected", "Mean area", "Median area")
    cells = "".join(f'<div class="cell"><div class="k">{k}</div><div class="v">{v}</div></div>'
                    for k, v in zip(keys, vals))
    return f'<div id="summary"{cls}>{cells}</div>'


def _status_html(text: str, kind: str = "", notes: List[str] = ()) -> str:
    items = "".join(f"<li>{html.escape(n)}</li>" for n in notes)
    extra = f"<ul>{items}</ul>" if items else ""
    return f'<div class="status {kind}">{html.escape(text)}{extra}</div>'


def _empty(status: str) -> Tuple:
    return (None,) * (N_OUTPUTS - 3) + (_summary_html(), None, status)


def build_ui() -> gr.Blocks:
    pipeline = CellCountingPipeline()
    device_text = device_label(pipeline.device)

    def segment(img, *values):
        if img is None:
            return _empty(_status_html(NO_IMAGE_MSG, "warn"))
        kw = {name: (int(v) if name in INT_FIELDS else float(v))
              for (name, *_), v in zip(SLIDERS, values)}
        try:
            params = dataclasses.replace(DEFAULTS, **kw)
            res = pipeline.run(img, params)
            stats = pipeline.analyzer.summary(res.cells)
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            stamp = f"{datetime.now():%Y%m%d_%H%M%S_%f}"[:-3]  # ms: no clash on quick reruns
            csv_path = OUTPUTS_DIR / f"{stamp}_cells.csv"
            pipeline.analyzer.to_dataframe(res.cells).to_csv(csv_path, index=False)
        except FileNotFoundError as e:
            return _empty(_status_html(str(e), "error"))
        except ValueError as e:
            return _empty(_status_html(f"Invalid input: {e}", "error"))
        except Exception as e:  # app must never crash (architecture p.6)
            _logger.exception("Segmentation failed")
            return _empty(_status_html(
                f"Segmentation failed ({type(e).__name__}). Details are in outputs/app.log.",
                "error"))
        if res.count == 0:
            status = _status_html(NO_CELLS_MSG, "warn", res.warnings)
        else:
            status = _status_html(f"Segmented {res.count} cells. Per-cell measurements "
                                  "are ready to download as CSV.", "", res.warnings)
        im = res.images
        return (im["overlay"], pipeline.analyzer.histogram(res.cells),
                im["original"], im["prob"], im["mask"], im["distance"],
                _summary_html(res.count, stats["mean_area"], stats["median_area"]),
                str(csv_path), status)

    def reset():
        return tuple(getattr(DEFAULTS, name) for name, *_ in SLIDERS) + (None,) + _empty(_status_html(NO_IMAGE_MSG))

    with gr.Blocks(title=TITLE, fill_width=True) as demo:
        gr.HTML(
            f'<header id="app-header"><div><h1>{TITLE}</h1>'
            "<p>Count and measure cells in microscopy images, including cells that touch. "
            "A U-Net finds cell pixels; watershed splits clumps into individual cells.</p></div>"
            f'<div class="device-chip">Running on {html.escape(device_text)}</div></header>')
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=320, elem_id="sidebar"):
                _step(1, "Choose an image")
                inp = gr.Image(type="numpy", label="Microscopy image (PNG, JPG, TIF)",
                               sources=["upload", "clipboard"], height=240)
                examples = _sample_images()
                if examples:
                    gr.Examples(examples, inputs=inp, label="Or try a sample",
                                cache_examples=False, examples_per_page=6)
                _step(2, "Segment")
                with gr.Row():
                    run_btn = gr.Button("Segment cells", variant="primary", scale=3,
                                        min_width=0, elem_id="run-btn")
                    reset_btn = gr.Button("Reset", variant="secondary", scale=1, min_width=0)
                _step(3, "Fine-tune (optional)")
                gr.HTML('<p class="hint">Defaults suit most images. Adjust, then segment again.</p>')
                sliders = []
                for title, is_open, group in SLIDER_GROUPS:
                    with gr.Accordion(title, open=is_open):
                        sliders += [gr.Slider(lo, hi, value=getattr(DEFAULTS, name), step=step,
                                              label=label, info=info)
                                    for name, label, lo, hi, step, info in group]
            with gr.Column(scale=3):
                summary = gr.HTML(_summary_html())
                status = gr.HTML(_status_html(NO_IMAGE_MSG))
                with gr.Tabs():
                    with gr.Tab("Result"):
                        with gr.Row(equal_height=True):
                            out_over = _out_image("Segmented cells (numbered)", height=460)
                            with gr.Column():
                                out_hist = gr.Plot(label="Cell area distribution")
                                out_csv = gr.File(label="Per-cell measurements (CSV)",
                                                  interactive=False)
                    with gr.Tab("Pipeline stages"):
                        with gr.Row():
                            out_orig = _out_image("Original image")
                            out_prob = _out_image("Cell probability (U-Net)")
                        with gr.Row():
                            out_mask = _out_image("Binary mask")
                            out_dist = _out_image("Distance map with seeds")

        outputs = [out_over, out_hist, out_orig, out_prob, out_mask, out_dist,
                   summary, out_csv, status]
        run_btn.click(segment, inputs=[inp, *sliders], outputs=outputs)
        reset_btn.click(reset, outputs=[*sliders, inp, *outputs])
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860, theme=THEME, css=CSS)
