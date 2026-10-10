"""Gradio UI entry point."""
import dataclasses
import html
from datetime import datetime
from typing import Dict, List, Tuple

import gradio as gr

from config import OUTPUTS_DIR, SAMPLES_DIR, Params
from core.pipeline import CellCountingPipeline
from utils.device import device_label
from utils.logger import get_logger

TITLE = "Cell Segmentation & Counting"
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
NO_IMAGE_MSG = "Upload an image or pick a sample, then press Segment cells."
NO_CELLS_MSG = "No cells found. Try the “Faint / low contrast” preset or lower τ and the minimum area."
DEFAULTS = Params()
# Parameter groups in UI order: (group title, [(field, label, min, max, step, help)]).
# blur_k stays at its default (not in mockup).
PARAM_GROUPS = [
    ("Detection", [
        ("thresh", "Probability threshold τ", 0.05, 0.95, 0.01,
         "Pixels above τ count as cell. Lower it for faint cells."),
        ("min_area", "Minimum cell area (px²)", 0, 1000, 10,
         "Smaller objects are discarded as debris."),
    ]),
    ("Splitting touching cells", [
        ("peak_ratio", "Peak ratio r", 0.1, 0.9, 0.05,
         "Lower values split clumps into more cells."),
        ("min_dist", "Minimum peak distance (px)", 1, 50, 1,
         "Roughly the radius of the smallest cell."),
    ]),
    ("Image cleanup", [
        ("clip", "Contrast boost (CLAHE clip)", 0.5, 8.0, 0.1,
         "Raise for dim or low-contrast images."),
        ("kernel", "Smoothing kernel size", 1, 15, 2,
         "Larger values fill holes and remove specks."),
    ]),
]
SLIDERS = [s for _, group in PARAM_GROUPS for s in group]
INT_FIELDS = {"kernel", "min_dist", "min_area"}
CUSTOM = "Custom"
# Preset name -> (one-line description, overrides of DEFAULTS). Starting points, not tuned.
PRESETS: Dict[str, Tuple[str, Dict[str, float]]] = {
    "Balanced (default)": ("Tuned on the validation split. Start here.", {}),
    "Faint / low contrast": ("Stronger contrast boost and a lower threshold.",
                             {"clip": 4.0, "thresh": 0.35}),
    "Dense, touching clumps": ("Splits clumps more aggressively.",
                               {"peak_ratio": 0.3, "min_dist": 6}),
    "Sparse, large cells": ("Fewer splits; ignores small fragments.",
                            {"peak_ratio": 0.5, "min_dist": 20, "min_area": 100, "kernel": 5}),
    "Noisy / debris": ("Heavier smoothing and a larger minimum area.",
                       {"thresh": 0.55, "kernel": 5, "min_area": 80}),
}
DEFAULT_PRESET = next(iter(PRESETS))
# Layer dropdown label -> key in PipelineResult.images.
LAYERS = {
    "Segmented cells (numbered)": "overlay",
    "Original image": "original",
    "Cell probability (U-Net)": "prob",
    "Binary mask": "mask",
    "Distance map with seeds": "distance",
}
DEFAULT_LAYER = next(iter(LAYERS))
TABLE_HEADERS = ["id", "area_px", "centroid_y", "centroid_x"]
OUT_H = 300  # px; equal panel height keeps the stage grid aligned

_logger = get_logger("app")

THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#EFF6FF", c100="#DBEAFE", c200="#BFDBFE", c300="#93C5FD", c400="#60A5FA",
        c500="#3B82F6", c600="#2563EB", c700="#1D4ED8", c800="#1E40AF", c900="#1E3A8A",
        c950="#172554"),
    secondary_hue="teal",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_md,
    spacing_size=gr.themes.sizes.spacing_md,
    font=[gr.themes.GoogleFont("Fira Sans"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Fira Code"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#F4F7FB",
    body_background_fill_dark="#0B1220",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#111A2E",
    block_border_color="#E2E8F0",
    block_border_color_dark="#1F2A44",
    block_border_width="1px",
    block_shadow="0 1px 2px rgba(15, 23, 42, 0.04)",
    block_label_text_weight="500",
    button_primary_background_fill="linear-gradient(135deg, *primary_600, #0D9488)",
    button_primary_background_fill_hover="linear-gradient(135deg, *primary_700, #0F766E)",
    button_primary_text_color="#FFFFFF",
    slider_color="*primary_600",
)

CSS = """
:root { --blue: #1D4ED8; --teal: #0D9488; --ink: #0F172A; --muted: #475569; --line: #E2E8F0;
  --card: #FFFFFF; --soft: #F8FAFC; --warn: #B45309; --err: #DC2626; }
.dark { --blue: #60A5FA; --teal: #2DD4BF; --ink: #E2E8F0; --muted: #94A3B8; --line: #1F2A44;
  --card: #111A2E; --soft: #0F172A; --warn: #F59E0B; --err: #F87171; }
.gradio-container { max-width: none !important; width: 100% !important;
  padding: 0 24px !important; }
footer { display: none !important; }
.num { font-family: 'Fira Code', ui-monospace, monospace; font-variant-numeric: tabular-nums; }

/* Header */
#app-header { display: flex; align-items: center; justify-content: space-between; gap: 16px;
  flex-wrap: wrap; margin: 12px 0 8px; padding: 18px 22px; border-radius: 14px;
  background: linear-gradient(120deg, #1E3A8A 0%, #1D4ED8 45%, #0D9488 100%); color: #fff;
  box-shadow: 0 6px 24px -12px rgba(29, 78, 216, 0.55); }
#app-header .brand { display: flex; align-items: center; gap: 14px; }
#app-header .logo { width: 44px; height: 44px; flex: none; }
#app-header h1 { margin: 0; font-size: 1.6rem; font-weight: 600; letter-spacing: -0.01em;
  color: #fff; }
#app-header p { margin: 2px 0 0; font-size: 0.93rem; color: rgba(255, 255, 255, 0.85);
  max-width: 72ch; }
#app-header .chips { display: flex; gap: 8px; flex-wrap: wrap; }
.chip { display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px;
  border-radius: 999px; font-size: 0.83rem; background: rgba(255, 255, 255, 0.14);
  border: 1px solid rgba(255, 255, 255, 0.25); color: #fff; white-space: nowrap; }
.chip.live::before { content: ""; width: 8px; height: 8px; border-radius: 50%;
  background: #5EEAD4; box-shadow: 0 0 0 3px rgba(94, 234, 212, 0.3); }

/* Panels */
.panel { background: var(--card) !important; border: 1px solid var(--line) !important;
  border-radius: 14px !important; padding: 14px !important; gap: 10px !important; }
.panel-title { display: flex; align-items: center; gap: 10px; margin: 0 0 2px;
  font-weight: 600; font-size: 0.98rem; color: var(--ink); }
.panel-title .n { display: inline-grid; place-items: center; width: 24px; height: 24px;
  border-radius: 8px; background: linear-gradient(135deg, var(--blue), var(--teal));
  color: #fff; font-size: 0.78rem; }
.panel-title svg { width: 18px; height: 18px; color: var(--blue); }
.hint { margin: 0; color: var(--muted); font-size: 0.86rem; line-height: 1.45; }
.group-label { margin: 8px 0 0; padding-top: 10px; border-top: 1px dashed var(--line);
  font-size: 0.74rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--teal); }
.group-label.first { border-top: 0; padding-top: 0; margin-top: 2px; }
#run-btn, #run-btn-2 { font-size: 1rem; font-weight: 600; min-height: 44px; }
button { cursor: pointer; transition: filter 150ms ease, transform 150ms ease; }
button:active { transform: translateY(1px); }

/* KPI strip */
#summary { display: grid; grid-template-columns: 1.3fr repeat(4, 1fr); gap: 10px; }
#summary .kpi { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
  padding: 12px 16px; position: relative; overflow: hidden; }
#summary .kpi::after { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
  background: var(--teal); }
#summary .kpi.main { background: linear-gradient(135deg, #EFF6FF, #F0FDFA); }
.dark #summary .kpi.main { background: linear-gradient(135deg, #172554, #042F2E); }
#summary .kpi.main::after { background: var(--blue); }
#summary .k { font-size: 0.8rem; color: var(--muted); }
#summary .v { font-size: 1.45rem; font-weight: 600; color: var(--ink); margin-top: 2px;
  font-family: 'Fira Code', ui-monospace, monospace; font-variant-numeric: tabular-nums; }
#summary .main .v { font-size: 2.4rem; line-height: 1.05; color: var(--blue); }
#summary .v small { font-size: 0.78rem; font-weight: 400; color: var(--muted); margin-left: 4px;
  font-family: 'Fira Sans', sans-serif; }
#summary.empty .v { color: var(--muted); }

/* Status */
.status { display: flex; gap: 10px; align-items: flex-start; padding: 10px 14px;
  border-radius: 10px; border: 1px solid color-mix(in srgb, var(--blue) 25%, transparent);
  background: color-mix(in srgb, var(--blue) 6%, transparent); color: var(--ink);
  font-size: 0.9rem; }
.status svg { width: 18px; height: 18px; flex: none; margin-top: 1px; color: var(--blue); }
.status.ok { border-color: color-mix(in srgb, var(--teal) 35%, transparent);
  background: color-mix(in srgb, var(--teal) 8%, transparent); }
.status.ok svg { color: var(--teal); }
.status.warn { border-color: color-mix(in srgb, var(--warn) 35%, transparent);
  background: color-mix(in srgb, var(--warn) 8%, transparent); }
.status.warn svg { color: var(--warn); }
.status.error { border-color: color-mix(in srgb, var(--err) 35%, transparent);
  background: color-mix(in srgb, var(--err) 8%, transparent); }
.status.error svg { color: var(--err); }
.status ul { margin: 4px 0 0; padding-left: 18px; color: var(--muted); }

/* Pipeline guide */
.guide { list-style: none; margin: 0; padding: 0; counter-reset: s; }
.guide li { position: relative; padding: 0 0 12px 34px; counter-increment: s; }
.guide li::before { content: counter(s); position: absolute; left: 0; top: 0; width: 22px;
  height: 22px; border-radius: 50%; display: grid; place-items: center; font-size: 0.74rem;
  font-weight: 600; color: var(--blue); border: 1.5px solid var(--blue); background: var(--card); }
.guide li:not(:last-child)::after { content: ""; position: absolute; left: 10.5px; top: 24px;
  bottom: 2px; width: 1.5px; background: var(--line); }
.guide b { display: block; font-size: 0.88rem; color: var(--ink); font-weight: 600; }
.guide span { display: block; font-size: 0.82rem; color: var(--muted); line-height: 1.4; }
.guide code { font-family: 'Fira Code', monospace; font-size: 0.74rem; padding: 1px 6px;
  border-radius: 6px; background: color-mix(in srgb, var(--teal) 12%, transparent);
  color: var(--teal); margin-right: 4px; }
.caption { margin: 2px 2px 0; font-size: 0.82rem; color: var(--muted); }

@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
@media (max-width: 1100px) {
  #summary { grid-template-columns: repeat(3, 1fr); }
  #summary .kpi.main { grid-column: 1 / -1; }
}
@media (max-width: 720px) {
  .gradio-container { padding: 0 12px !important; }
  #summary { grid-template-columns: 1fr 1fr; }
}
"""

LOGO_SVG = (
    '<svg class="logo" viewBox="0 0 48 48" fill="none" aria-hidden="true">'
    '<rect width="48" height="48" rx="12" fill="rgba(255,255,255,0.14)"/>'
    '<ellipse cx="19" cy="21" rx="9" ry="8" stroke="#fff" stroke-width="2"/>'
    '<ellipse cx="31" cy="28" rx="8" ry="7" stroke="#5EEAD4" stroke-width="2"/>'
    '<circle cx="19" cy="21" r="2.6" fill="#fff"/><circle cx="31" cy="28" r="2.4" fill="#5EEAD4"/>'
    '</svg>')
# Lucide-style 24x24 stroke icons.
_ICON_PATHS = {
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
    "ok": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "warn": '<path d="m21.7 18-8-14a2 2 0 0 0-3.4 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.7-3Z"/>'
            '<path d="M12 9v4M12 17h.01"/>',
    "error": '<circle cx="12" cy="12" r="10"/><path d="m15 9-6 6M9 9l6 6"/>',
    "sliders": '<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/>',
    "book": '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2zM22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',
}


def _icon(name: str) -> str:
    return (f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            f'{_ICON_PATHS[name]}</svg>')


GUIDE_HTML = (
    f'<div class="panel-title">{_icon("book")}How it works</div><ol class="guide">'
    '<li><b>Preprocess</b><span><code>clip</code>CLAHE lifts local contrast, then a light blur '
    'removes noise.</span></li>'
    '<li><b>U-Net probability</b><span>A ResNet34 U-Net scores every pixel 0–1 for '
    '“is this a cell?”.</span></li>'
    '<li><b>Threshold + cleanup</b><span><code>τ</code><code>kernel</code>Pixels above τ become '
    'the mask; morphology fills holes and drops specks.</span></li>'
    '<li><b>Watershed split</b><span><code>r</code><code>min dist</code>Distance-map peaks seed '
    'one region per cell, so touching cells separate.</span></li>'
    '<li><b>Count &amp; measure</b><span><code>min area</code>Tiny regions are dropped; the rest '
    'are numbered and measured.</span></li></ol>')


def _sample_images() -> List[List[str]]:
    if not SAMPLES_DIR.is_dir():
        return []
    return [[str(p)] for p in sorted(SAMPLES_DIR.iterdir()) if p.suffix.lower() in IMAGE_EXT]


def _out_image(label: str, height: int = OUT_H) -> gr.Image:
    return gr.Image(label=label, format="png", interactive=False, height=height,
                    buttons=["download", "fullscreen"])


def _title(n, text: str) -> gr.HTML:
    badge = f'<span class="n">{n}</span>' if isinstance(n, int) else _icon(n)
    return gr.HTML(f'<div class="panel-title">{badge}{text}</div>')


def _summary_html(stats: Dict[str, float] = None, areas: Tuple[int, int] = None) -> str:
    keys = ("Cells detected", "Mean area", "Median area", "Std. deviation", "Area range")
    if stats is None:
        vals, cls = ("–",) * 5, ' class="empty"'
    else:
        px = "<small>px²</small>"
        rng = f"{areas[0]}–{areas[1]}{px}" if areas else "–"
        vals = (str(stats["count"]), f"{stats['mean_area']:.1f}{px}",
                f"{stats['median_area']:.1f}{px}", f"{stats['std_area']:.1f}{px}", rng)
        cls = ""
    tiles = "".join(
        f'<div class="kpi{" main" if i == 0 else ""}"><div class="k">{k}</div>'
        f'<div class="v">{v}</div></div>' for i, (k, v) in enumerate(zip(keys, vals)))
    return f'<div id="summary"{cls} aria-live="polite">{tiles}</div>'


def _status_html(text: str, kind: str = "info", notes: List[str] = ()) -> str:
    items = "".join(f"<li>{html.escape(n)}</li>" for n in notes)
    extra = f"<ul>{items}</ul>" if items else ""
    return (f'<div class="status {kind}" role="status">{_icon(kind)}'
            f'<div>{html.escape(text)}{extra}</div></div>')


def _preset_values(name: str) -> List[float]:
    overrides = PRESETS[name][1]
    return [overrides.get(field, getattr(DEFAULTS, field)) for field, *_ in SLIDERS]


def _preset_hint(name: str) -> str:
    text = PRESETS[name][0] if name in PRESETS else "Your own settings."
    return f'<p class="hint">{html.escape(text)}</p>'


def _empty(status: str) -> Tuple:
    # overlay, hist, 4 stages, summary, csv, table, layers state, layer dropdown, status
    return ((None,) * 6 + (_summary_html(), None, None, {}, DEFAULT_LAYER, status))


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
            df = pipeline.analyzer.to_dataframe(res.cells)
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            stamp = f"{datetime.now():%Y%m%d_%H%M%S_%f}"[:-3]  # ms: no clash on quick reruns
            csv_path = OUTPUTS_DIR / f"{stamp}_cells.csv"
            df.to_csv(csv_path, index=False)
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
            status = _status_html(f"Segmented {res.count} cells. Explore layers with the "
                                  "dropdown; per-cell measurements are in the CSV.", "ok",
                                  res.warnings)
        im = res.images
        areas = (int(df.area_px.min()), int(df.area_px.max())) if res.count else None
        return (im["overlay"], pipeline.analyzer.histogram(res.cells),
                im["original"], im["prob"], im["mask"], im["distance"],
                _summary_html(stats, areas), str(csv_path), df[TABLE_HEADERS],
                im, DEFAULT_LAYER, status)

    def show_layer(layer, layers):
        return layers.get(LAYERS[layer]) if layers else None

    def apply_preset(name):
        if name not in PRESETS:  # "Custom": keep the user's slider values
            return (gr.skip(),) * len(SLIDERS) + (_preset_hint(name),)
        return tuple(_preset_values(name)) + (_preset_hint(name),)

    def reset():
        return (tuple(_preset_values(DEFAULT_PRESET)) + (DEFAULT_PRESET, _preset_hint(DEFAULT_PRESET), None)
                + _empty(_status_html(NO_IMAGE_MSG)))

    with gr.Blocks(title=TITLE, fill_width=True) as demo:
        gr.HTML(
            f'<header id="app-header"><div class="brand">{LOGO_SVG}<div><h1>{TITLE}</h1>'
            "<p>Count and measure cells in stained microscopy images, including cells that "
            "touch. A U-Net finds cell pixels; watershed splits clumps into single cells.</p>"
            '</div></div><div class="chips"><span class="chip">U-Net · ResNet34</span>'
            '<span class="chip">Watershed</span>'
            f'<span class="chip live">Running on {html.escape(device_text)}</span></div></header>')
        layers = gr.State({})
        with gr.Row(equal_height=False):
            # Left: input + guide
            with gr.Column(scale=1, min_width=300):
                with gr.Column(variant="panel"):
                    _title(1, "Choose an image")
                    inp = gr.Image(type="numpy", label="Microscopy image (PNG, JPG, TIF)",
                                   sources=["upload", "clipboard"], height=230)
                    examples = _sample_images()
                    if examples:
                        gr.Examples(examples, inputs=inp, label="Or try a sample",
                                    cache_examples=False, examples_per_page=6)
                with gr.Column(variant="panel"):
                    _title(2, "Segment")
                    with gr.Row():
                        run_btn = gr.Button("Segment cells", variant="primary", scale=3,
                                            min_width=0, elem_id="run-btn")
                        reset_btn = gr.Button("Reset", variant="secondary", scale=1,
                                              min_width=0)
                with gr.Column(variant="panel"):
                    gr.HTML(GUIDE_HTML)

            # Center: results
            with gr.Column(scale=3, min_width=480):
                summary = gr.HTML(_summary_html())
                status = gr.HTML(_status_html(NO_IMAGE_MSG))
                with gr.Tabs():
                    with gr.Tab("Result"):
                        with gr.Row(equal_height=False):
                            with gr.Column(scale=3, min_width=360):
                                layer_dd = gr.Dropdown(
                                    list(LAYERS), value=DEFAULT_LAYER, label="Show layer",
                                    info="Compare each pipeline stage in the same view.")
                                out_view = _out_image("Viewer", height=520)
                            with gr.Column(scale=2, min_width=320):
                                out_hist = gr.Plot(label="Cell area distribution")
                                out_csv = gr.File(label="Per-cell measurements (CSV)",
                                                  interactive=False)
                    with gr.Tab("Pipeline stages"):
                        gr.HTML('<p class="caption">Left to right, top to bottom: what the '
                                "model sees at each step. Change a parameter on the right and "
                                "segment again to see its effect.</p>")
                        with gr.Row():
                            out_orig = _out_image("1 · Original image")
                            out_prob = _out_image("2 · Cell probability (U-Net)")
                        with gr.Row():
                            out_mask = _out_image("3 · Binary mask (τ + morphology)")
                            out_dist = _out_image("4 · Distance map with seeds (r, min dist)")
                    with gr.Tab("Measurements"):
                        out_table = gr.Dataframe(headers=TABLE_HEADERS, interactive=False,
                                                 max_height=560, label="One row per cell")

            # Right: parameters
            with gr.Column(scale=1, min_width=300, variant="panel"):
                _title("sliders", "Parameters")
                preset_dd = gr.Dropdown(list(PRESETS) + [CUSTOM], value=DEFAULT_PRESET,
                                        label="Image type preset",
                                        info="Pick the closest match, then fine-tune.")
                preset_hint = gr.HTML(_preset_hint(DEFAULT_PRESET))
                sliders = []
                for i, (title, group) in enumerate(PARAM_GROUPS):
                    gr.HTML(f'<div class="group-label{" first" if i == 0 else ""}">{title}</div>')
                    sliders += [gr.Slider(lo, hi, value=getattr(DEFAULTS, name), step=step,
                                          label=label, info=info)
                                for name, label, lo, hi, step, info in group]
                run_btn_2 = gr.Button("Apply & segment", variant="primary", elem_id="run-btn-2")

        outputs = [out_view, out_hist, out_orig, out_prob, out_mask, out_dist,
                   summary, out_csv, out_table, layers, layer_dd, status]
        for btn in (run_btn, run_btn_2):
            btn.click(segment, inputs=[inp, *sliders], outputs=outputs)
        layer_dd.change(show_layer, inputs=[layer_dd, layers], outputs=out_view)
        preset_dd.change(apply_preset, inputs=preset_dd, outputs=[*sliders, preset_hint])
        for s in sliders:  # manual edits (not preset updates) mark the settings as custom
            s.input(lambda: CUSTOM, outputs=preset_dd)
        reset_btn.click(reset, outputs=[*sliders, preset_dd, preset_hint, inp, *outputs])
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860, theme=THEME, css=CSS)
