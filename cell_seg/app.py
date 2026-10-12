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
        c50="#EFF4FF", c100="#DBE6FE", c200="#BFD3FE", c300="#93B4FD", c400="#6090FA",
        c500="#3B6FF6", c600="#2557EB", c700="#1D45D8", c800="#1E3AAF", c900="#1E348A",
        c950="#172254"),
    secondary_hue="teal",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_md,
    spacing_size=gr.themes.sizes.spacing_md,
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#F6F8FB",
    body_background_fill_dark="#0E1420",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#151C2B",
    block_border_color="#E3E8EF",
    block_border_color_dark="#263042",
    block_border_width="1px",
    block_shadow="none",
    block_label_text_weight="500",
    button_primary_background_fill="*primary_700",
    button_primary_background_fill_hover="*primary_800",
    button_primary_background_fill_dark="*primary_600",
    button_primary_background_fill_hover_dark="*primary_700",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#FFFFFF",
    button_secondary_background_fill="#FFFFFF",
    button_secondary_background_fill_hover="#F1F4F9",
    button_secondary_border_color="#D5DCE6",
    button_secondary_border_color_dark="#33405A",
    button_border_width="1px",
    slider_color="*primary_700",
    slider_color_dark="*primary_500",
)

CSS = """
:root { --blue: #1D45D8; --teal: #0F766E; --ink: #0F172A; --muted: #5B6779; --line: #E3E8EF;
  --card: #FFFFFF; --warn: #B45309; --err: #C62828; }
.dark { --blue: #7FA6FF; --teal: #2DD4BF; --ink: #E6EAF2; --muted: #93A0B4; --line: #263042;
  --card: #151C2B; --warn: #F2A93B; --err: #F47174; }
.gradio-container { max-width: none !important; width: 100% !important;
  padding: 0 28px 24px !important; }
footer { display: none !important; }

/* Top bar: quiet, one line */
#topbar { align-items: center; gap: 12px; padding: 14px 0; margin-bottom: 12px;
  border-bottom: 1px solid var(--line); }
#topbar > div { background: transparent !important; border: 0 !important; padding: 0 !important; }
#app-header { display: flex; align-items: center; justify-content: space-between; gap: 16px;
  flex-wrap: wrap; }
#app-header .brand { display: flex; align-items: center; gap: 12px; }
#app-header .logo { width: 32px; height: 32px; flex: none; }
#app-header h1 { margin: 0; font-size: 1.12rem; font-weight: 600; color: var(--ink); }
#app-header p { margin: 1px 0 0; font-size: 0.86rem; color: var(--muted); }
.device { display: inline-flex; align-items: center; gap: 8px; font-size: 0.84rem;
  color: var(--muted); white-space: nowrap; }
.device::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: var(--teal); }
#theme-btn { flex: none !important; width: 36px; min-width: 36px !important; height: 36px;
  padding: 0 !important; font-size: 0 !important; box-shadow: none !important;
  display: flex !important; align-items: center; justify-content: center; line-height: 0; }
#theme-btn:focus-visible { outline: 2px solid var(--blue); outline-offset: 2px; }
#theme-btn::before { content: ""; width: 18px; height: 18px; background: var(--muted);
  -webkit-mask: var(--icon) center / contain no-repeat; mask: var(--icon) center / contain no-repeat;
  --icon: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z'/%3E%3C/svg%3E"); }
.dark #theme-btn::before {
  --icon: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='4'/%3E%3Cpath d='M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4'/%3E%3C/svg%3E"); }

/* Side panels: one flat surface each, no cards inside cards */
.panel { background: var(--card) !important; border: 1px solid var(--line) !important;
  border-radius: 10px !important; padding: 16px !important; gap: 12px !important; }
.panel .block:not(#inp), .panel .form { border: 0 !important; box-shadow: none !important;
  background: transparent !important; }
.panel .block:not(#inp) { padding-left: 0 !important; padding-right: 0 !important; }
.panel .html-container, .html-container:has(#summary) { padding: 0 !important; }
.section-title { margin: 0; font-size: 0.95rem; font-weight: 600; color: var(--ink); }
.group-title { margin: 6px 0 -4px; padding-top: 14px; border-top: 1px solid var(--line);
  font-size: 0.86rem; font-weight: 600; color: var(--ink); }
.hint { margin: -8px 0 0; color: var(--muted); font-size: 0.84rem; line-height: 1.45; }
.divider { border-top: 1px solid var(--line); margin: 2px 0; }
#run-btn { min-height: 42px; font-weight: 600; }

/* Summary: one strip, the count is the only emphasis */
#summary { display: grid; grid-template-columns: 1.2fr repeat(4, 1fr); background: var(--card);
  border: 1px solid var(--line); border-radius: 10px; }
#summary .kpi { padding: 14px 18px; border-left: 1px solid var(--line); }
#summary .kpi:first-child { border-left: 0; }
#summary .k { font-size: 0.8rem; color: var(--muted); }
#summary .v { margin-top: 4px; font-size: 1.3rem; font-weight: 500; color: var(--ink);
  font-variant-numeric: tabular-nums; }
#summary .main .v { font-size: 2.1rem; line-height: 1.05; font-weight: 600; color: var(--blue); }
#summary .v small { margin-left: 3px; font-size: 0.76rem; font-weight: 400; color: var(--muted); }
#summary.empty .v { color: var(--muted); font-weight: 400; }

/* Status: plain line; colour only for problems */
.status { display: flex; gap: 8px; align-items: flex-start; padding: 2px;
  color: var(--muted); font-size: 0.88rem; }
.status svg { width: 16px; height: 16px; flex: none; margin-top: 2px; }
.status.ok svg { color: var(--teal); }
.status.warn, .status.error { padding: 9px 12px; border-radius: 8px; color: var(--ink); }
.status.warn { background: color-mix(in srgb, var(--warn) 9%, transparent); }
.status.warn svg { color: var(--warn); }
.status.error { background: color-mix(in srgb, var(--err) 9%, transparent); }
.status.error svg { color: var(--err); }
.status ul { margin: 4px 0 0; padding-left: 18px; color: var(--muted); }

/* How it works */
.guide { max-width: 72ch; margin: 8px 0 0; padding: 0 0 0 1.4em; color: var(--ink); }
.guide li { margin-bottom: 14px; padding-left: 4px; }
.guide li::marker { color: var(--blue); font-weight: 600; }
.guide b { display: block; font-weight: 600; }
.guide p { display: block; margin: 2px 0 0 !important; color: var(--muted) !important; font-size: 0.9rem; line-height: 1.5; }
.guide em { font-style: normal; color: var(--ink); }
.caption { margin: 0; font-size: 0.86rem; color: var(--muted); }
/* Matplotlib renders a light PNG; invert it in dark mode (hue-rotate keeps the blue bars blue). */
.dark #hist img { filter: invert(0.9) hue-rotate(180deg); }

@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
@media (max-width: 1100px) {
  #summary { grid-template-columns: repeat(3, 1fr); }
  #summary .kpi:first-child { grid-column: 1 / -1; border-bottom: 1px solid var(--line); }
  #summary .kpi:nth-child(2) { border-left: 0; }
}
@media (max-width: 720px) {
  .gradio-container { padding: 0 12px 16px !important; }
  #summary { grid-template-columns: 1fr 1fr; }
  #summary .kpi:nth-child(odd):not(:first-child) { border-left: 0; }
}
"""

LOGO_SVG = (
    '<svg class="logo" viewBox="0 0 32 32" fill="none" aria-hidden="true">'
    '<rect width="32" height="32" rx="8" fill="#1D45D8"/>'
    '<ellipse cx="13" cy="14" rx="6" ry="5.4" stroke="#fff" stroke-width="1.6"/>'
    '<ellipse cx="20.5" cy="19" rx="5.2" ry="4.6" stroke="#5EEAD4" stroke-width="1.6"/>'
    '</svg>')
# Lucide-style 24x24 stroke icons.
_ICON_PATHS = {
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
    "ok": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "warn": '<path d="m21.7 18-8-14a2 2 0 0 0-3.4 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.7-3Z"/>'
            '<path d="M12 9v4M12 17h.01"/>',
    "error": '<circle cx="12" cy="12" r="10"/><path d="m15 9-6 6M9 9l6 6"/>',
}


def _icon(name: str) -> str:
    return (f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            f'{_ICON_PATHS[name]}</svg>')


GUIDE_HTML = (
    '<ol class="guide">'
    '<li><b>Enhance contrast</b><p>CLAHE brightens dim regions, then a light blur removes '
    'noise. Setting: <em>contrast boost</em>.</p></li>'
    '<li><b>Find cell pixels</b><p>A U-Net (ResNet34 encoder) gives every pixel a probability '
    'of belonging to a cell.</p></li>'
    '<li><b>Make the mask</b><p>Pixels above the threshold become the mask; smoothing fills '
    'holes and removes specks. Settings: <em>probability threshold τ</em>, '
    '<em>smoothing kernel</em>.</p></li>'
    '<li><b>Split touching cells</b><p>Peaks of the distance map seed a watershed, giving one '
    'region per cell. Settings: <em>peak ratio r</em>, <em>minimum peak distance</em>.</p></li>'
    '<li><b>Count and measure</b><p>Regions below the minimum area are dropped; the rest are '
    'numbered in reading order and measured. Setting: <em>minimum cell area</em>.</p></li>'
    '</ol>')


THEME_KEY = "cellseg-theme"
# Gradio puts the "dark" class on <body>; we flip it and remember the choice per browser.
TOGGLE_THEME_JS = f"""() => {{
  const dark = !document.body.classList.contains('dark');
  document.body.classList.toggle('dark', dark);
  try {{ localStorage.setItem('{THEME_KEY}', dark ? 'dark' : 'light'); }} catch (e) {{}}
}}"""
RESTORE_THEME_JS = f"""() => {{
  let saved = null;
  try {{ saved = localStorage.getItem('{THEME_KEY}'); }} catch (e) {{}}
  if (saved) document.body.classList.toggle('dark', saved === 'dark');
}}"""


def _sample_images() -> List[List[str]]:
    if not SAMPLES_DIR.is_dir():
        return []
    return [[str(p)] for p in sorted(SAMPLES_DIR.iterdir()) if p.suffix.lower() in IMAGE_EXT]


def _out_image(label: str, height: int = OUT_H) -> gr.Image:
    return gr.Image(label=label, format="png", interactive=False, height=height,
                    buttons=["download", "fullscreen"])


def _heading(text: str, cls: str = "section-title") -> gr.HTML:
    return gr.HTML(f'<div class="{cls}">{text}</div>')


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
            status = _status_html(f"Segmented {res.count} cells. Switch layers to "
                                  "compare steps; per-cell data is in the CSV.", "ok",
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
        with gr.Row(elem_id="topbar", equal_height=True):
            gr.HTML(
                f'<header id="app-header"><div class="brand">{LOGO_SVG}<div><h1>{TITLE}</h1>'
                "<p>Count and measure touching cells in stained microscopy images.</p></div>"
                f'</div><span class="device">{html.escape(device_text)}</span></header>')
            theme_btn = gr.Button("Toggle dark mode", elem_id="theme-btn", scale=0, min_width=36)
        layers = gr.State({})
        with gr.Row(equal_height=False):
            # Left: image in, run
            with gr.Column(scale=1, min_width=290, variant="panel"):
                _heading("Image")
                inp = gr.Image(type="numpy", label="Microscopy image (PNG, JPG, TIF)",
                               sources=["upload", "clipboard"], height=220, elem_id="inp")
                examples = _sample_images()
                if examples:
                    gr.Examples(examples, inputs=inp, label="Samples",
                                cache_examples=False, examples_per_page=6)
                gr.HTML('<div class="divider"></div>')
                with gr.Row():
                    run_btn = gr.Button("Segment cells", variant="primary", scale=3,
                                        min_width=0, elem_id="run-btn")
                    reset_btn = gr.Button("Reset", variant="secondary", scale=1, min_width=0)

            # Center: results
            with gr.Column(scale=3, min_width=480):
                summary = gr.HTML(_summary_html())
                status = gr.HTML(_status_html(NO_IMAGE_MSG))
                with gr.Tabs():
                    with gr.Tab("Result"):
                        with gr.Row(equal_height=False):
                            with gr.Column(scale=3, min_width=360):
                                layer_dd = gr.Dropdown(list(LAYERS), value=DEFAULT_LAYER,
                                                       label="Layer")
                                out_view = _out_image("Viewer", height=440)
                            with gr.Column(scale=2, min_width=320):
                                out_hist = gr.Plot(label="Cell area distribution", elem_id="hist")
                                out_csv = gr.File(label="Per-cell measurements (CSV)",
                                                  interactive=False)
                    with gr.Tab("Pipeline stages"):
                        gr.HTML('<p class="caption">Intermediate images from each step. Change '
                                "a setting and segment again to compare.</p>")
                        with gr.Row():
                            out_orig = _out_image("1. Original image")
                            out_prob = _out_image("2. Cell probability (U-Net)")
                        with gr.Row():
                            out_mask = _out_image("3. Binary mask")
                            out_dist = _out_image("4. Distance map with seeds")
                    with gr.Tab("Measurements"):
                        out_table = gr.Dataframe(headers=TABLE_HEADERS, interactive=False,
                                                 max_height=560, label="One row per cell")
                    with gr.Tab("How it works"):
                        gr.HTML(GUIDE_HTML)

            # Right: settings
            with gr.Column(scale=1, min_width=300, variant="panel"):
                _heading("Settings")
                preset_dd = gr.Dropdown(list(PRESETS) + [CUSTOM], value=DEFAULT_PRESET,
                                        label="Preset")
                preset_hint = gr.HTML(_preset_hint(DEFAULT_PRESET))
                sliders = []
                for title, group in PARAM_GROUPS:
                    _heading(title, "group-title")
                    sliders += [gr.Slider(lo, hi, value=getattr(DEFAULTS, name), step=step,
                                          label=label, info=info)
                                for name, label, lo, hi, step, info in group]
                run_btn_2 = gr.Button("Segment with these settings", variant="secondary")

        outputs = [out_view, out_hist, out_orig, out_prob, out_mask, out_dist,
                   summary, out_csv, out_table, layers, layer_dd, status]
        for btn in (run_btn, run_btn_2):
            btn.click(segment, inputs=[inp, *sliders], outputs=outputs)
        layer_dd.change(show_layer, inputs=[layer_dd, layers], outputs=out_view)
        preset_dd.change(apply_preset, inputs=preset_dd, outputs=[*sliders, preset_hint])
        for s in sliders:  # manual edits (not preset updates) mark the settings as custom
            s.input(lambda: CUSTOM, outputs=preset_dd)
        reset_btn.click(reset, outputs=[*sliders, preset_dd, preset_hint, inp, *outputs])
        theme_btn.click(None, js=TOGGLE_THEME_JS)
        demo.load(None, js=RESTORE_THEME_JS)
    return demo


if __name__ == "__main__":
    # server_port=None: Gradio tries 7860 first, then the next free port.
    build_ui().launch(server_name="127.0.0.1", server_port=None, inbrowser=True,
                      theme=THEME, css=CSS)
