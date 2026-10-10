"""Gradio UI entry point."""
import dataclasses
from datetime import datetime
from typing import List, Tuple

import gradio as gr

from config import OUTPUTS_DIR, SAMPLES_DIR, Params
from core.pipeline import CellCountingPipeline
from utils.device import device_label
from utils.logger import get_logger

TITLE = "Cell Segmentation & Counting"
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
NO_IMAGE_MSG = "⚠️ Please upload an image first."
NO_CELLS_MSG = "⚠️ No cells found — try lowering τ or the min cell area."
DEFAULTS = Params()
# (field, label, min, max, step) in UI order; blur_k stays at its default (not in mockup).
SLIDERS = [
    ("thresh", "Probability threshold τ", 0.05, 0.95, 0.01),
    ("kernel", "Morphology kernel size", 1, 15, 2),
    ("peak_ratio", "Peak ratio r", 0.1, 0.9, 0.05),
    ("min_dist", "Min peak distance (px)", 1, 50, 1),
    ("min_area", "Min cell area (px²)", 0, 1000, 10),
    ("clip", "CLAHE clip limit", 0.5, 8.0, 0.1),
]
INT_FIELDS = {"kernel", "min_dist", "min_area"}
N_OUTPUTS = 11  # 5 images, plot, 3 stats, csv, status
OUT_H = 260  # px; equal panel height keeps the 2x3 grid aligned

_logger = get_logger("app")


def _sample_images() -> List[List[str]]:
    if not SAMPLES_DIR.is_dir():
        return []
    return [[str(p)] for p in sorted(SAMPLES_DIR.iterdir()) if p.suffix.lower() in IMAGE_EXT]


def _out_image(label: str) -> gr.Image:
    return gr.Image(label=label, format="png", interactive=False, height=OUT_H)


def _empty(status: str) -> Tuple:
    return (None,) * (N_OUTPUTS - 1) + (status,)


def build_ui() -> gr.Blocks:
    pipeline = CellCountingPipeline()
    device_text = device_label(pipeline.device)

    def segment(img, *values):
        if img is None:
            return _empty(NO_IMAGE_MSG)
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
            return _empty(f"❌ {e}")
        except ValueError as e:
            return _empty(f"❌ Invalid input: {e}")
        except Exception as e:  # app must never crash (architecture p.6)
            _logger.exception("Segmentation failed")
            return _empty(f"❌ Unexpected error ({type(e).__name__}: {e}). Details in outputs/app.log.")
        status = NO_CELLS_MSG if res.count == 0 else f"✅ Done — {res.count} cells detected."
        im = res.images
        return (im["original"], im["prob"], im["mask"], im["distance"], im["overlay"],
                pipeline.analyzer.histogram(res.cells), str(res.count),
                f"{stats['mean_area']:.1f} px²", f"{stats['median_area']:.1f} px²",
                str(csv_path), status)

    def reset():
        return tuple(getattr(DEFAULTS, name) for name, *_ in SLIDERS) + (None,) + _empty("")

    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"# {TITLE}\nU-Net probability map → threshold + morphology → "
                    "distance transform + watershed → numbered cells.")
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                inp = gr.Image(type="numpy", label="Upload image (PNG / JPG / TIF)",
                               sources=["upload", "clipboard"], height=260)
                sliders = [gr.Slider(lo, hi, value=getattr(DEFAULTS, name), step=step, label=label)
                           for name, label, lo, hi, step in SLIDERS]
                with gr.Row():
                    run_btn = gr.Button("Run segmentation", variant="primary")
                    reset_btn = gr.Button("Reset")
                examples = _sample_images()
                if examples:
                    gr.Examples(examples, inputs=inp, label="Sample images", cache_examples=False)
            with gr.Column(scale=3):
                with gr.Row():
                    out_orig = _out_image("Original image")
                    out_prob = _out_image("Probability heatmap")
                    out_mask = _out_image("Binary mask")
                with gr.Row():
                    out_dist = _out_image("Distance map + seeds")
                    out_over = _out_image("Segmented & numbered cells")
                    out_hist = gr.Plot(label="Cell area distribution", show_label=False)
                with gr.Row():
                    st_count = gr.Textbox(label="Cells detected", interactive=False)
                    st_mean = gr.Textbox(label="Mean area", interactive=False)
                    st_median = gr.Textbox(label="Median area", interactive=False)
                    gr.Textbox(value=device_text, label="Device", interactive=False)
                    out_csv = gr.File(label="Download CSV", interactive=False)
                status = gr.Markdown()

        outputs = [out_orig, out_prob, out_mask, out_dist, out_over, out_hist,
                   st_count, st_mean, st_median, out_csv, status]
        run_btn.click(segment, inputs=[inp, *sliders], outputs=outputs)
        reset_btn.click(reset, outputs=[*sliders, inp, *outputs])
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860, allowed_paths=[str(OUTPUTS_DIR)])
