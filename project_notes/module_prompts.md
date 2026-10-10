# Module Prompts — Cell Adhesion Segmentation & Counting

One prompt per module, in build order. Each prompt is self-contained: copy **Section 0 (Shared context)** + **one module prompt** into a new request. Do one module per request (assignment rule: no whole-project generation at once).

After each module: log the dialogue in `iterationlog.md`, tick `TODO.md`, update `project_progress.md`, and add any new design choice to `decisions.md`.

## Build order

| # | Module | File(s) | Depends on | Phase |
|---|---|---|---|---|
| M1 | Device + logging | `utils/device.py`, `utils/logger.py` | — | 2 |
| M2 | Preprocessing | `core/preprocess.py` | — | 2 |
| M3 | U-Net segmenter | `core/unet_model.py` | M1, M2 | 2 |
| M4 | Mask refinement | `core/postprocess.py` | — | 2 |
| M5 | Watershed separation | `core/watershed.py` | M4 | 2 |
| M6 | Cell analysis | `core/analysis.py` | M5 | 2 |
| M7 | Visualisation | `utils/visualize.py` | M6 | 2 |
| M8 | Metrics | `utils/metrics.py` | — | 2 |
| M9 | Pipeline controller | `core/pipeline.py` | M1–M7 | 2 |
| M10 | Gradio UI | `app.py` | M9 | 2 |
| M11 | Training data + loss | `training/dataset.py`, `training/losses.py` | M2 | Training |
| M12 | Training loop | `training/train.py` | M3, M8, M11 | Training |
| M13 | Unit tests | `tests/` | M1–M9 | 3 |
| M14 | Integration & robustness | pipeline / unet / app | all | 3 |
| M15 | Tuning & ablation | `outputs/`, notes | all | 2–3 |

M4–M8 do not need trained weights — test them with synthetic masks / probability maps. M11–M12 can run in parallel with M4–M10.

---

## 0. Shared context (paste at top of every module prompt)

```text
PROJECT CONTEXT
Course project (Intro to AI, Topic 3): segment touching/adhering cells in stained
microscopy images and count them. Pipeline:
  input RGB image
  → Preprocessor (grayscale, CLAHE, Gaussian blur)
  → UNetSegmenter (ResNet34 U-Net, smp) → H×W probability map in [0,1]
  → MaskRefiner (threshold τ + opening/closing + remove small objects) → binary mask
  → WatershedSplitter (distance transform → peak seeds → watershed) → (distance map, label image)
  → CellAnalyzer (per-cell id, area, centroid, bbox; histogram)
  → Gradio UI shows every intermediate image, stats, CSV download.

CODE LOCATION: cell_seg/  (run all commands from inside cell_seg/; modules import
`from config import ...` and `from core.x import ...`, `from utils.x import ...`).
Architecture reference: Docs/Architechture.pdf (p.2 data flow, p.3 U-Net/training,
p.4 interfaces, p.5 UI mockup, p.6 error handling).

SHARED TYPES (cell_seg/config.py — already written, do not change field names):
  Params(clip=2.0, blur_k=3, thresh=0.5, kernel=5, peak_ratio=0.4, min_dist=10, min_area=50)
  CellInfo(id: int, area: int, centroid: Tuple[float,float], bbox: Tuple[int,int,int,int])
  PipelineResult(images: Dict[str, np.ndarray], cells: List[CellInfo], count: int)
  Paths: ROOT_DIR, WEIGHTS_PATH (weights/unet_cells.pth), SAMPLES_DIR, OUTPUTS_DIR

STACK: Python 3.8+ (use typing.List/Tuple/Dict, not list[...]), PyTorch,
segmentation-models-pytorch, OpenCV, scikit-image, SciPy, NumPy, Matplotlib, Gradio, pandas.

RULES
- Implement ONLY the module named below. Keep the public signature exactly as in the
  existing stub (approved in Phase 1). Private helpers (_name) are allowed.
- Short docstrings, type hints, no unused code, no print() — use utils.logger.get_logger.
- Validate inputs at the public method and raise ValueError with a clear message.
- Do not touch other modules except to fix an import; if another module must change,
  say so and stop.
- After coding: run the "Done when" checks, show the output, then update
  project_notes/ (iterationlog.md round entry, TODO.md tick, project_progress.md log line,
  decisions.md if a design choice was made).
```

---

## M1 — Device manager + logging

```text
MODULE: utils/device.py and utils/logger.py

GOAL
Shared infrastructure: pick GPU if available else CPU; consistent logging with
per-stage timing so the pipeline and UI can report where time is spent and where an
error happened.

INTERFACE (keep)
  utils/device.py: get_device() -> torch.device
  utils/logger.py: get_logger(name: str) -> logging.Logger

REQUIREMENTS
device.py
- Return torch.device("cuda") if torch.cuda.is_available(), else torch.device("cpu").
- Log which device was chosen (name of GPU if CUDA) once.
- Add device_label(device: torch.device) -> str returning "CUDA (GPU)" or "CPU"
  (UI "Device" card, mockup p.5).

logger.py
- get_logger: one logger per name, format "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
  level INFO, no duplicate handlers when called twice with the same name.
- Also log to file cell_seg/outputs/app.log (create folder if missing).
- Add a context manager `stage_timer(logger, stage: str)` that logs
  "<stage> done in X.XX s" on success and logs the exception with traceback
  (logger.exception) then re-raises on failure. Pipeline wraps each stage in it.

DONE WHEN
- python -c "from utils.device import get_device, device_label; d=get_device(); print(d, device_label(d))"
- python -c "from utils.logger import get_logger, stage_timer; import time; l=get_logger('t'); \
  exec('with stage_timer(l, \"sleep\"): time.sleep(0.1)')"  → one timing line, no duplicates.
- Calling get_logger('t') twice does not duplicate lines.
```

---

## M2 — Preprocessing

```text
MODULE: core/preprocess.py — class Preprocessor

GOAL
Normalise contrast and noise so the U-Net sees consistent input (data flow stage 1).
The SAME preprocessing is reused during training (M11), so it must be deterministic.

INTERFACE (keep)
  Preprocessor.run(img: np.ndarray, clip: float = 2.0, blur_k: int = 3) -> np.ndarray

INPUT
- img: H×W (gray), H×W×3 (RGB) or H×W×4 (RGBA) array; dtype uint8, uint16 or float.
  Gradio gives RGB uint8; TIF files may be uint16.

STEPS
1. Validate: ndim 2 or 3, channels 1/3/4, H and W ≥ 32, not empty. Else ValueError.
2. Convert to uint8: uint16 → scale by max; float → clip to [0,1] if max ≤ 1 else [0,255].
3. Drop alpha, convert RGB → grayscale (cv2.COLOR_RGB2GRAY — input is RGB, not BGR).
4. CLAHE: cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).
5. Gaussian blur with kernel blur_k. blur_k must be odd: if even, use blur_k + 1;
   blur_k ≤ 1 means no blur.
6. Return H×W uint8 grayscale (same H, W as input).

DONE WHEN
- Output shape == input H×W, dtype uint8, for RGB, gray, RGBA and uint16 inputs.
- clip=4.0 gives higher std-dev than clip=1.0 on a low-contrast synthetic image.
- Even blur_k does not crash.
- Show before/after on one sample image (save to outputs/preprocess_check.png).
```

---

## M3 — U-Net segmenter

```text
MODULE: core/unet_model.py — class UNetSegmenter

GOAL
Load the ResNet34 U-Net once and predict a per-pixel cell probability map
(data flow stage 2, architecture p.3).

INTERFACE (keep)
  UNetSegmenter.__init__(weights: str, device: torch.device) -> None
  UNetSegmenter.predict_prob(img: np.ndarray) -> np.ndarray   # H×W float32 in [0,1]

MODEL
- smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1).
- Load state_dict from `weights` with map_location=device; accept both a raw
  state_dict and a checkpoint dict with key "model" (train.py may save either).
- If weights file is missing: raise FileNotFoundError with message
  "U-Net weights not found at <path>. Train with training/train.py or copy unet_cells.pth
  into cell_seg/weights/." (pipeline turns this into a UI message).
- model.eval(), move to device once in __init__.

predict_prob STEPS
1. Input: H×W uint8 grayscale from Preprocessor. Validate ndim == 2.
2. Stack to 3 channels, scale to [0,1], normalise with ImageNet mean/std
   (must match training/dataset.py — put MEAN/STD constants at module top).
3. Reflect-pad H and W to a multiple of 32 (encoder downsamples 5×), remember padding.
4. torch.no_grad(), forward, sigmoid, crop padding off, return float32 H×W numpy.

NOTES
- No tiling yet — OOM handling/tiling is added in M14.
- Model must NOT be reloaded per call; pipeline holds one instance.

DONE WHEN
- With no weights file: clear FileNotFoundError.
- With a temporary randomly-initialised state_dict saved to scratch: predict_prob on
  a 300×451 image returns shape (300, 451), dtype float32, min ≥ 0, max ≤ 1.
- Works on CPU device.
```

---

## M4 — Mask refinement

```text
MODULE: core/postprocess.py — class MaskRefiner

GOAL
Turn the probability map into a clean binary cell mask (data flow stage 3).

INTERFACE (keep)
  MaskRefiner.run(prob: np.ndarray, thresh: float, kernel: int, min_area: int) -> np.ndarray

STEPS
1. Validate: prob 2-D, values in [0,1] (tolerate tiny float error), 0 < thresh < 1,
   kernel ≥ 1, min_area ≥ 0.
2. mask = prob > thresh.
3. Elliptical structuring element of size kernel (cv2.getStructuringElement MORPH_ELLIPSE).
   Opening (remove specks) then closing (fill small gaps). kernel == 1 → skip morphology.
4. Fill holes inside cells (scipy.ndimage.binary_fill_holes).
5. Remove connected components smaller than min_area (skimage.morphology.remove_small_objects).
6. Return uint8 mask, values {0, 255}, same H×W.

DONE WHEN
- Synthetic test: prob map with two discs + salt noise. Noise removed, discs kept,
  hole inside a disc filled.
- Higher thresh → mask pixel count never increases.
- All-zero prob → all-zero mask, no crash.
```

---

## M5 — Watershed separation

```text
MODULE: core/watershed.py — class WatershedSplitter

GOAL
Split touching/overlapping cells that merged into one blob in the mask
(data flow stages 4–5; p.2 "Why distance transform + watershed?").

INTERFACE (keep)
  WatershedSplitter.run(img: np.ndarray, mask: np.ndarray, peak_ratio: float,
                        min_dist: int) -> Tuple[np.ndarray, np.ndarray]
  returns (distance_map float32 H×W, labels int32 H×W; 0 = background, 1..N = cells)

STEPS
1. Validate: img and mask same H×W; 0 < peak_ratio < 1; min_dist ≥ 1.
   img is the preprocessed grayscale — used only for shape check (kept for interface).
2. dist = cv2.distanceTransform(mask>0 as uint8, cv2.DIST_L2, 5).
3. Seeds per blob: skimage.feature.peak_local_max(dist, min_distance=min_dist,
   threshold_abs=peak_ratio * dist.max(), labels=connected-components of mask,
   exclude_border=False).
   NOTE: global max can suppress small cells next to big ones. Use a per-blob
   threshold instead: for each connected component, keep peaks with
   dist ≥ peak_ratio * (max dist in that component). Document the choice in decisions.md.
4. Guarantee every blob has ≥ 1 seed: if a component got no peak, add its argmax.
5. markers = label seeds; labels = skimage.segmentation.watershed(-dist, markers, mask=mask>0).
6. Return (dist.astype(float32), labels.astype(int32)).

DONE WHEN
- Two overlapping synthetic discs (centres 30 px apart, radius 20) → 2 labels.
- Three separate discs → 3 labels.
- Empty mask → dist all zeros, labels all zeros, no crash.
- Larger min_dist → label count never increases on the same input.
```

---

## M6 — Cell analysis

```text
MODULE: core/analysis.py — class CellAnalyzer

GOAL
Number every cell, measure area/centroid/bbox, produce area histogram
(data flow stage 6).

INTERFACE (keep)
  CellAnalyzer.run(labels: np.ndarray, min_area: int) -> List[CellInfo]
  CellAnalyzer.histogram(cells: List[CellInfo]) -> matplotlib.figure.Figure

renumber(labels, min_area) -> np.ndarray  (stub exists)
1. Validate: labels 2-D integer array.
2. Drop regions with area < min_area (watershed can create small slivers).
3. Sort remaining regions top-to-bottom then left-to-right (by centroid row bucketed
   to ~cell height, then column) so numbers read naturally.
4. Return new int32 label image with ids 1..N in that order (0 = background).

run STEPS
1. labels = self.renumber(labels, min_area).
2. skimage.measure.regionprops(labels) — region.label is already the final id.
3. CellInfo(id, area=int(area), centroid=(row, col) floats, bbox=(min_row, min_col,
   max_row, max_col)).

ALSO (stubs exist)
- summary(cells) -> Dict[str, float]: count, mean_area, median_area, std_area
  (0 values when no cells). Used by UI stat cards.
- to_dataframe(cells) -> pandas.DataFrame with columns
  id, area_px, centroid_y, centroid_x, bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col.

histogram
- Matplotlib Figure (use matplotlib.figure.Figure, not pyplot — thread-safe for Gradio),
  bins="auto", x label "Cell area (px²)", y label "Count", title "Cell area distribution",
  dashed vertical line at the mean. Empty list → figure with text "No cells detected".

DONE WHEN
- Label image with 3 squares of known area (one below min_area) → 2 CellInfo, correct areas.
- ids are consecutive 1..N.
- histogram([]) and histogram(cells) both return a Figure; save one to outputs/hist_check.png.
```

---

## M7 — Visualisation

```text
MODULE: utils/visualize.py

GOAL
Produce the images shown in the UI (p.5): probability heatmap, distance map with seeds,
numbered contour overlay.

INTERFACE (keep)
  prob_heatmap(prob: np.ndarray) -> np.ndarray
  overlay_contours(img: np.ndarray, labels: np.ndarray, cells: List[CellInfo]) -> np.ndarray

ADD
  distance_map_view(dist: np.ndarray, labels: np.ndarray) -> np.ndarray
  mask_view(mask: np.ndarray) -> np.ndarray

All return RGB uint8 H×W×3 (Gradio displays RGB).

DETAILS
- prob_heatmap: viridis colormap (matplotlib.cm) on prob in [0,1] — fixed scale, not
  min-max normalised, so τ is comparable across images.
- distance_map_view: dist normalised to [0,255], gray/bone colormap, red dot (radius 3)
  at the max-distance point of each label (the seed).
- mask_view: 0/255 mask → 3-channel.
- overlay_contours: original image (convert gray → RGB if needed; resize-safe: must
  match labels H×W), green contours of each cell in `cells`, white number at centroid
  with dark outline for readability; font scale adapts to image size.
  `labels` is the RENUMBERED image from CellAnalyzer.renumber, so label value == cell.id
  (decided in Phase 1, see decisions.md).

DONE WHEN
- All four functions return uint8 RGB with input H×W.
- Saved check images in outputs/vis_check_*.png look right on a synthetic case.
```

---

## M8 — Metrics

```text
MODULE: utils/metrics.py

GOAL
Quantitative evaluation: segmentation quality (IoU, Dice) for training/validation,
image similarity (PSNR, SSIM) for the ablation study.

INTERFACE (keep)
  iou(pred, gt) -> float
  dice(pred, gt) -> float
  psnr(a, b) -> float
  ssim(a, b) -> float

ADD
  count_error(pred_count: int, gt_count: int) -> float   # |pred-gt| / gt * 100, gt=0 handled
  cell_precision_recall(pred_labels, gt_labels, iou_thr=0.5) -> Tuple[float, float]
     # match instances by IoU ≥ thr (greedy), p.3 "precision / recall per cell"

DETAILS
- iou/dice: binarise (> 0), same shape required. Both empty → return 1.0 (perfect match).
- psnr/ssim: use skimage.metrics with data_range from dtype (255 for uint8, 1.0 for float);
  images must be same shape; identical images → psnr returns float('inf') — keep that,
  UI/report formats it.
- ssim on RGB: channel_axis=-1.

DONE WHEN
- iou(x, x) == 1, dice(x, x) == 1, iou(x, ~x) == 0.
- Known example: pred 4 px, gt 4 px, overlap 2 → IoU 2/6, Dice 0.5.
- ssim(x, x) == 1.0; psnr decreases as noise increases.
```

---

## M9 — Pipeline controller

```text
MODULE: core/pipeline.py — class CellCountingPipeline

GOAL
Run all stages in order, cache the U-Net output, collect every intermediate result
(architecture p.1 "Application layer").

INTERFACE (keep)
  CellCountingPipeline.run(img: np.ndarray, params: Params) -> PipelineResult

DESIGN
- __init__(weights: str = str(WEIGHTS_PATH), device: Optional[torch.device] = None):
  device = get_device() if None; create Preprocessor, MaskRefiner, WatershedSplitter,
  CellAnalyzer; load UNetSegmenter LAZILY on first run (so the UI can start and show
  an error message if weights are missing).
- U-Net cache: key = (hash of input image bytes + shape, params.clip, params.blur_k).
  If key unchanged, reuse stored prob map — moving τ / kernel / peak sliders must NOT
  re-run the network. Keep only the last entry (memory).
- Each stage inside stage_timer(logger, "<stage>") from utils.logger.

STAGES → PipelineResult.images keys (exact names; UI depends on them)
  "original"     input as RGB uint8
  "preprocessed" Preprocessor output
  "prob"         prob_heatmap(prob)
  "mask"         mask_view(mask)
  "distance"     distance_map_view(dist, labels)
  "overlay"      overlay_contours(original, labels_renumbered, cells)
Also fill raw-array fields (already in config.py since Phase 1):
  result.prob (float32 prob map), result.mask (uint8 0/255), result.labels (renumbered).
labels = CellAnalyzer.renumber(raw_labels, params.min_area);
cells = CellAnalyzer.run(raw_labels, params.min_area); count = len(cells).

VALIDATION
- Params range check (thresh in (0,1), kernel ≥ 1, etc.) → ValueError with which field.

DONE WHEN
- With trained or random weights: run(sample, Params()) returns all 6 image keys, count ≥ 0.
- Second run with only thresh changed is much faster (log shows U-Net stage skipped).
- Timing lines for every stage appear in outputs/app.log.
```

---

## M10 — Gradio UI

```text
MODULE: app.py — build_ui()

GOAL
Interactive web UI matching the mockup (Docs/Architechture.pdf p.5) at localhost:7860.

INTERFACE (keep)
  build_ui() -> gr.Blocks ; `python app.py` launches it.

LAYOUT (gr.Blocks, title "Cell Segmentation & Counting")
Left column:
  1. Upload image (gr.Image type="numpy", accepts PNG/JPG/TIF)
  2. Sliders with defaults from Params():
       Probability threshold τ   0.05–0.95 step 0.01  (0.50)
       Morphology kernel size    1–15 step 2          (5)
       Peak ratio r              0.1–0.9 step 0.05    (0.40)
       Min peak distance (px)    1–50 step 1          (10)
       Min cell area (px²)       0–1000 step 10       (50)
       CLAHE clip limit          0.5–8.0 step 0.1     (2.0)
     (blur_k stays at default 3 — not in mockup)
  Buttons: "Run segmentation" (primary), "Reset" (restores defaults, clears outputs).
  Example images from samples/ via gr.Examples (if folder has images).
Right area, 2×3 grid:
  Original image | Probability heatmap | Binary mask
  Distance map + seeds | Segmented & numbered cells | Cell area distribution (gr.Plot)
Bottom row: stat cards Cells detected / Mean area / Median area / Device,
  "Download CSV" (gr.File) — CSV written to outputs/<timestamp>_cells.csv.
Status message box (gr.Markdown) for warnings/errors.

BEHAVIOUR
- One pipeline instance created at startup (global), reused for all requests.
- Callback wraps pipeline.run in try/except: on any exception show a readable
  message in the status box and return empty outputs — app never crashes (p.6).
- No cells found → hint "No cells found — try lowering τ or the min cell area." (p.6)
- No image uploaded → "Please upload an image first."

DONE WHEN
- python app.py starts; page loads at http://localhost:7860.
- Upload sample → all 6 panels + stats + CSV download work.
- Missing weights file → UI starts and shows the weights error message on Run.
- Reset restores all sliders to defaults.
```

---

## M11 — Training dataset + loss

```text
MODULE: training/dataset.py — DSB2018Dataset ; training/losses.py — BCEDiceLoss

GOAL
Prepare Kaggle Data Science Bowl 2018 (stage1_train: 670 images, each folder has
images/<id>.png and masks/*.png instance masks) for U-Net training (p.3).

INTERFACE (keep)
  DSB2018Dataset(root: str, split: str = "train", size: int = 256)
  __len__() -> int ; __getitem__(idx) -> Tuple[Tensor image 3×S×S float, Tensor mask 1×S×S float]
  BCEDiceLoss.forward(logits, target) -> Tensor (scalar)

DATASET
- root = path to stage1_train/. List sample ids sorted, then deterministic split
  80/10/10 with fixed seed 42 → "train" / "val" / "test".
- Image: read RGB (drop alpha) → Preprocessor().run(img, clip=2.0, blur_k=3)
  (SAME as inference) → stack to 3 channels.
- Mask: union (logical OR) of all instance masks → {0,1}.
- Resize both to size×size (image INTER_LINEAR, mask INTER_NEAREST).
- Augment only for split == "train" (albumentations): HorizontalFlip, VerticalFlip,
  RandomRotate90, ElasticTransform (p=0.2), RandomBrightnessContrast (p=0.3).
- Normalise with the SAME ImageNet MEAN/STD constants as core/unet_model.py
  (import them — single source).
- Clear error if root missing, with download hint (Kaggle "data-science-bowl-2018").

LOSS
- L = BCEWithLogits + (1 − Dice), Dice computed on sigmoid(logits),
  smooth = 1.0, per-batch mean.

DONE WHEN
- len(train)+len(val)+len(test) == number of sample folders; splits do not overlap.
- ds[0] shapes (3,256,256) and (1,256,256), mask values in {0,1}.
- Save a grid of 4 augmented image/mask pairs to outputs/dataset_check.png.
- Loss: perfect logits (+10 where target=1, −10 else) ≈ 0; random ≈ > 0.5.
```

---

## M12 — Training loop

```text
MODULE: training/train.py — train()

GOAL
Fine-tune U-Net, save best checkpoint to weights/unet_cells.pth (p.3).

SETUP
- argparse CLI: --data (stage1_train path, required), --epochs 50, --batch 8,
  --lr 1e-4, --patience 8, --size 256, --workers 2 (0 on Windows if issues), --out WEIGHTS_PATH.
- Model: smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1).
- Optimiser Adam(lr), loss BCEDiceLoss, device from get_device().
- Seeds fixed (random, numpy, torch) for reproducibility.

LOOP
- Per epoch: train loss; validation loss, mean IoU and Dice (utils.metrics, pred > 0.5).
- Save state_dict when val Dice improves (save {"model": state_dict, "epoch", "val_dice"}).
  UNetSegmenter must load this format (M3 handles both).
- Early stop when val Dice does not improve for `patience` epochs.
- Log every epoch; write history to outputs/train_history.csv and a loss/Dice
  curve PNG to outputs/train_curves.png (for the report).
- At the end: evaluate best checkpoint on "test" split → IoU, Dice, cell-count error %
  (count via MaskRefiner + WatershedSplitter + CellAnalyzer on predictions vs GT instance count).
  Print a results table; save to outputs/test_metrics.json.
- Optional AMP when CUDA available.

DONE WHEN
- Smoke run: --epochs 1 on a 20-image subset (add --limit N flag) finishes on CPU
  and writes a checkpoint that `UNetSegmenter` loads.
- Full run produces train_curves.png, test_metrics.json, weights/unet_cells.pth.
```

---

## M13 — Unit tests

```text
MODULE: tests/ (pytest)

GOAL
Fast tests (< 30 s total, CPU, no dataset, no trained weights) proving each module's
contract.

FILES
- tests/conftest.py: add cell_seg/ to sys.path; fixtures: synthetic RGB image with
  N known discs (some touching), matching prob map, matching mask.
- test_preprocess.py, test_postprocess.py, test_watershed.py, test_analysis.py,
  test_metrics.py, test_visualize.py, test_pipeline.py.
- test_pipeline.py: monkeypatch UNetSegmenter with a fake returning the synthetic prob
  map → check count == N, all image keys present, cache hit on second run.
- Edge cases: empty image, all-black prob, single pixel cell below min_area,
  invalid params raise ValueError.

DONE WHEN
- `cd cell_seg && pytest -q` all green; paste output.
```

---

## M14 — Integration & robustness (Phase 3)

```text
MODULE: integration — core/unet_model.py, core/pipeline.py, app.py

GOAL
Implement the error-handling flow on Docs/Architechture.pdf p.6 and fix bugs found
running end-to-end on the real samples.

TASKS
1. Run app on every image in samples/ (≥ 3: sparse, dense/touching, noisy/low-contrast).
   Paste full tracebacks for any failure; fix root cause, one fix per round
   (each round logged in iterationlog.md — need ≥ 15 corrective rounds total).
2. OOM handling in UNetSegmenter.predict_prob:
   catch torch.cuda.OutOfMemoryError (or RuntimeError containing "out of memory") →
   torch.cuda.empty_cache() → retry with tiled inference (tiles 512×512, overlap 64,
   blend overlaps by averaging) → if still OOM, move model to CPU and retry; log each step.
3. Input validation in pipeline: unsupported dtype/channels, tiny images, huge images
   (> 4096 px side → downscale with warning, report scale factor).
4. Empty-result hint (no cells) surfaced in UI.
5. Very large images: check UI response time; report per-stage timing table.

DONE WHEN
- All samples run end-to-end without crash; outputs saved in outputs/.
- Forced OOM test (monkeypatch forward to raise OOM once) → tiled retry succeeds.
- pytest still green.
```

---

## M15 — Parameter tuning & ablation ("run first, optimise later")

```text
MODULE: experiments — new script cell_seg/ablation.py (+ notes)

GOAL
Satisfy assignment: ≥ 3 tuning dialogues and an ablation on the same image with
3 parameter sets, reporting PSNR / SSIM (and count).

TASKS
1. ablation.py: for one sample image, run pipeline with 3 Params sets, e.g.
     A: τ=0.3, kernel=3, r=0.3  (under-segmented / merged)
     B: defaults τ=0.5, kernel=5, r=0.4
     C: τ=0.7, kernel=7, r=0.6  (over-segmented / lost cells)
   Compare each mask/overlay to a reference (ground-truth mask if available; otherwise
   set B as reference and say so) → table: params, count, mean area, IoU/Dice (if GT),
   PSNR, SSIM, runtime. Save outputs/ablation.csv and a side-by-side figure
   outputs/ablation.png.
2. Tuning dialogues: for each, state the observed problem (e.g. "two touching cells
   counted as one"), the parameter changed, before/after count and image, and why.
   Log each as a round in iterationlog.md.

DONE WHEN
- ablation.csv + ablation.png exist; table pasted into the reply for the report.
- ≥ 3 tuning rounds logged.
```

---

## Iteration log entry template

```markdown
### Round N — YYYY-MM-DD · Phase X · <module>
- **Instruction:** <what I asked>
- **Agent response:** <what was produced / changed>
- **Problem found:** <error / wrong result, traceback snippet>
- **Feedback / correction rationale:** <what I told the agent to change and why>
```
