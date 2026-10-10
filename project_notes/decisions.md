# Design Decisions

| Date | Decision | Reason |
|---|---|---|
| 2026-10-10 | Code lives in `cell_seg/`, layout from Architechture.pdf p.4 | matches approved architecture slides |
| 2026-10-10 | Shared types (`Params`, `CellInfo`, `PipelineResult`) in `config.py` | single source for slider defaults + data contracts |
| 2026-10-10 | Default params: τ=0.5, kernel=5, r=0.4, min_dist=10, min_area=50, clip=2.0, blur_k=3 | from UI mockup (p.5) |
| 2026-10-10 | U-Net via `segmentation-models-pytorch` (ResNet34, ImageNet) | pretrained encoder, lightweight |
| 2026-10-10 | `CellAnalyzer.renumber` returns label image with value == `CellInfo.id` | overlay numbers must match CSV ids; avoids centroid lookup |
| 2026-10-10 | `PipelineResult` gets optional `prob`, `mask`, `labels` raw arrays | metrics + ablation need raw data; `images` holds only RGB views |
| 2026-10-10 | ImageNet `MEAN`/`STD` live in `core/unet_model.py`, imported by `training/dataset.py` | single source, identical train/inference normalisation |
| 2026-10-10 | No database; plain files (weights, CSV, logs, JSON) | single local user, stateless runs |
| 2026-10-10 | Python 3.11 conda env at `D:\envs\cellseg`; pkgs/pip cache/dataset on D: | system Python 3.14 too new for ML stack; C: only 7 GB free |
| 2026-10-10 | torch from cu124 wheel index | use RTX 2060 SUPER GPU; default pip wheel may be CPU-only |
| 2026-10-10 | `opencv-python-headless` only (not `opencv-python`) | albumentations needs headless; two builds clash in same `cv2` folder; no GUI windows needed |
| 2026-10-10 | `requirements-lock.txt` with exact versions | reproducible install; installed majors (gradio 6, pandas 3, numpy 2, cv2 5) newer than prompt assumptions |
| 2026-10-10 | Dataset from BBBC038 mirror instead of Kaggle API | same DSB 2018 data, public, no Kaggle token needed |
| 2026-10-10 | Samples: sparse / dense-touching / low-contrast from stage1_train, GT in `samples/gt/` | covers the 3 required cases; GT enables IoU/Dice in ablation |
| 2026-10-10 | Loggers set `propagate=False`, handlers added only if logger has none | prevents duplicate lines when `get_logger` called repeatedly or root logger configured (e.g. by Gradio) |
| 2026-10-10 | `stage_timer` re-raises after `logger.exception` | pipeline/UI decide how to show error; log keeps full traceback + elapsed time |
| 2026-10-10 | Preprocessor rejects dtypes other than uint8/uint16/float (e.g. int32) | prompt lists only these; silent cast of signed ints could wrap values |
| 2026-10-10 | No intensity inversion for bright-field vs fluorescence | DSB training set has both polarities; U-Net learns both; keeps train/inference preprocessing identical |
| 2026-10-10 | Public helper `build_unet(encoder_weights)` in `core/unet_model.py` | train (ImageNet init) and inference (no init, load weights) must build the same architecture; one definition |
| 2026-10-10 | `torch.load(..., weights_only=True)` | safe loading (no pickle code execution); checkpoint holds only tensors + int/float |
| 2026-10-10 | Reflect-pad to multiple of 32 with `np.pad`, crop after | encoder downsamples 5×; reflect avoids dark border artefacts; np.pad works for any pad size |
| 2026-10-10 | Small-object removal via `ndimage.label` + `np.bincount`, not `skimage.morphology.remove_small_objects` | skimage 0.26 deprecated `min_size` (→ `max_size`, "≤" instead of "<"); own code behaves the same on skimage 0.19–0.26, keeps "area < min_area removed" |
| 2026-10-10 | 8-connectivity for mask components | diagonal-touching pixels belong to one blob; matches OpenCV default |
| 2026-10-10 | Prob map tolerance 1e-4 outside [0,1]; NaN/inf rejected | sigmoid/float rounding can slightly overshoot; NaN would silently become background |
| 2026-10-10 | Keep default min_area=50 / kernel=5 for now despite `01_sparse` losing small nuclei | defaults are from approved mockup; revisit as tuning dialogue in M15 with trained weights |
| 2026-10-10 | Watershed seeds: per-blob threshold `dist ≥ peak_ratio × max dist of that blob`, not global max | global max left 10 of 30 blobs on dense sample without a seed (small cells beside big ones) |
| 2026-10-10 | Every blob gets ≥ 1 seed (argmax added if none) | no mask blob may vanish from the count |
| 2026-10-10 | Seed pixels labelled with 8-connectivity before watershed | flat distance ridges give adjacent equal peaks; merge them so one cell = one marker |
| 2026-10-10 | `watershed(..., connectivity=2)` | matches 8-connected blobs from MaskRefiner; no unlabelled pixels inside mask |
| 2026-10-10 | Keep default min_dist=10 for now; min_dist=5 fits dense sample better | defaults from mockup; revisit in M15 tuning with trained weights |
| 2026-10-10 | Reading order: sort by (centroid row // median cell bbox height, centroid col) | cells on one visual line read left→right even if centroids differ by a few px |
| 2026-10-10 | Histogram y-axis integer ticks | counts are integers |
| 2026-10-10 | bbox in CSV uses skimage convention (max row/col exclusive) | direct from regionprops; documented for report |
| 2026-10-10 | Build order M6 → M8 → M11 → M12 (train) → M7 → M9 → M10 → M13 → M14 → M15 | training is the longest step; GPU trains while UI modules are coded |
| 2026-10-10 | `count_error` with gt = 0 divides by 1 | finite value for CSV/JSON; 0 vs 0 → 0 % |
| 2026-10-10 | Per-cell P/R: one side empty → (0, 0); both empty → (1, 1) | empty prediction must not score as perfect |
| 2026-10-10 | PSNR/SSIM require same dtype; data_range from dtype (uint16 → 65535) | mixing uint8 and float silently gives wrong dB values |
| 2026-10-10 | Dataset caches preprocessed + resized images/masks in RAM at construction | mask union from hundreds of PNGs is slow; ~90 MB for 670 × 256² × 2 |
| 2026-10-10 | ElasticTransform alpha 50, sigma 7 | albumentations 2.x defaults (alpha 1, sigma 50) are a no-op |
| 2026-10-10 | Keep seed-42 split even though samples 02/03 land in train | moving them would bias the test split; report states it |
| 2026-10-10 | Target = nuclei (DSB labels nuclei only) | ground truth defines what is counted; stated in report |
| 2026-10-10 | Contours via `find_boundaries(mode="inner")` on renumbered labels (not per-cell `findContours`) | one vectorised pass; touching cells get a boundary between them |
| 2026-10-10 | Overlay font scale clip(min(H,W)/700, 0.3, 1.2), line thickness grows > 1000 px | readable on 256 px samples and large microscope images |
| 2026-10-10 | `get_device` requires `device_count() > 0` besides `is_available()` | torch on Windows reports available with `CUDA_VISIBLE_DEVICES=""` but has 0 devices |
| 2026-10-10 | Pipeline cache key = SHA-1(image bytes) + shape + dtype + clip + blur_k | only inputs that change the U-Net output; slider moves on later stages reuse the prob map |
| 2026-10-10 | `visualize.to_rgb_uint8` public | pipeline "original" panel and overlay use one conversion |
| 2026-10-10 | Test evaluation also through the real inference path at native resolution | 256 px metrics alone would hide resize effects; app runs at native size |
| 2026-10-10 | `HF_HUB_OFFLINE=1` when training here | HF download hangs on this network; smp falls back to download.pytorch.org torchvision weights |
| 2026-10-10 | AMP on CUDA | 3.6 s/epoch on RTX 2060 SUPER; no accuracy issue seen |
| 2026-10-10 | Output images `format="png"`, fixed height 260 px | Gradio 6 defaults to lossy webp; equal heights keep 2×3 grid aligned |
| 2026-10-10 | CSV name has millisecond timestamp | two quick runs in one second overwrote each other |
| 2026-10-10 | App binds 127.0.0.1 only, `allowed_paths=[outputs]` | local single-user tool; Gradio may only serve files from outputs/ |
| 2026-10-10 | `train.py` inserts `cell_seg/` at front of `sys.path` | lets `python training/train.py` import project `config`/`core`/`utils` instead of same-named pip packages (flagged by security review as module shadowing; accepted — folder is the project's own code); `python -m training.train` also works |
| 2026-10-10 | `train.py` raises if no checkpoint saved this run | NaN val Dice would otherwise evaluate an old `unet_cells.pth` and report wrong test metrics |
| 2026-10-10 | Tests use a fake U-Net via monkeypatch + synthetic 5-disc scene | fast (< 5 s), no weights/dataset, exact known count |
| 2026-10-10 | No `allowed_paths` in `launch()` | security review: it exposed all of outputs/ (logs, old CSVs); Gradio serves returned files from its own cache anyway |
| 2026-10-10 | Generic UI error shows exception type only | avoid leaking paths/internals; full traceback in app.log |
| 2026-10-10 | OOM chain: full → 512 px tiles (64 overlap, averaged) → CPU | tiles match full inference (99.98 % mask agreement) |
| 2026-10-10 | Wrong-architecture weights → ValueError with plain message | raw state_dict error is unreadable in the UI |
| 2026-10-10 | Images > 4096 px downscaled (INTER_AREA); `PipelineResult.warnings` added | 4096 px peaks at 5.2 GB GPU; user told scale factor and area correction |
