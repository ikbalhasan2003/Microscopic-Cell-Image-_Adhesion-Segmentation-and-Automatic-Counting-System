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
