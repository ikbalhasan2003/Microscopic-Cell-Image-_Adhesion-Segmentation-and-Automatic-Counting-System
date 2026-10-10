# TODO

## Phase 1 — Architecture
- [x] Directory structure
- [x] Typed class/function signatures
- [x] Interface summary for review (phase1_interface.md)
- [ ] Get interface approved by instructor/TA

## Phase 2 — Core modules
- [ ] `utils/device.py` — GPU→CPU fallback
- [ ] `utils/logger.py` — logging, per-stage timing
- [ ] `core/preprocess.py` — CLAHE + blur
- [ ] `core/unet_model.py` — load ResNet34 U-Net, predict prob map
- [ ] `core/postprocess.py` — threshold + morphology
- [ ] `core/watershed.py` — distance transform + watershed
- [ ] `core/analysis.py` — numbering, area, histogram
- [ ] `utils/visualize.py` — heatmap, contour overlay
- [ ] `utils/metrics.py` — IoU, Dice, PSNR, SSIM
- [ ] `core/pipeline.py` — orchestrator + U-Net cache
- [ ] `app.py` — Gradio UI, sliders, CSV export
- [ ] ≥3 "run first, optimize later" tuning dialogues

## Training
- [ ] Download DSB 2018 (Kaggle)
- [ ] `training/dataset.py`, `losses.py`, `train.py`
- [ ] Train → `weights/unet_cells.pth`

## Phase 3 — Integration
- [ ] End-to-end run on samples
- [ ] OOM → tile + retry; empty-result hint
- [ ] `tests/` unit tests

## Deliverables
- [ ] ≥3 test images in `cell_seg/samples/`
- [ ] Ablation: same image, 3 param sets, PSNR/SSIM
- [ ] `iterationlog.md` ≥15 corrective rounds
- [ ] Report PDF (paper review, method, flowchart, results table, summary)
- [ ] requirements.txt installs clean in Anaconda
- [ ] Package ZIP
