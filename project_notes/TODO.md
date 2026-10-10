# TODO

## Phase 1 — Architecture
- [x] Directory structure
- [x] Typed class/function signatures
- [x] Interface summary for review (phase1_interface.md)
- [x] Get interface approved by instructor/TA (2026-10-10)

## Pre-phase-2 setup
- [x] Git repo + GitHub push
- [x] Conda env (Python 3.11, CUDA torch) on D:
- [x] requirements install clean + lock file
- [x] VS Code interpreter set
- [x] Dataset downloaded
- [x] 3 sample images + GT

## Phase 2 — Core modules
- [x] `utils/device.py` — GPU→CPU fallback
- [x] `utils/logger.py` — logging, per-stage timing
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
- [x] Download DSB 2018 (BBBC038 mirror → `D:\datasets\dsb2018\stage1_train`)
- [ ] `training/dataset.py`, `losses.py`, `train.py`
- [ ] Train → `weights/unet_cells.pth`

## Phase 3 — Integration
- [ ] End-to-end run on samples
- [ ] OOM → tile + retry; empty-result hint
- [ ] `tests/` unit tests

## Deliverables
- [x] ≥3 test images in `cell_seg/samples/`
- [ ] Ablation: same image, 3 param sets, PSNR/SSIM
- [ ] `iterationlog.md` ≥15 corrective rounds
- [ ] Report PDF (paper review, method, flowchart, results table, summary)
- [ ] requirements.txt installs clean in Anaconda
- [ ] Package ZIP
