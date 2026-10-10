# Microscopic Cell Image — Adhesion Segmentation & Automatic Counting

Course: Introduction to AI final project (Topic 3), Wuhan Institute of Technology. Advisor: Huang Qing.

**Goal:** stained microscopy image with touching cells → numbered cell contours, per-cell area, area histogram, count, CSV.

**Pipeline:** Preprocess (CLAHE + blur) → U-Net (ResNet34) probability map → threshold + morphology → distance transform + watershed → count & measure.

**Stack:** Python 3.8+, PyTorch, OpenCV, scikit-image, Gradio, Matplotlib.

## Layout
| Path | Purpose |
|---|---|
| `cell_seg/` | source code (see `Docs/Architechture.pdf` p.4) |
| `Docs/` | assignment brief, architecture PDF |
| `project_notes/` | tracking: README, progress, TODO, iteration log, decisions |

## Notes files
- `project_progress.md` — phase status
- `TODO.md` — task checklist
- `iterationlog.md` — required Agent dialogue log (deliverable)
- `decisions.md` — design decisions + reasons
- `module_prompts.md` — one prompt per module, build order
- `phase1_interface.md` — interface summary for instructor review

## Run (once implemented)
```
pip install -r cell_seg/requirements.txt
python cell_seg/app.py        # http://localhost:7860
```
