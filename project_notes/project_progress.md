# Project Progress

| Phase | Scope | Status |
|---|---|---|
| 1 — Architecture | dir structure + typed class signatures | ✅ complete; interface approved by teacher (2026-10-10) |
| 2 — Core coding | modules one by one, ≥3 tuning dialogues | ✅ M1–M12 + M15 done; 4 tuning rounds |
| 3 — Integration & debug | tracebacks → targeted fixes | ✅ M13 tests + M14 integration done |
| Report | PDF + iterationlog.md + ablation | ⬜ not started |

## Log
- **2026-10-10** — Created `cell_seg/` skeleton per Architechture.pdf p.4. All modules are signature stubs (`NotImplementedError`). Created `project_notes/`.
- **2026-10-10** — Wrote `module_prompts.md` (per-module prompts, build order M1–M15).
- **2026-10-10** — Phase 1 complete: helper stubs added, `PipelineResult` raw fields, `phase1_interface.md` written for review.
- **2026-10-10** — Git repo initialised, pushed to GitHub (`main`).
- **2026-10-10** — Environment ready: Miniconda `D:\miniconda3`, env `D:\envs\cellseg` (Python 3.11.17), torch 2.6.0+cu124 on RTX 2060 SUPER, all requirements import, `requirements-lock.txt` written. DSB 2018 `stage1_train` (670) at `D:\datasets\dsb2018\stage1_train`. 3 samples + GT in `cell_seg/samples/`. Ready for M1.
- **2026-10-10** — Teacher approved Phase 1 interface. Iteration log Rounds 1–4 feedback filled. Next: M1 in a new session.
- **2026-10-10** — M1 done: `utils/device.py` (GPU→CPU, `device_label`) + `utils/logger.py` (console + `outputs/app.log`, `stage_timer`). All done-when checks pass on RTX 2060 SUPER. Next: M2 preprocessing.
- **2026-10-10** — M2 done: `core/preprocess.py` (to-uint8, RGB→gray, CLAHE, odd-kernel blur, input validation). Checks pass; `outputs/preprocess_check.png` saved. Next: M3 U-Net segmenter.
- **2026-10-10** — M3 done: `core/unet_model.py` (weights load raw/checkpoint, pad-to-32 inference, `build_unet` shared builder). Checks pass on CPU + CUDA with random weights. Next: M4 mask refinement.
- **2026-10-10** — M4 done: `core/postprocess.py` (threshold, elliptical open/close, hole fill, small-blob removal, input validation). Synthetic checks pass; `outputs/postprocess_check.png` saved. Found tiny nuclei in `01_sparse` lost at default min_area=50 → tuning item for M15. Next: M5 watershed.
- **2026-10-10** — M5 done: `core/watershed.py` (distance transform, per-blob peak threshold, ≥1 seed per blob, watershed). Checks pass; `outputs/watershed_check.png` saved. Tuning item for M15: min_dist 5 matches dense GT (88 vs 87) better than default 10 (65). Next: M6 cell analysis.
- **2026-10-10** — M6 done: `core/analysis.py` (renumber reading order, regionprops → CellInfo, summary, DataFrame, histogram Figure). Checks pass; `outputs/hist_check.png`. Build order changed: M8 → M11 → M12 next so full training runs on the GPU while M7, M9, M10 are written.
- **2026-10-10** — M8 done: `utils/metrics.py` (IoU, Dice, PSNR, SSIM, count error, greedy per-cell P/R). All checks pass. Next: M11 dataset + loss.
- **2026-10-10** — M11 done: `training/dataset.py` (seed-42 80/10/10 split, shared preprocessing, RAM cache, augmentations) + `training/losses.py` (BCE + Dice). Checks pass; `outputs/dataset_check.png`. Samples 02/03 fall in train split (optimistic), 01 in test. Next: M12 training loop.
- **2026-10-10** — M7 done: `utils/visualize.py` (fixed-scale viridis heatmap, mask view, bone distance map + red seeds, green contours + outlined ids). Checks pass; `outputs/vis_check_*.png`.
- **2026-10-10** — M1 fix: `get_device` also requires `device_count() > 0` (crash with hidden GPU found during M12 CPU smoke test).
- **2026-10-10** — M9 done: `core/pipeline.py` (lazy U-Net, last-entry prob cache, per-stage timing, field-named param validation). Cache hit 0.03 s vs 0.42 s first run.
- **2026-10-10** — M12 done: `training/train.py`. Full GPU training 28 epochs (early stop), best val Dice 0.902; test Dice 0.914 (256 px) / 0.907 (native, full pipeline), count error median 10 %, cell P/R 0.87/0.78. Weights at `cell_seg/weights/unet_cells.pth` (not in git).
- **2026-10-10** — M10 done: `app.py` Gradio UI (mockup layout, sliders, Reset, examples, stats, CSV, error/empty hints). Tested in browser (Playwright) + gradio_client incl. missing-weights instance. Screenshot `outputs/ui_check.png`.
- **2026-10-10** — Security-review fix in `train.py`: refuse to evaluate a stale checkpoint when no epoch saved one.
- **2026-10-10** — M13 done: `tests/` — 58 pytest tests, all green in 4.4 s (CPU, fake U-Net, synthetic discs).
- **2026-10-10** — Security fixes in `app.py`: no `allowed_paths` (outputs/ not served), generic errors show type only.
- **2026-10-10** — M14 done: OOM → tiled → CPU fallback, clear wrong-weights error, > 4096 px downscale with UI warning; all samples + odd inputs run; timing table in iterationlog Round 21. Weak point: instance splitting on dense sample (36/87) → M15.
- **2026-10-10** — M15 done: `tuning.py` (288-setting val grid search) → defaults kernel 3, min_area 20 (test F1 0.823 → 0.852, median count error 10 % → 6.9 %). `ablation.py` → `outputs/ablation.csv` + `ablation.png`. Tuning rounds 22–25 logged. All modules M1–M15 complete.
