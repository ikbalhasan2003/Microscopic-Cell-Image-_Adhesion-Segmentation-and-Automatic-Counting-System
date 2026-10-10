# Project Progress

| Phase | Scope | Status |
|---|---|---|
| 1 — Architecture | dir structure + typed class signatures | ✅ complete; interface approved by teacher (2026-10-10) |
| 2 — Core coding | modules one by one, ≥3 tuning dialogues | 🟨 in progress (M1–M6 done; next: M8, M11, M12 so training runs early) |
| 3 — Integration & debug | tracebacks → targeted fixes | ⬜ not started |
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
