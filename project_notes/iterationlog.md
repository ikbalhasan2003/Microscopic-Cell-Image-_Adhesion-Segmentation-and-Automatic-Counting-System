# Iteration Log (Agent dialogue)

Format: **Round X** — timestamp · My instruction → Agent response → My feedback / correction rationale.

---

### Round 1 — 2026-10-10 · Phase 1
- **Instruction:** Make file structure for project; keep a folder for tracking files.
- **Agent response:** Created `cell_seg/` tree per architecture (core/, training/, utils/, weights/, samples/, outputs/, tests/) with typed signature stubs; created `project_notes/` (README, progress, TODO, iterationlog, decisions).
- **Feedback / rationale:** Checked the folder tree against Architechture.pdf p.4: same layout (core/, training/, utils/, weights/, samples/, outputs/, tests/). Stubs only, no logic, which follows the Phase 1 rule. Accepted.

### Round 2 — 2026-10-10 · Phase 1
- **Instruction:** Divide project module by module; write a complete, understandable prompt for each and save in a prompt file.
- **Agent response:** Created `module_prompts.md`: shared-context block + 15 module prompts (M1 device/logger … M15 ablation) in dependency order, each with goal, fixed interface, steps, edge cases, "done when" checks.
- **Feedback / rationale:** Read all 15 prompts. Build order is correct: each module depends only on earlier ones, and M4–M8 can be tested without trained weights. Each prompt has clear 'Done when' checks I can verify. Accepted.

### Round 3 — 2026-10-10 · Phase 1
- **Instruction:** Complete Phase 1.
- **Agent response:** Added typed stubs for helpers introduced in module prompts (device_label, stage_timer, renumber/summary/to_dataframe, mask_view/distance_map_view, count_error/cell_precision_recall, pipeline `__init__`, train `parse_args`, `ablation.py`); added raw-array fields to `PipelineResult`; MEAN/STD constants in `unet_model.py`. Wrote `phase1_interface.md` for instructor review. All files parse.
- **Feedback / rationale:** Reviewed phase1_interface.md. New helpers are small and change no signature from the architecture slides. Raw arrays in PipelineResult are needed for metrics/ablation. Sent to teacher; teacher approved the interface on 2026-10-10.

### Round 4 — 2026-10-10 · Pre-Phase 2 setup
- **Instruction:** Check what preparation is needed before module work; then do all setup steps 1–8. Commit and push everything to my GitHub repo.
- **Agent response:** Found blockers: no packages, Python 3.14 too new, C: 97% full, no conda, no samples/dataset. Installed Miniconda + Python 3.11 env on D:, CUDA torch, requirements (+lock file), VS Code interpreter, DSB 2018 data, 3 samples with GT. Initialised git, pushed to GitHub.
- **Problem found:** `opencv-python` and `opencv-python-headless` both installed (same `cv2` folder) → kept headless only, updated requirements.txt.
- **Feedback / rationale:** Agreed with the blockers found: Python 3.14 too new, C: almost full, so env and data go on D: with Python 3.11. Checked GPU works (torch.cuda.is_available() = True). Duplicate OpenCV fix is reasonable. Samples cover the 3 cases I need (sparse, touching, low contrast). Everything pushed to GitHub. Ready for Phase 2.

### Round 5 — 2026-10-10 · Phase 2 · M1 device + logging
- **Instruction:** Start M1 (device + logging) from `module_prompts.md`.
- **Agent response:** Implemented `utils/logger.py`: `get_logger` (console + `outputs/app.log`, format `time | level | name | msg`, INFO, `propagate=False`, handlers added only once per name) and `stage_timer` (logs "<stage> done in X.XX s"; on exception logs traceback via `logger.exception` and re-raises). Implemented `utils/device.py`: `get_device` (CUDA else CPU, logs choice + GPU name once per process) and `device_label` ("CUDA (GPU)" / "CPU", ValueError on non-device input). Done-when checks: `cuda CUDA (GPU)`, one "sleep done in 0.10 s" line, `get_logger('t')` twice → same object, 2 handlers, no duplicate lines; failure path logs traceback and re-raises; all lines also in `outputs/app.log`.
- **Problem found:** _(none yet — fill after review)_
- **Feedback / rationale:** _(fill in)_
