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
- **Problem found:** None. All done-when checks passed first try.
- **Feedback / rationale:** Checked the output myself: device chosen = CUDA with GPU name, logged only once even when `get_device()` is called twice. Timer prints exactly one line. Calling `get_logger` twice does not double the lines (`propagate=False` stops Gradio's root logger from repeating them). Failure test shows full traceback in `outputs/app.log` and the error still reaches the caller, so the UI can show it. Accepted.

### Round 6 — 2026-10-10 · Phase 2 · M2 preprocessing
- **Instruction:** Start M2 (`core/preprocess.py`, `Preprocessor.run`) from `module_prompts.md`.
- **Agent response:** Implemented validate → to-uint8 → RGB→gray → CLAHE (8×8 tiles) → Gaussian blur. uint16 scaled by image max (all-zero → zeros); float clipped to [0,1]×255 if max ≤ 1 else [0,255], NaN → 0; alpha dropped; H×W×1 accepted; even `blur_k` → `blur_k+1`, `blur_k ≤ 1` → no blur. ValueError for empty, <32 px side, wrong ndim/channels, unsupported dtype (e.g. int32), `clip ≤ 0`. Checks: all 7 input types (RGB/gray/H×W×1/RGBA/uint16/float01/float255, 300×451) → (300, 451) uint8; low-contrast synthetic std 5.05 → clip 1.0: 9.40, clip 4.0: 22.18; blur_k 0/1/2/3/4/7 no crash; 6 invalid inputs raise clear ValueError; `outputs/preprocess_check.png` on `03_low_contrast.png` (gray std 35.5 → 48.5, nuclei edges sharper).
- **Problem found:** No code error. Observation: grayscale polarity differs by sample (fluorescence = bright nuclei on dark, H&E = dark nuclei on light). Not inverted here — U-Net is trained on DSB which has both, so it must learn both.
- **Feedback / rationale:** Looked at `outputs/preprocess_check.png`: nuclei in the H&E sample are clearer after CLAHE (std 35.5 → 48.5) and blur 3 does not wash out edges. Higher clip gives more contrast as expected (9.40 → 22.18). Agreed not to invert polarity: training data has both kinds, and train/inference must use the same preprocessing. Accepted.

### Round 7 — 2026-10-10 · Phase 2 · M3 U-Net segmenter
- **Instruction:** Start M3 (`core/unet_model.py`, `UNetSegmenter`) from `module_prompts.md`.
- **Agent response:** `__init__`: FileNotFoundError with train/copy hint if weights missing; `torch.load(map_location=device, weights_only=True)`; accepts raw state_dict or `{"model": ...}` checkpoint; strict load; `.to(device).eval()` once; MEAN/STD tensors cached on device. `predict_prob`: validates H×W uint8, reflect-pads to multiple of 32, 1→3 channels, /255, ImageNet normalise, `no_grad` + sigmoid, crops padding, returns float32. Added `build_unet(encoder_weights)` so `train.py` (M12) builds the identical model. Checks: missing weights → clear FileNotFoundError; random weights 300×451 on CPU → (300, 451) float32, min 0.0 max 1.0, 0.13 s; raw state_dict and checkpoint dict give identical output; RGB/float/empty input → ValueError; wrong-architecture file → RuntimeError "Error(s) in loading state_dict"; real sample 256×320 and odd 250×317 crop work on CPU and CUDA.
- **Problem found:** No code error. Observations: (1) random-weight probs saturate at exactly 0/1 — expected for untrained BatchNorm running stats, real check comes after training (M12). (2) wrong-architecture weights give a long raw RuntimeError — UI message for this handled in M14.
- **Feedback / rationale:** Output shape, dtype and [0,1] range correct on CPU and GPU, including odd image size, so padding/cropping works. Saturated 0/1 values are fine for random weights; will re-check with trained weights in M12. Agent added public `build_unet()` outside the approved interface — I accepted it, because training and inference must build the exact same model and one shared function prevents mismatch. Raw RuntimeError for wrong weights is left for M14 error handling. Accepted.
