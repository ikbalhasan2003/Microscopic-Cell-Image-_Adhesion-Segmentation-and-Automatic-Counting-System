# Phase 1 — Architecture & Interface Definition (for review)

**Project:** Microscopic Cell Image — Adhesion Segmentation & Automatic Counting (Topic 3)
**Course:** Introduction to AI, Wuhan Institute of Technology · Advisor: Huang Qing
**Date:** 2026-10-10 · **Status:** approved by teacher (2026-10-10)

Every module is a signature-only stub (`raise NotImplementedError`). No logic will be written until this interface is approved. Diagrams are in `Docs/Architechture.pdf` (p.1 system, p.2 data flow, p.3 U-Net, p.4 structure, p.5 UI, p.6 error handling).

## 1. Pipeline

```
RGB image → Preprocessor → UNetSegmenter → MaskRefiner → WatershedSplitter → CellAnalyzer → UI / CSV
            (gray, CLAHE,   (H×W prob      (τ, open/close,  (distance map,      (id, area,
             blur)           map 0–1)        min area)        seeds, labels)      centroid, bbox)
```

## 2. Directory structure

```
cell_seg/
├── app.py              Gradio UI entry point
├── config.py           defaults, paths, @dataclass types
├── ablation.py         3-parameter-set comparison (PSNR / SSIM)
├── requirements.txt
├── core/               pipeline.py · preprocess.py · unet_model.py · postprocess.py · watershed.py · analysis.py
├── training/           dataset.py · losses.py · train.py
├── utils/              device.py · logger.py · visualize.py · metrics.py
├── weights/            unet_cells.pth
├── samples/            ≥ 3 test images
├── outputs/            results, CSV, logs
└── tests/              unit tests
```

## 3. Shared data types (`config.py`)

```python
@dataclass
class Params:                      # all UI slider values
    clip: float = 2.0              # CLAHE clip limit
    blur_k: int = 3                # Gaussian blur kernel
    thresh: float = 0.5            # probability threshold τ
    kernel: int = 5                # morphology kernel size
    peak_ratio: float = 0.4        # distance-map peak ratio r
    min_dist: int = 10             # min peak distance (px)
    min_area: int = 50             # min cell area (px²)

@dataclass
class CellInfo:
    id: int
    area: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]

@dataclass
class PipelineResult:
    images: Dict[str, np.ndarray]  # "original", "preprocessed", "prob", "mask", "distance", "overlay"
    cells: List[CellInfo]
    count: int
    prob: Optional[np.ndarray]     # raw probability map (for metrics / ablation)
    mask: Optional[np.ndarray]     # raw binary mask
    labels: Optional[np.ndarray]   # label image; value == CellInfo.id
```

## 4. Core module interfaces (`core/`)

| Class | Method | Signature |
|---|---|---|
| `Preprocessor` | `run` | `(img: np.ndarray, clip: float = 2.0, blur_k: int = 3) -> np.ndarray` |
| `UNetSegmenter` | `__init__` | `(weights: str, device: torch.device) -> None` |
| | `predict_prob` | `(img: np.ndarray) -> np.ndarray  # H×W, 0–1` |
| `MaskRefiner` | `run` | `(prob: np.ndarray, thresh: float, kernel: int, min_area: int) -> np.ndarray` |
| `WatershedSplitter` | `run` | `(img: np.ndarray, mask: np.ndarray, peak_ratio: float, min_dist: int) -> Tuple[np.ndarray, np.ndarray]` |
| `CellAnalyzer` | `run` | `(labels: np.ndarray, min_area: int) -> List[CellInfo]` |
| | `histogram` | `(cells: List[CellInfo]) -> Figure` |
| | `renumber` * | `(labels: np.ndarray, min_area: int) -> np.ndarray` |
| | `summary` * | `(cells: List[CellInfo]) -> Dict[str, float]` |
| | `to_dataframe` * | `(cells: List[CellInfo]) -> pd.DataFrame` |
| `CellCountingPipeline` | `__init__` * | `(weights: str = WEIGHTS_PATH, device: Optional[torch.device] = None)` |
| | `run` | `(img: np.ndarray, params: Params) -> PipelineResult` |

## 5. Utilities (`utils/`)

| Module | Function | Signature |
|---|---|---|
| `device.py` | `get_device` | `() -> torch.device` |
| | `device_label` * | `(device: torch.device) -> str` |
| `logger.py` | `get_logger` | `(name: str) -> logging.Logger` |
| | `stage_timer` * | `(logger, stage: str)` — context manager |
| `visualize.py` | `prob_heatmap` | `(prob) -> np.ndarray` |
| | `mask_view` * | `(mask) -> np.ndarray` |
| | `distance_map_view` * | `(dist, labels) -> np.ndarray` |
| | `overlay_contours` | `(img, labels, cells: List[CellInfo]) -> np.ndarray` |
| `metrics.py` | `iou`, `dice`, `psnr`, `ssim` | `(a, b) -> float` |
| | `count_error` * | `(pred_count: int, gt_count: int) -> float` |
| | `cell_precision_recall` * | `(pred_labels, gt_labels, iou_thr=0.5) -> Tuple[float, float]` |

## 6. Training (`training/`)

| Item | Signature / setting |
|---|---|
| `DSB2018Dataset` | `(root: str, split: str = "train", size: int = 256)`; `__getitem__ -> (img 3×256×256, mask 1×256×256)` |
| `BCEDiceLoss` | `forward(logits, target) -> Tensor`; L = BCE + (1 − Dice) |
| `train.py` | `parse_args()` *, `train()`; Adam 1e-4, batch 8, 50 epochs, early stop on validation Dice |
| Data | Kaggle DSB 2018, 670 images, split 80 / 10 / 10 (seed 42) |

## 7. Additions since the architecture slides (*)

Items marked * above are small helpers added while planning the modules. None of them changes an existing signature.

| Addition | Reason |
|---|---|
| `PipelineResult.prob / mask / labels` | metrics and ablation need the raw arrays, not only the RGB display images |
| `CellAnalyzer.renumber` | produces one label image whose values equal `CellInfo.id`, so the numbers drawn on the image match the CSV |
| `CellAnalyzer.summary`, `to_dataframe` | UI statistics cards (count, mean, median) and CSV export |
| `CellCountingPipeline.__init__` | loads the U-Net once and caches its output (p.1, "Caches U-Net output") |
| `device_label`, `stage_timer` | UI "Device" card; per-stage timing and try/except (p.6) |
| `mask_view`, `distance_map_view` | UI panels "Binary mask" and "Distance map + seeds" (p.5) |
| `count_error`, `cell_precision_recall` | evaluation metrics listed on p.3 |
| `MEAN`, `STD` constants in `unet_model.py` | training and inference use identical normalisation |
| `ablation.py` | required comparison of 3 parameter sets on the same image |

## 8. Questions for the reviewer

1. Are the added helper signatures (section 7) acceptable?
2. Is DSB 2018 acceptable as the training dataset, or is a course-provided dataset preferred?
