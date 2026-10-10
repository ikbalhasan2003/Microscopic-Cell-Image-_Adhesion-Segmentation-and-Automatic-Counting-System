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
