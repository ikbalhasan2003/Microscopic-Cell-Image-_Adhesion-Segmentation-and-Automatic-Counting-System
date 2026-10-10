# Sample images

Source: Kaggle Data Science Bowl 2018, `stage1_train` (public mirror: Broad BBBC038, CC0).
Picked by script: instance count, touching ratio (instances ÷ connected blobs), foreground/background contrast.

| File | DSB id | Size (H×W) | GT cells | Case |
|---|---|---|---|---|
| 01_sparse.png | `ec486143ecfec847…` | 256×256 | 11 | fluorescence, isolated nuclei |
| 02_dense_touching.png | `a31deaf0ac279d5f…` | 256×320 | 87 | stained, 87 cells in 30 blobs (touch ratio 2.9) |
| 03_low_contrast.png | `4193474b2f1c72f7…` | 256×320 | 12 | H&E tissue, lowest contrast |

`gt/<name>_mask.png` — binary union mask (0/255). `gt/<name>_labels.png` — uint16 instance labels (1..N).

**Note:** these ids are in `stage1_train`. If the seed-42 split (M11) puts one in the train split, its scores are optimistic; check during M12 and mention it in the report.
