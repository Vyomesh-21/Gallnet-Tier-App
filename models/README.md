# Model metadata

`setup_windows.bat` downloads the **Self-Attention-Guided Residual CNN** authors' checkpoint as `model-031-0.979685-0.992509.keras` and the reference notebook. This is **not the original GallNet architecture**.

Create `labels.json` containing **exactly nine class names in the model's actual output index order**, derived from training code. Do not copy the ordering of disease names in the paper or README without confirming it is the model's class-index ordering. Supported canonical labels are: gallstones; cholecystitis; polyps and cholesterol crystals; adenomyomatosis; wall thickening; membranous and gangrenous cholecystitis; perforation; carcinoma; abdomen and retroperitoneum. See `backend/tier_mapping.py` for aliases.

Also inspect notebook training preprocessing and set `MODEL_PREPROCESS` in the Windows backend launcher to one of `zero_one`, `zero_255`, or `minus_one_one` **only if it matches**. If the notebook uses additional preprocessing, modify the adapter accordingly.

No fake default metadata is supplied because incorrect label order would silently produce wrong severity results.
