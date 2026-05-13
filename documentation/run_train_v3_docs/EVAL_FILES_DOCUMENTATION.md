# Evaluation Module Documentation

Complete guide to all Python files in the `eval/` folder for AFAGNetV3 deepfake detection model.

---

##  Table of Contents

1. [evaluate.py](#evaluatepy) - Core evaluation metrics computation
2. [run_eval_v3.py](#run_eval_v3py) - Complete model evaluation pipeline
3. [cross_dataset.py](#cross_datasetpy) - Cross-dataset evaluation runner
4. [cross_dataset_eval.py](#cross_dataset_evalpy) - Interactive cross-dataset evaluation
5. [ablation.py](#ablationpy) - Component ablation study
6. [ablation_cross_dataset.py](#ablation_cross_datasetpy) - Cross-dataset ablation study
7. [psuedo_ablation.py](#psuedo_ablationpy) - Pseudo-mask weight ablation
8. [robustness_eval_v3.py](#robustness_eval_v3py) - Adversarial robustness evaluation

---

## evaluate.py

**Purpose:** Core evaluation module that computes comprehensive metrics for AFAGNetV3.

### Key Features
- **Metrics Computation:**
  - Per-sample: Accuracy, Precision, Recall, F1, AUC, IoU
  - Per-domain: Domain-specific metrics for each manipulation method
  - Overall: Aggregate metrics across all samples

- **Domain Handling:**
  - Real samples: YouTube, Actors
  - Fake samples: Deepfakes, Face2Face, FaceSwap, FaceShifter, NeuralTextures
  - Cross-domain AUC: Each fake method evaluated AGAINST all real samples 

- **Visualization Charts:**
  1. `overall_metrics_bar.png` - Bar chart of all 6 metrics
  2. `domain_acc_bar.png` - Per-manipulation accuracy 
  3. `domain_auc_bar.png` - Per-manipulation AUC 
  4. `domain_table.png` - Complete metrics table image
  5. `confusion_matrix.png` - Binary confusion matrix
  6. `roc_curve_overall.png` - Overall ROC curve
  7. `per_domain_roc.png` - Per-domain ROC curves overlaid
  8. `domain_radar.png` - Radar chart for multi-metric comparison

### Core Functions

#### `evaluate(model, loader, device, return_details=False, threshold=0.5)`
Runs full evaluation on a dataset loader.

**Parameters:**
- `model` - AFAGNetV3 model in eval mode
- `loader` - DataLoader yielding batches
- `device` - 'cuda' or 'cpu'
- `return_details` - If True, returns per-domain breakdown
- `threshold` - Decision threshold for classification (default 0.5)

**Returns:**
- `metrics` - Dict with keys: Accuracy, Precision, Recall, F1, AUC, IoU
- `details` (optional) - Dict containing:
  - `labels`, `preds`, `probs` - Raw predictions
  - `per_domain` - Metrics per manipulation method
  - `dom_labels`, `dom_preds`, `dom_probs` - Per-domain arrays

#### `compute_iou(pred_mask, gt_mask, threshold=0.5)`
Computes Intersection-over-Union between predicted and ground-truth masks.

#### `compute_domain_metrics(labels, preds, probs, iou_list)`
Computes all 6 metrics for a single domain, handling single-class domains gracefully.

#### `save_evaluation_visuals(...)`
Master function that saves all 8 charts at once.


---

## run_eval_v3.py

**Purpose:** Complete end-to-end evaluation pipeline for AFAGNetV3 with publication-quality reporting.

### Key Features
- Loads best/EMA/latest checkpoint with health checks (no NaN/Inf weights)
- Computes overall and per-domain metrics
- Finds optimal classification threshold via F1 sweep
- Generates 8 publication-quality charts
- Prints comparison table 
- Prints per-domain breakdown table

### Configuration
```python
ROOT = "FaceForensics_Data"
CACHE_DIR = "dataset/cache/ffpp_processed_v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FORCE_CHECKPOINT = None  # Leave None to auto-detect; set to specific path to override
```

### Workflow
1. Build FFPPDatasetV2 from `FaceForensics_Data` root
2. Load validation split (val_ratio=0.2, seed=42 — matches training split)
3. Resolve best checkpoint (checks health)
4. Sweep thresholds [0.05, 0.95] to find optimal F1
5. Evaluate model on validation set with optimal threshold
6. Print results + paper comparison table
7. Save 8 charts to `eval/<checkpoint_name>/`

### Run Command
```bash
python -m eval.run_eval_v3
```

---

## cross_dataset.py

**Purpose:** Simple cross-dataset evaluation runner that loads multiple datasets and evaluates model.

### Key Function

#### `run_cross_dataset(model_path, datasets_dict, device="cuda")`
Evaluates model on multiple datasets sequentially.

**Parameters:**
- `model_path` - Path to model checkpoint
- `datasets_dict` - Dict mapping dataset names to FFPPDatasetV2 instances
- `device` - Device to run on

**Returns:**
- `results` - Dict mapping dataset names to metric dicts

### Example
```python
from eval.cross_dataset import run_cross_dataset

datasets = {
    "FFPP": ffpp_dataset,
    "Celeb-DF": celeb_df_dataset,
}
results = run_cross_dataset("checkpoints/best_model.pth", datasets)
# results["FFPP"]["AUC"] → AUC on FFPP
```

---

## cross_dataset_eval.py

**Purpose:** Interactive cross-dataset evaluation with user dataset selection.

### Key Features
- **Interactive Dataset Selection:**
  - Lists available dataset roots
  - User can select by number or name
  - Supports multiple selections (comma-separated)
  - Option to select all (default)

- **Configuration:**
  ```python
  DATASET_ROOTS = {
      "FFPP": "FaceForensics_Data",
      "Celeb-DF": "path/to/celeb_df",
      "Celeb-DF-V2": "path/to/celeb_df_v2",
      "DFDC": "path/to/DFDC",
      "FDR": "path/to/FDR",
  }
  ```

### Workflow
1. List all available dataset roots
2. Prompt user to select dataset(s)
3. Build selected datasets
4. Resolve model checkpoint
5. Run evaluation on each dataset
6. Save visual outputs per dataset
7. Print summary for each dataset

### User Interaction
```
[INFO] Available datasets for validation:
  1. FFPP
  2. Celeb-DF
  3. Celeb-DF-V2
  0. All datasets
Enter dataset number(s) or name(s) separated by comma (default: all):
```

### Run Command
```bash
python -m eval.cross_dataset_eval
```

---

## ablation.py

**Purpose:** Component ablation study to measure contribution of each AFAGNetV3 component.

### Key Features
- Tests 6 ablation variants:
  1. **Full Model** - Baseline with all components
  2. **No Frequency** - Disable FrequencyBranch (set to zeros)
  3. **No Noise** - Disable NoiseBranch (set to zeros)
  4. **No CGAF** - Skip fusion, use only spatial branch
  5. **No Temporal** - Replace transformer with mean pooling
  6. **No F_high** - Disable high-level semantic features from backbone

- **AblationModelV3 Class:**
  - Inherits from AFAGNetV3
  - Disables components by setting outputs to zero or replacing modules
  - Loads same checkpoint weights with `strict=False`

### Metrics Per Variant
Each variant is evaluated on validation split (same as training: val_ratio=0.2, seed=42):
- Accuracy, Precision, Recall, F1, AUC, IoU
- Per-domain breakdowns
- Visual outputs saved per variant

### Output Structure
```
eval/ablation/
├── full_model/
│   ├── ablation_full_model_overall_metrics.png
│   ├── ablation_full_model_confusion_matrix.png
│   └── ...
├── no_frequency/
│   ├── ablation_no_frequency_overall_metrics.png
│   └── ...
├── ablation_auc_summary.png        # AUC comparison across all variants
└── ablation_auc.png                # Final AUC bar chart
```

### Summary Output
Prints table showing:
- Component name
- Accuracy, AUC, F1, IoU for each variant
- AUC delta from full model (how much each component contributes)

### Run Command
```bash
python -m eval.ablation
```

### Example Output
```
============================================================
Component          Accuracy       AUC        F1         IoU
------------------------------------------------------------
Full Model            0.9847     0.9912     0.9845     0.7832   (base)
No Frequency          0.9756     0.9754     0.9721     0.7214   (-0.0158)
No Noise              0.9823     0.9891     0.9818     0.7745   (-0.0021)
No CGAF               0.9645     0.9621     0.9534     0.6987   (-0.0291)
No Temporal           0.8234     0.8156     0.7892     0.5234   (-0.1756)
No F_high             0.9712     0.9654     0.9687     0.7123   (-0.0258)
============================================================
```

---

## ablation_cross_dataset.py

**Purpose:** Cross-dataset ablation study — tests component contribution across multiple datasets.

### Key Features
- Runs same 6 ablation variants as `ablation.py`
- Evaluates on user-selected datasets
- Creates multi-line chart showing AUC across variants for each dataset
- Reveals which components matter most for cross-dataset generalization

### Workflow
1. List available datasets
2. User selects datasets to test
3. For each dataset:
   - For each ablation variant:
     - Load model with variant config
     - Evaluate and record AUC
4. Save line chart with dataset curves overlaid

### Output
```
eval/ablation_cross_dataset_auc.png
```
Chart shows:
- X-axis: Ablation variants
- Y-axis: AUC (%)
- Multiple lines: One per dataset
- Reveals generalization patterns (e.g., which components matter for cross-dataset robustness)

### Run Command
```bash
python -m eval.ablation_cross_dataset
```

---

## psuedo_ablation.py

**Purpose:** Pseudo-mask generation weight ablation — tests different weighting strategies for pseudo-mask generation.

### Key Features
- Tests combinations of (alpha, beta, gamma) weights:
  - `alpha` - GradCAM weight (off by default on 4GB GPU, on by default on T4)
  - `beta` - Frequency branch weight
  - `gamma` - Noise branch weight

- Evaluates pseudo-mask quality via IoU with ground-truth masks
- Uses validation split (val_ratio=0.2, seed=42)

### Configuration
```python
# Enable GradCAM on T4 (disabled by default for 4GB safety)
use_gradcam = False  # Change to True on T4

# Test different weight combinations
alpha_values = [0.0, 0.2, 0.4]
beta_values = [0.3, 0.5, 0.7]
gamma_values = [0.2, 0.3, 0.4]
```

### Metrics
- **IoU** - Intersection-over-Union between generated and ground-truth masks
- Only computed on samples with ground-truth masks
- Higher IoU = better pseudo-mask quality

### Run Command
```bash
python -m eval.psuedo_ablation
```

### Use Case
Determines optimal weighting for training pipeline where ground-truth masks are unavailable (unsupervised learning scenario).

---

## robustness_eval_v3.py


### 8 Attack Types × 5 Severity Levels

| Attack | Name | Severity Levels |
|--------|------|-----------------|
| CS | Color Saturation | 0.4, 0.3, 0.2, 0.1, 0.0 |
| CC | Color Contrast | 0.85, 0.725, 0.6, 0.475, 0.35 |
| BW | Block-wise | 16, 32, 48, 64, 80 blocks |
| GN | Gaussian Noise | 0.001, 0.002, 0.005, 0.01, 0.05 |
| GB | Gaussian Blur | σ=3, 5, 7, 9, 13 |
| JPEG | JPEG Compression | 90, 70, 50, 30, 20 quality |
| RO | Rotate | 30°, 60°, 90°, 120°, 150° |
| AF | Affine Transform | scale=0.8, 0.9, 1.0, 1.1, 1.2 |

### Workflow
1. Load validation dataset
2. For each attack type:
   - For each severity level:
     - Apply attack to all frames
     - Evaluate model
     - Record AUC
3. Plot mean AUC per attack type
4. Compare with published results (SPSL, MAT, GocNet, HIFE, MH-FFNet)

### Output
```
eval/robustness/robustness_results.png
```
Chart shows:
- X-axis: Attack types
- Y-axis: AUC at highest severity
- Bars colored by model
- Includes paper benchmark results

### Published Benchmark
MH-FFNet results (Table 4):
```
CS:   98.42%    CC:   98.59%    BW:   94.69%    GN:   77.84%
GB:   97.66%    JPEG: 91.04%    RO:   84.10%    AF:   98.73%
```

### Run Command
```bash
python -m eval.robustness_eval_v3
```

### Interpretation
- **High robustness domains** (>95% AUC): Color changes, affine transforms, rotation
- **Vulnerable domains** (<80% AUC): Gaussian noise, block-wise corruption
- Useful for identifying weak points in deployment scenarios

---

## Summary Table

| File | Purpose | Input | Output |
|------|---------|-------|--------|
| evaluate.py | Core metrics | Model + DataLoader | 6 metrics + 8 charts |
| run_eval_v3.py | Full evaluation | Model checkpoint | Console report + charts |
| cross_dataset.py | Multi-dataset eval | Model + datasets dict | Results dict |
| cross_dataset_eval.py | Interactive cross-eval | Model + user input | Eval reports per dataset |
| ablation.py | Component ablation | Model + validation set | 6 variants + AUC comparison |
| ablation_cross_dataset.py | Cross-dataset ablation | Model + datasets | Line chart across datasets |
| psuedo_ablation.py | Weight ablation | Model + validation set | IoU per weight combo |
| robustness_eval_v3.py | Adversarial robustness | Model + attacks | Attack resilience chart |

---

## Common Configuration

```python
ROOT = "FaceForensics_Data"                    # Dataset root directory
CACHE_DIR = "dataset/cache/ffpp_processed_v2" # Processed frame cache
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

## Output Directory Structure

```
eval/
├── ablation/
│   ├── full_model/
│   ├── no_frequency/
│   ├── no_noise/
│   ├── no_cgaf/
│   ├── no_temporal/
│   ├── no_f_high/
│   ├── ablation_auc_summary.png
│   └── ablation_auc.png
├── best_model/
│   ├── overall_metrics.png
│   ├── confusion_matrix.png
│   ├── domain_acc.png
│   ├── domain_auc.png
│   ├── domain_table.png
│   ├── domain_radar.png
│   ├── per_domain_roc.png
│   └── roc_curve.png
└── robustness/
    └── robustness_results.png
```
