# AFAGNetV3: Advanced Face Forgery Detection Network

## Quick Start
For a quick setup guide and essential commands, see **[QUICK_START.md](QUICK_START.md)**

For interactive project management with automatic setup and dataset downloads, use:
```bash
python launcher.py  # Cross-platform launcher
# or
run_project.bat    # Windows batch file
```

## Evaluation Framework

### Core Metrics (6 Metrics)
- **Accuracy**: Binary classification accuracy
- **Precision**: True positive rate / predicted positive rate
- **Recall**: True positive rate / actual positive rate
- **F1 Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve
- **IoU**: Intersection over union for localization

### Per-Domain Evaluation
- **Real Domains**: youtube, actors
- **Fake Domains**: Deepfakes, Face2Face, FaceSwap, FaceShifter, NeuralTextures, NeuralTextures
- **Cross-Domain AUC**: Each fake method evaluated against all real samples

### Evaluation Scripts

#### run_eval_v3.py (Complete Evaluation)
```bash
python -m eval.run_eval_v3
```
- Loads best/EMA/latest checkpoint
- Computes all metrics + per-domain breakdown
- Generates 8 publication-quality charts
- Prints comparison with state-of-the-art

#### cross_dataset_eval.py (Interactive Cross-Dataset)
```bash
python -m eval.cross_dataset_eval
```
- User selects datasets: FFPP, Celeb-DF, Celeb-DF-V2, DFDC, FDR
- Evaluates model generalization across datasets
- Saves visuals per dataset

#### ablation.py (Component Ablation)
```bash
python -m eval.ablation
```
- Tests 6 variants: Full, No Frequency, No Noise, No CGAF, No Temporal, No F_high
- Measures contribution of each component
- Saves ablation comparison charts

#### robustness_eval_v3.py (Adversarial Robustness)
```bash
python -m eval.robustness_eval_v3
```
- 8 attacks × 5 severity levels
- Matches MH-FFNet paper protocol
- Generates robustness charts

### Visualization Outputs
1. **overall_metrics.png** - Bar chart of 6 metrics
2. **domain_acc.png** - Accuracy per manipulation method
3. **domain_auc.png** - AUC per manipulation method
4. **domain_table.png** - Complete metrics table
5. **confusion_matrix.png** - Binary confusion matrix
6. **roc_curve.png** - Overall ROC curve
7. **domain_radar.png** - Multi-metric radar chart
8. **per_domain_roc.png** - Per-domain ROC curves

---

## Visualization Tools

### Core Visualization (visualize_v3.py)
- **Input**: Model outputs (pred_cls, pred_mask, confidence, stability)
- **Output**: Publication-quality grids with color legends
- **Colormaps**:
  - JET: Forgery masks (red=high probability)
  - VIRIDIS: Confidence (yellow=high certainty)
  - COOL: Stability (magenta=temporal inconsistency)

### Pipeline Visualization (pipeline_visualize.py)
- **15-Step Process**: From raw input to final prediction
- **Feature Maps**: Backbone outputs, branch activations, fusion similarities
- **Interpretation**: Shows what each component detects

### Test Visualization (test_visualize_v3.py)
- **Quick Demo**: 4-sample balanced test (2 real, 2 fake)
- **Output**: `outputs/test_viz/` directory

### Usage Examples
```bash
# Quick visualization test
python -m utils.test_visualize_v3

# Full pipeline analysis
python -m utils.pipeline_visualize --video_idx 42

# Custom visualization
from utils.visualize_v3 import visualize_batch
visualize_batch(batch_cpu, output_cpu, save_path="outputs/custom")
```



## Installation and Setup

###  Quick Setup with Project Manager
For the easiest setup experience, use the interactive project manager:

```bash
# Windows
run_project.bat

# Or directly
python launcher.py
```

The project manager will:
-  Check/create virtual environment
-  Install all dependencies
-  Verify project status
-  Provide interactive dataset downloads
-  Show help for all commands

### Manual Setup (Alternative)
```bash
# Create virtual environment
python -m venv btech
btech\Scripts\activate  # Windows

# Install dependencies
pip install -r btech/requirements.txt

# Download FF++ dataset
# Place in FaceForensics_Data/ directory
```

### Project Structure Setup
```
c:\Users\mayur\Desktop\dummmmmmmmy\
├── launcher.py            # Main entry point
├── project.py   
├── run_project.bat        # Windows launcher
├── PROJECT_README.md # Manager documentation
├── btech\                 # Virtual environment
├── FaceForensics_Data\    # Dataset root
├── checkpoints\           # Model weights
├── dataset\cache\         # Processed samples cache
├── outputs\               # Visualization outputs
└── eval\                  # Evaluation results
```

---

## Usage Guide

###  Quick Start with Project Manager
The easiest way to use AFAGNetV3 is through the interactive project manager:

```bash
# Launch interactive manager
python launcher.py

# Or use Windows batch file
run_project.bat
```

**Project Manager Features:**
-  Automatic environment setup
-  Interactive dataset downloads
-  Command help for all scripts
-  Project status monitoring
-  Custom command execution

### Manual Usage

#### Training
```bash
# Train AFAGNetV3
python train/run_train_v3.py

# Train with specific config
python train/run_train_v3.py --config config/train_v3.yaml
```

#### Evaluation
```bash
# Complete evaluation with charts
python -m eval.run_eval_v3

# Cross-dataset evaluation
python -m eval.cross_dataset_eval

# Ablation study
python -m eval.ablation

# Robustness testing
python -m eval.robustness_eval_v3
```

### Visualization
```bash
# Quick test visualization
python -m utils.test_visualize_v3

# Pipeline analysis for video #42
python -m utils.pipeline_visualize --video_idx 42

# Custom visualization
python -c "
from utils.visualize_v3 import visualize_batch
# ... load model, get batch, visualize
"
```

### Model Loading
```python
from models.afag_net_v3 import AFAGNetV3

model = AFAGNetV3()
model.load_state_dict(torch.load("checkpoints/best_model.pth"))
model.eval()
```

---

## File Structure

```
dummmmmmmmy/
├── models/                    # Model architectures
│   ├── afag_net_v3.py        # Main model class
│   ├── backbone.py           # EfficientNet backbone
│   ├── branches/             # Feature extraction branches
│   │   ├── spatial_branch.py
│   │   ├── frequency_branch_raw.py
│   │   └── noise_branch_raw.py
│   ├── fusion/
│   │   └── cgaf.py          # Cross-branch fusion
│   ├── temporal/
│   │   └── temporal_model.py # Transformer temporal modeling
│   └── heads/                # Output heads
│       ├── classification_head.py
│       └── localization_head.py
├── train/                     # Training scripts
│   ├── run_train_v3.py       # Main training script
│   ├── train_pipeline_v3.py  # Training pipeline
│   └── psuedo_mask.py        # Pseudo-mask generation
├── eval/                      # Evaluation framework
│   ├── evaluate.py           # Core evaluation functions
│   ├── run_eval_v3.py        # Complete evaluation
│   ├── cross_dataset.py      # Cross-dataset runner
│   ├── cross_dataset_eval.py # Interactive cross-dataset
│   ├── ablation.py           # Component ablation
│   ├── ablation_cross_dataset.py # Cross-dataset ablation
│   ├── psuedo_ablation.py    # Pseudo-mask ablation
│   ├── robustness_eval_v3.py # Adversarial robustness
│   └── EVAL_FILES_DOCUMENTATION.md
├── utils/                     # Visualization utilities
│   ├── visualize_v3.py       # Core visualization
│   ├── test_visualize_v3.py  # Quick test
│   ├── pipeline_visualize.py # Full pipeline viz
│   └── UTILS_FILES_DOCUMENTATION.md
├── data/                      # Data loading and processing
│   ├── loader/
│   │   ├── build_dataset.py  # Dataset construction
│   │   ├── ffpp_dataset_v2.py # FFPP dataset class
│   │   ├── ffpp_dataset.py   # Legacy dataset
│   │   └── splits.py         # Train/val splitting
│   └── augmentation.py       # Data augmentation
├── checkpoints/               # Model checkpoints
│   ├── best_model.pth        # Best validation AUC
│   ├── ema_model.pth         # EMA model
│   └── latest_model.pth      # Latest epoch
├── dataset/                   # Processed data cache
│   └── cache/
├── outputs/                   # Visualization outputs
├── Explanation/               # Component explanations
├── FaceForensics_Data/        # Raw dataset
├── requirements.txt           # Python dependencies
├── FF.py                      # Legacy script
└── PROJECT_DOCUMENTATION.md   # This file
```



## Troubleshooting

### Common Issues

#### Out of Memory Errors
```python
# Reduce batch size
loader = DataLoader(dataset, batch_size=2)

# Use gradient checkpointing
model = AFAGNetV3(use_gradient_checkpointing=True)

# Clear cache between evaluations
torch.cuda.empty_cache()
```

#### Slow Training/Inference
```python
# Disable face alignment for faster processing
dataset = FFPPDatasetV2(use_alignment=False)

# Use CPU for visualization
model.to('cpu')
```
