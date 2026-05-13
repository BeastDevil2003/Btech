# Utilities Module Documentation

Complete guide to visualization and pipeline utilities for AFAGNetV3.

---

## 📋 Table of Contents

1. [visualize_v3.py](#visualize_v3py) - Core visualization functions
2. [test_visualize_v3.py](#test_visualize_v3py) - Quick visualization test
3. [pipeline_visualize.py](#pipeline_visualizepy) - Full pipeline feature map visualization

---

## visualize_v3.py

**Purpose:** Core visualization module for AFAGNetV3 feature maps, predictions, and analysis outputs.

### Key Features
- **Tensor-to-Image Conversion:**
  - Handles 3-channel (RGB) and 6-channel (RGB+YCbCr) inputs
  - Converts [-1, 1] normalized tensors to [0, 255] uint8 images
  - Maintains color accuracy through proper channel ordering

- **Heatmap Generation:**
  - Forgery mask heatmaps (JET colormap)
  - Confidence maps (VIRIDIS colormap)
  - Stability maps (COOL cyan→magenta colormap)

- **Overlay Compositing:**
  - Forgery mask overlay on face image
  - Confidence map overlay
  - Temporal stability patterns overlay
  - Configurable alpha blending for transparency

- **Grid Visualization:**
  - Multi-sample batch visualization
  - Side-by-side comparison layout
  - Color-coded legend strips explaining each visualization

### Color Meanings

#### Forgery Mask Overlay (JET Colormap)
```
Dark Blue    → Low forgery probability (real region)
Cyan/Green   → Moderate suspicion
Yellow/Orange→ High forgery probability
Red          → Very high forgery probability (most suspicious)
```

#### Confidence Map (VIRIDIS)
```
Dark Purple  → Low confidence (uncertain)
Bright Yellow→ High confidence (very certain)
```

#### Temporal Stability (COOL)
```
Cyan         → Stable temporal patterns (consistent)
Magenta/Pink → Flickering patterns (temporal inconsistency = fake)
```

### Core Functions

#### `tensor_to_rgb(tensor) → np.ndarray`
Converts (C, H, W) tensor in [-1, 1] to (H, W, 3) uint8 RGB image.

**Parameters:**
- `tensor` - Torch tensor with shape (C, H, W) where C=3 or C=6
- Returns: (H, W, 3) uint8 uint8 numpy array

#### `mask_to_heatmap(mask, colormap=cv2.COLORMAP_JET) → np.ndarray`
Converts (1, H, W) or (H, W) mask in [0, 1] to (H, W, 3) BGR heatmap.

**Parameters:**
- `mask` - Forgery mask tensor
- `colormap` - OpenCV colormap constant (default JET)
- Returns: (H, W, 3) BGR heatmap

#### `bgr_to_rgb(img) → np.ndarray`
Converts BGR image to RGB by reversing channel order.

#### `make_overlay(face_rgb, mask_tensor, alpha=0.45) → np.ndarray`
Overlays forgery mask heatmap on face image.

**Parameters:**
- `face_rgb` - (H, W, 3) uint8 RGB face image
- `mask_tensor` - (1, H, W) or (H, W) float mask in [0, 1]
- `alpha` - Overlay transparency [0, 1], default 0.45
- Returns: (H, W, 3) uint8 RGB overlay image

#### `visualize_batch(batch_cpu, output_cpu, save_path="outputs/viz")`
Creates and saves publication-quality visualization grid for a batch.

**Parameters:**
- `batch_cpu` - Batch dict (moved to CPU) with keys:
  - `"frames"` - Video frames (B, N, C, H, W)
  - `"label"` - Binary labels (B,)
  - `"domain"` - Manipulation type list
- `output_cpu` - Model output dict with:
  - `"pred_cls"` - Classification logits (B, 1)
  - `"pred_mask"` - Forgery mask (B, 1, 224, 224)
  - `"mask_sequence"` - Temporal mask sequence (B, N, 1, 224, 224)
  - `"confidence"` - Confidence map (B, 1, 224, 224)
  - `"stability"` - Temporal stability (B, N-1, 1, 224, 224)
- `save_path` - Directory to save visualizations

#### `visualize_sample(batch_cpu, output_cpu, sample_idx=0, save_path="outputs/viz")`
Visualizes a single sample from a batch.

**Parameters:**
- `batch_cpu` - Single sample dict
- `output_cpu` - Model output for that sample
- `sample_idx` - Index in batch to visualize (default 0)
- `save_path` - Output directory

### Output Layout

Grid contains rows (from top to bottom):
1. **Input Frame** - Middle frame of 16-frame clip (raw RGB)
2. **Forgery Mask** - Model's localization output with jet colormap
3. **Confidence Map** - Pixel-level certainty (yellow=certain)
4. **Temporal Stability** - Frame-to-frame consistency (magenta=unstable/fake)
5. **Legend Strip** - Color gradients + labels explaining rows 2-4


### Usage Example
```python
from utils.visualize_v3 import visualize_batch

# After forward pass
batch_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
             for k, v in batch.items()}
output_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
              for k, v in output.items()}

visualize_batch(batch_cpu, output_cpu, 
                save_path="outputs/sample_viz")
```

### Output Files
- `viz_sample_0_LABEL_DOMAIN.png` - Single sample visualization grid
- `viz_batch_DATETIME.png` - Full batch comparison grid

---

## test_visualize_v3.py

**Purpose:** Quick test script to verify visualization pipeline on a small balanced sample.

### What It Does
1. Loads 2 real + 2 fake videos from dataset
2. Runs forward pass through AFAGNetV3
3. Generates visualization grid
4. Saves to `outputs/test_viz/`

### Key Features
- **Balanced Sample:** Ensures both real and fake examples
- **Quick Execution:** Uses only 4 samples (faster testing)
- **Checkpoint Health Check:** Verifies no NaN/Inf in weights
- **Auto-detection:** Finds best/EMA checkpoint automatically

### Configuration
```python
ROOT = "FaceForensics_Data"
CACHE_DIR = "dataset/cache/ffpp_processed_v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

### Workflow
1. Build FFPPDatasetV2 from FaceForensics
2. Sample 2 real videos (labels=0)
3. Sample 2 fake videos (labels=1)
4. Load best checkpoint (health-checked)
5. Run forward pass on 4-sample batch
6. Generate visualization grid
7. Save to `outputs/test_viz/`

### Run Command
```bash
python -m utils.test_visualize_v3
```

### Output Example
```
Loading checkpoint: checkpoints/best_model.pth
Done. Check outputs/test_viz/
```

Files created:
- `outputs/test_viz/viz_batch_TIMESTAMP.png` - 4-sample grid

### Use Cases
- **Quick validation** after model training
- **Visual sanity check** on predictions
- **Demo visualization** of model capabilities
- **Debugging** visualization pipeline

---

## pipeline_visualize.py

**Purpose:** Comprehensive pipeline feature map visualization showing every processing step from input to output.

### What It Shows (15 Visualizations)

| Step | Component | Description |
|------|-----------|-------------|
| 0 | Input Frame | Raw RGB face, middle frame of 16-frame clip |
| 1a | Backbone F_low | Local texture features (14×14, upsampled to 224) |
| 1b | Backbone F_high | Semantic context features (7×7, upsampled to 224) |
| 2a | Spatial Branch | Dual-path spatial features with CBAM attention |
| 2b | Frequency Branch | FFT magnitude + phase in frequency domain |
| 2c | Noise Branch | SRM-filtered noise residual features |
| 3a | CGAF Similarity Csf | Spatial–frequency cross-branch agreement |
| 3b | CGAF Similarity Csn | Spatial–noise cross-branch agreement |
| 3c | Fused Features | Ffusion — gated adaptive fusion output |
| 4a | Forgery Mask | Localization head output on face |
| 4b | Confidence Map | Pixel-level certainty |
| 4c | Temporal Stability | Frame-to-frame consistency metric |
| 4d | Classification Result | Binary prediction + probability |
| - | Legend Strips | Color gradient explanations |

### Feature Map Visualization Strategy

#### Dense Feature Layers → RGB Heatmap
For multi-channel feature maps (C, H, W):
1. Compute L2 norm across channels → energy map (H, W)
2. Normalize to [0, 1]
3. Resize to target (224, 224)
4. Apply VIRIDIS colormap
5. Result: "Where is this branch most active?"

#### Similarity Maps
- (1, H, W) in [-1, 1]
- Map [-1, 1] → [0, 1] linearly
- Apply diverging colormap:
  - Red = high agreement (+1)
  - Blue = disagreement (-1)

### Key Classes & Functions

#### `tensor_to_rgb(t) → np.ndarray`
Converts (C, H, W) in [-1, 1] to (H, W, 3) uint8 RGB. Handles 3/6 channel inputs.

#### `feature_map_to_rgb(feat, target_hw=(224,224), colormap=VIRIDIS) → np.ndarray`
Collapses multi-channel feature map to single heatmap.

**Strategy:** L2 norm across channels captures total activation energy.

#### `similarity_map_to_rgb(sim, target_hw=(224,224)) → np.ndarray`
Converts [-1, 1] similarity map to RGB heatmap.

**Color meaning:**
- Red (similarity=+1) = high cross-branch agreement
- Blue (similarity=-1) = disagreement
- Purple = neutral

### Advanced Features
- **Batch Processing:** Visualizes all samples in batch
- **Spatial Alignment:** All visualizations resized to 224×224
- **Legend Strips:** Each row has color gradient + labels
- **Domain Annotation:** Shows manipulation type (DF, F2F, etc.)
- **Confidence Scores:** Displays model's probability for each sample

### Key Components Explained

#### Backbone Features
- **F_low** : Spatial textures, local patterns
  - Captures fine-grained noise and manipulation artifacts
  - Low semantic level, high spatial specificity
- **F_high**: Semantic context, face structure
  - Captures global face geometry
  - High semantic level, low spatial detail

#### Branch Outputs
- **Spatial (Fs):** Combines F_low + F_high with CBAM attention
  - Focuses on spatial patterns of manipulation
- **Frequency (Ff):** FFT magnitude + phase information
  - Detects unnatural frequency patterns from manipulation
- **Noise (Fn):** SRM residual filtering
  - Isolates camera sensor noise patterns

#### CGAF Fusion
- **Csf:** Cross-agreement between spatial and frequency
  - Where do Spatial and Frequency branches agree?
  - High Csf = strong spatial-frequency evidence
- **Csn:** Cross-agreement between spatial and noise
  - Where do Spatial and Noise branches agree?
  - High Csn = strong artifact evidence
- **Ffusion:** Gated adaptive combination of Fs, Ff, Fn
  - Dynamically weights branches based on confidence

#### Output Layer
- **Forgery Mask:** 224×224 localization map
  - Pixel-level forgery probability
  - Can identify which facial regions are manipulated
- **Confidence Map:** Certainty of forgery mask
  - Yellow = very certain
  - Dark = uncertain
- **Temporal Stability:** Frame-to-frame consistency
  - Real faces have stable masks
  - Fake faces show flickering/inconsistency

### Usage

#### Command Line
```bash
python -m utils.pipeline_visualize                    # Default: first video
python -m utils.pipeline_visualize --video_idx 42     # Video 42
python -m utils.pipeline_visualize --save_dir outputs/pipeline_analysis
```

#### Programmatic
```python
from utils.pipeline_visualize import visualize_full_pipeline

visualize_full_pipeline(
    model,
    dataset,
    video_idx=0,
    save_dir="outputs/pipeline",
    device="cuda"
)
```

### Configuration
```python
ROOT = "FaceForensics_Data"
CACHE_DIR = "dataset/cache/ffpp_processed_v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

### Run Command
```bash
python -m utils.pipeline_visualize
```

### Output Structure
```
outputs/pipeline/
├── pipeline_full_sample_0_REAL_youtube.png
├── pipeline_full_sample_1_FAKE_Deepfakes.png
└── ...
```

Each file contains 15-panel grid showing complete processing pipeline.

### Use Cases
- **Understanding Model Behavior:** See what each component does
- **Debugging Predictions:** Why did it predict real when it's fake?
- **Paper Figures:** Publication-quality feature visualizations
- **Model Explanation:** Interpretability for stakeholders
- **Teaching:** Demonstrating deepfake detection pipeline
- **Failure Analysis:** Identify where model goes wrong

