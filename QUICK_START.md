# AFAGNetV3 Quick Start Guide

## Virtual Environment Setup

### If venv is not set up:

1. **Create virtual environment:**
   ```bash
   python -m venv btech
   ```

2. **Activate virtual environment:**
   - **Windows:** `.\btech\Scripts\activate`
   - **Linux/Mac:** `source btech/bin/activate`

3. **Install requirements:**
   ```bash
   pip install -r btech/requirements.txt
   ```

### If venv is already set up:

1. **Activate virtual environment:**
   - **Windows:** `.\btech\Scripts\activate`
   - **Linux/Mac:** `source btech/bin/activate`

2. **Run any file, for example:**
   ```bash
   python -m train.run_train_v3
   ```

## Important Commands

### Training Model
```bash
python -m train.run_train_v3
```

### Evaluating Model
```bash
# Cross-dataset evaluation
python -m eval.cross_dataset_eval

# Ablation study
python -m eval.ablation

# Cross-dataset ablation
python -m eval.ablation_cross_dataset

# Complete evaluation pipeline
python -m eval.run_eval_v3

# Robustness evaluation
python -m eval.robustness_eval_v3

# Pseudo ablation
python -m eval.psuedo_ablation
```

### Visualizing Results
```bash
# Test visualization
python -m utils.test_visualize_v3

# Pipeline visualization
python -m utils.pipeline_visualize
```

## Quick Project  Access

For an interactive  with automatic setup, dataset downloads, and command help:

```bash
# Windows
run_project.bat

# Cross-platform
python launcher.py
```

## Dataset Structure

After downloading, your dataset will be organized as follows:

```
FaceForensics_Data/
├── manipulated_sequences/
│   ├── Deepfakes/
│   │   ├── c23/           # Compression level c23
│   │   │   ├── videos/    # Video files
│   │   │   └── masks/     # Forgery mask files
│   │   └── c40/           # Compression level c40
│   │       ├── videos/    # Video files
│   │       └── masks/     # Forgery mask files
│   └── [other_datasets...]/
└── original_sequences/
    ├── youtube/
    │   ├── c23/
    │   │   └── videos/    # Only videos for original sequences
    │   └── c40/
    │       └── videos/
    └── actors/
        ├── c23/
        │   └── videos/
        └── c40/
            └── videos/
```


## To Download FF++ dataset


**Download C23 videos**
   ```bash
   python FF.py FaceForensics_Data -d all -c c23 -t videos --server EU2
   ```

**Download C23 masks**
   ```bash
   python FF.py FaceForensics_Data -d all -c c23 -t masks --server EU2
   ```


**Download C40 videos**
   ```bash
   python FF.py FaceForensics_Data -d all -c c40 -t videos --server EU2
   ```

**Download C40 masks**
   ```bash
   python FF.py FaceForensics_Data -d all -c c23 -t masks --server EU2
   ```

**Download raw videos**
   ```bash
   python FF.py FaceForensics_Data -d all -c raw -t videos --server EU2
   ```

**Download raw masks**
   ```bash
   python FF.py FaceForensics_Data -d all -c raw -t masks --server EU2
   ```

### Note: some time server not work to try with different server 
e.g  .... --server EU or --server CA
