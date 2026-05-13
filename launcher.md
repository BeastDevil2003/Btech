# AFAGNetV3 Project 

## Quick Start

### First Time Setup
```bash
# Run the launcher
python launcher.py
```

The launcher will automatically:
1. Check for virtual environment
2. Create venv if needed
3. Install dependencies
4. Show the main menu

### Non-Interactive Setup
```bash
# Setup only (no interactive menu)
python launcher.py --setup-only

# Check project status
python launcher.py --non-interactive
```

## Main Features

### 1. Environment Management
-  Automatic virtual environment detection and creation
-  Dependency installation from `btech/requirements.txt`
-  Environment health checking

### 2. Dataset Downloads
Interactive download manager for:
- **FaceForensics++**: Via custom FF.py script with multiple options (raw, c23, c40, videos, masks, models)
- **Celeb-DF v1/v2**: Google Drive downloads
- **DeeperForensics**: Direct download support

### 3. Command Help System
Detailed help for all scripts:

#### Evaluation Scripts (`eval/`)
- `evaluate.py` - Core evaluation metrics
- `run_eval_v3.py` - Complete evaluation pipeline
- `cross_dataset_eval.py` - Cross-dataset evaluation
- `ablation.py` - Ablation studies
- `robustness_eval_v3.py` - Robustness testing

#### Utilities Scripts (`utils/`)
- `visualize_v3.py` - Visualization functions
- `pipeline_visualize.py` - Training pipeline plots
- `test_visualize_v3.py` - Visualization testing

### 4. Custom Command Execution
Run any command in the virtual environment with proper Python/pip path resolution.

##  Usage Examples

### Interactive Mode
```bash
python launcher.py
```
Shows menu with options:
1. Setup/Verify Project Environment
2. Download Datasets
3. Evaluation Scripts Help
4. Utilities Scripts Help
5. Show Project Status
6. Run Custom Command

### Dataset Downloads
- Select dataset from interactive menu
- Choose specific download options for FaceForensics++
- Automatic Google Drive downloads for other datasets

### Getting Help
- Browse available scripts by category
- See detailed parameter descriptions
- View usage examples and code snippets

##  Project Structure

```
project-root/
├── launcher.py              # Main entry point
├── project.py       
├── link.txt                 # Dataset download links
├── FF.py                    # FaceForensics++ downloader
├── btech/                   # Virtual environment
│   └── requirements.txt     # Python dependencies
├── eval/                    # Evaluation scripts
├── utils/                   # Utility scripts
├── models/                  # Model definitions
├── data/                    # Data loading code
└── checkpoints/             # Model checkpoints
```

## Advanced Usage

### Custom Commands
```bash
# In interactive mode, select option 6
python launcher.py

# Then enter commands like:
python eval/run_eval_v3.py
pip install additional-package
python -c "import torch; print(torch.cuda.is_available())"
```

### Direct Script Access
```bash
# Access project manager directly
python project_manager.py --setup-only

# Use individual components
python FF.py FaceForensics_Data -d all -c raw -t videos
```
