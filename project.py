#!/usr/bin/env python3
import os
import sys
import subprocess
import json
import argparse
from pathlib import Path
import urllib.request
import shutil

PROJECT_ROOT = Path(__file__).parent.absolute()
VENV_DIR = PROJECT_ROOT / "btech"
REQUIREMENTS_FILE = PROJECT_ROOT / "btech" / "requirements.txt"
LINKS_FILE = PROJECT_ROOT / "link.txt"

DATASET_LINKS = {
    "faceforensics++": {
        "description": "FaceForensics++ dataset (videos, masks, models) - organized by compression",
        "script": "FF.py",
        "commands": [
            "python FF.py FaceForensics_Data -d all -c raw -t videos --server EU2   # Raw videos in raw/videos/",
            "python FF.py FaceForensics_Data -d all -c c23 -t videos --server EU2    # C23 videos in c23/videos/",
            "python FF.py FaceForensics_Data -d all -c c40 -t videos --server EU2  # C40 videos in c40/videos/",
            "python FF.py FaceForensics_Data -d all -c raw -t masks --server EU2  # Raw masks in raw/masks/",
            "python FF.py FaceForensics_Data -d all -c c23 -t masks --server EU2     # C23 masks in c23/masks/",
            "python FF.py FaceForensics_Data -d all -c c40 -t masks --server EU2     # C40 masks in c40/masks/",
        ]
    },
    "celeb-df-v2": {
        "description": "Celeb-DF v2 dataset (Google Drive)",
        "link": "https://drive.google.com/open?id=1iLx76wsbi9itnkxSqz9BVBl4ZvnbIazj",
        "download_cmd": "gdown https://drive.google.com/uc?id=1iLx76wsbi9itnkxSqz9BVBl4ZvnbIazj"
    },
    "celeb-df-v1": {
        "description": "Celeb-DF v1 dataset (Google Drive)",
        "link": "https://drive.google.com/open?id=10NGF38RgF8FZneKOuCOdRIsPzpC7_WDd",
        "download_cmd": "gdown https://drive.google.com/uc?id=10NGF38RgF8FZneKOuCOdRIsPzpC7_WDd"
    },
    "deeperforensics": {
        "description": "DeeperForensics-1.0 dataset (Google Drive)",
        "link": "https://drive.google.com/open?id=1s3KwYyTIXT78VzkRazn9QDPuNh18TWe-",
        "download_cmd": "gdown https://drive.google.com/uc?id=1s3KwYyTIXT78VzkRazn9QDPuNh18TWe-"
        "This can't download the dataset directly due to Google Drive restrictions. Please visit the link "
    },
    "WildDeepfake": {
        "description": "WildDeepfake dataset (Google Drive)",
        "link": "https://huggingface.co/datasets/xingjunm/WildDeepfake/tree/main/deepfake_in_the_wild"
    }
}

EVAL_COMMANDS = {
    "evaluate.py": {
        "description": "Core evaluation script for model performance metrics",
        "usage": "python -m eval.evaluate",
        "parameters": {
            "model_path": "Path to model checkpoint (.pth file)",
            "dataset_root": "Root directory of evaluation dataset",
            "output_dir": "Directory to save evaluation results and plots",
            "batch_size": "Batch size for evaluation (default: 8)",
            "device": "Device to run evaluation on (cuda/cpu)",
            "save_visuals": "Whether to save evaluation visualizations"
        },
        "examples": [
            "python -c \"from eval.evaluate import evaluate; evaluate('checkpoints/best_model.pth', 'FaceForensics_Data', 'eval_results')\"",
            "python -m eval.evaluate --model_path checkpoints/ema_model.pth --dataset_root FaceForensics_Data"
        ]
    },
    "run_eval_v3.py": {
        "description": "Complete evaluation pipeline for AFAGNetV3",
        "usage": "python eval/run_eval_v3.py",
        "parameters": {
            "checkpoint": "Specific checkpoint path (optional, auto-detects best/ema/latest)",
            "dataset_root": "Dataset root directory (default: FaceForensics_Data)",
            "cache_dir": "Cache directory for processed data",
            "output_dir": "Output directory for results",
            "batch_size": "Evaluation batch size",
            "num_workers": "Number of data loading workers"
        },
        "examples": [
            "python eval/run_eval_v3.py  # Auto-detect best model",
            "python eval/run_eval_v3.py --checkpoint checkpoints/custom_model.pth"
        ]
    },
    "cross_dataset_eval.py": {
        "description": "Cross-dataset evaluation on multiple datasets",
        "usage": "python eval/cross_dataset_eval.py",
        "parameters": {
            "model_path": "Path to trained model checkpoint",
            "datasets": "Comma-separated list of dataset names",
            "output_dir": "Directory to save cross-dataset results",
            "batch_size": "Batch size for evaluation"
        },
        "examples": [
            "python eval/cross_dataset_eval.py  # Interactive dataset selection",
            "python eval/cross_dataset_eval.py --datasets Celeb-DF-v2,DFDC"
        ]
    },
    "ablation.py": {
        "description": "Ablation study for model components",
        "usage": "python eval/ablation.py",
        "parameters": {
            "model_path": "Path to model checkpoint",
            "output_dir": "Directory to save ablation results",
            "components": "Comma-separated list of components to ablate",
            "metrics": "Metrics to evaluate (acc,auc,f1,precision,recall)"
        },
        "examples": [
            "python eval/ablation.py  # Full ablation study",
            "python eval/ablation.py --components spatial,frequency,noise"
        ]
    },
    "robustness_eval_v3.py": {
        "description": "Robustness evaluation against perturbations",
        "usage": "python eval/robustness_eval_v3.py",
        "parameters": {
            "model_path": "Path to model checkpoint",
            "perturbation_types": "Types of perturbations to test",
            "severity_levels": "Severity levels for perturbations",
            "output_dir": "Directory to save robustness results"
        },
        "examples": [
            "python eval/robustness_eval_v3.py",
            "python eval/robustness_eval_v3.py --perturbation_types blur,noise,compression"
        ]
    }
}

UTILS_COMMANDS = {
    "visualize_v3.py": {
        "description": "Visualization utilities for model predictions and masks",
        "usage": "python -c \"from utils.visualize_v3 import *\"",
        "functions": {
            "tensor_to_rgb": "Convert model output tensor to RGB image",
            "mask_to_heatmap": "Convert prediction mask to heatmap overlay",
            "make_overlay": "Create face image with prediction mask overlay",
            "plot_comparison": "Plot side-by-side comparison of predictions",
            "save_visualization_grid": "Save grid of visualization examples"
        },
        "examples": [
            "python -c \"from utils.visualize_v3 import make_overlay; overlay = make_overlay(face_img, pred_mask)\"",
            "python utils/test_visualize_v3.py  # Test visualization functions"
        ]
    },
    "pipeline_visualize.py": {
        "description": "Training pipeline visualization and monitoring",
        "usage": "python utils/pipeline_visualize.py",
        "parameters": {
            "log_dir": "Directory containing training logs",
            "output_dir": "Directory to save pipeline visualizations",
            "metrics": "Metrics to plot (loss,acc,auc,f1)"
        },
        "examples": [
            "python utils/pipeline_visualize.py --log_dir experiments/logs",
            "python utils/pipeline_visualize.py --metrics loss,acc,auc"
        ]
    },
    "test_visualize_v3.py": {
        "description": "Test script for visualization functions",
        "usage": "python utils/test_visualize_v3.py",
        "parameters": {
            "model_path": "Path to model for testing visualizations",
            "sample_data": "Path to sample data for testing",
            "output_dir": "Directory to save test visualizations"
        },
        "examples": [
            "python utils/test_visualize_v3.py",
            "python utils/test_visualize_v3.py --model_path checkpoints/best_model.pth"
        ]
    }
}

class ProjectManager:
    def __init__(self):
        self.venv_python = None
        self.venv_pip = None
        self.check_venv()

    def check_venv(self):
        if VENV_DIR.exists():
            python_exe = VENV_DIR / "Scripts" / "python.exe"
            pip_exe = VENV_DIR / "Scripts" / "pip.exe"

            if python_exe.exists() and pip_exe.exists():
                self.venv_python = str(python_exe)
                self.venv_pip = str(pip_exe)
                print(" Virtual environment found at:", VENV_DIR)
                return True

        print("Virtual environment not found or incomplete")
        return False

    def create_venv(self):
        print(f"Creating virtual environment at {VENV_DIR}")
        try:
            subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
            print("Virtual environment created successfully")
            self.check_venv()
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to create virtual environment: {e}")
            return False

    def install_requirements(self):
        if not self.venv_pip:
            print("Virtual environment not available")
            return False

        if not REQUIREMENTS_FILE.exists():
            print(f"Requirements file not found: {REQUIREMENTS_FILE}")
            return False

        print(" Installing requirements...")
        try:
            subprocess.run([self.venv_pip, "install", "-r", str(REQUIREMENTS_FILE)], check=True)
            print("Requirements installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install requirements: {e}")
            return False

    def setup_project(self):
        print("Setting up AFAGNetV3 project...")

        if not self.check_venv():
            if not self.create_venv():
                return False

        if not self.install_requirements():
            return False

        print("Project setup complete!")
        return True

    def show_dataset_menu(self):
        print("\n" + "="*60)
        print("DATASET DOWNLOAD MANAGER")
        print("="*60)

        for i, (key, info) in enumerate(DATASET_LINKS.items(), 1):
            print(f"{i}. {key.upper()}")
            print(f"   {info['description']}")
            if 'link' in info:
                print(f"   Link: {info['link']}")
            print()

        print("0. Back to main menu")
        print("-"*60)

        while True:
            try:
                choice = input(f"Select dataset to download (0-{len(DATASET_LINKS)}): ").strip()

                if choice == "0":
                    return

                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(DATASET_LINKS):
                    dataset_key = list(DATASET_LINKS.keys())[choice_idx]
                    self.download_dataset(dataset_key)
                    break
                else:
                    print("Invalid choice. Please select 0-4.")

            except ValueError:
                print("Please enter a valid number.")

    def download_dataset(self, dataset_key):
        if dataset_key not in DATASET_LINKS:
            print(f"Unknown dataset: {dataset_key}")
            return

        info = DATASET_LINKS[dataset_key]
        print(f"\nDownloading {dataset_key.upper()}")
        print(f"Description: {info['description']}")

        if 'script' in info:
            print("\nAvailable commands for FaceForensics++:")
            for i, cmd in enumerate(info['commands'], 1):
                print(f"{i}. {cmd}")

            cmd_choice = input("\nSelect command number or enter custom command: ").strip()

            if cmd_choice.isdigit() and 1 <= int(cmd_choice) <= len(info['commands']):
                command = info['commands'][int(cmd_choice) - 1]
            else:
                command = cmd_choice

            command = command.split('#')[0].strip()
            print(f"\nExecuting: {command}")
            try:
                subprocess.run(command, shell=True, check=True)
                print("Download completed!")
            except subprocess.CalledProcessError as e:
                print(f"Download failed: {e}")

        elif 'download_cmd' in info:
            command = info['download_cmd']
            command = command.split('#')[0].strip()
            print(f"\nExecuting: {command}")
            try:
                if command.startswith("gdown "):
                    url = command.split(" ", 1)[1]
                    cmd_parts = [self.venv_python, "-c", f"import gdown; gdown.download('{url}')"]
                    subprocess.run(cmd_parts, check=True)
                else:
                    subprocess.run(command, shell=True, check=True)
                print("Download completed!")
            except subprocess.CalledProcessError as e:
                print(f"Download failed: {e}")
        else:
            if 'link' in info:
                print(f"Link: {info['link']}")
                print("Please download this dataset manually from the link above.")
                print("You can use your browser or download tools like wget/curl.")

    def show_eval_menu(self):
        print("\n" + "="*60)
        print("EVALUATION SCRIPTS")
        print("="*60)

        for i, (script, info) in enumerate(EVAL_COMMANDS.items(), 1):
            print(f"{i}. {script}")
            print(f"   {info['description']}")
            print()

        print("0. Back to main menu")
        print("-"*60)

        while True:
            try:
                choice = input("Select script to learn about (0-5): ").strip()

                if choice == "0":
                    return

                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(EVAL_COMMANDS):
                    script_name = list(EVAL_COMMANDS.keys())[choice_idx]
                    self.show_command_help(script_name, EVAL_COMMANDS[script_name])
                    break
                else:
                    print("Invalid choice. Please select 0-5.")

            except ValueError:
                print("Please enter a valid number.")

    def show_utils_menu(self):
        print("\n" + "="*60)
        print("UTILITIES SCRIPTS")
        print("="*60)

        for i, (script, info) in enumerate(UTILS_COMMANDS.items(), 1):
            print(f"{i}. {script}")
            print(f"   {info['description']}")
            print()

        print("0. Back to main menu")
        print("-"*60)

        while True:
            try:
                choice = input("Select script to learn about (0-3): ").strip()

                if choice == "0":
                    return

                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(UTILS_COMMANDS):
                    script_name = list(UTILS_COMMANDS.keys())[choice_idx]
                    self.show_command_help(script_name, UTILS_COMMANDS[script_name])
                    break
                else:
                    print("Invalid choice. Please select 0-3.")

            except ValueError:
                print("Please enter a valid number.")

    def show_command_help(self, script_name, info):
        print(f"\nHELP: {script_name}")
        print("="*60)
        print(f"Description: {info['description']}")
        print(f"Usage: {info['usage']}")

        if 'parameters' in info:
            print("\nParameters:")
            for param, desc in info['parameters'].items():
                print(f"  --{param}: {desc}")

        if 'functions' in info:
            print("\nAvailable Functions:")
            for func, desc in info['functions'].items():
                print(f"  {func}(): {desc}")

        if 'examples' in info:
            print("\nExamples:")
            for example in info['examples']:
                print(f"  {example}")

        print("\n" + "-"*60)
        input("Press Enter to continue...")

    def show_main_menu(self):
        while True:
            print("\n" + "="*60)
            print("AFAGNetV3")
            print("="*60)
            print("1. Setup/Verify Project Environment")
            print("2. Download Datasets")
            print("3. Evaluation Scripts Help")
            print("4. Utilities Scripts Help")
            print("5. Show Project Status")
            print("6. Run Custom Command")
            print("0. Exit")
            print("-"*60)

            choice = input("Select option (0-6): ").strip()

            if choice == "0":
                print("Goodbye!")
                break
            elif choice == "1":
                self.setup_project()
            elif choice == "2":
                self.show_dataset_menu()
            elif choice == "3":
                self.show_eval_menu()
            elif choice == "4":
                self.show_utils_menu()
            elif choice == "5":
                self.show_project_status()
            elif choice == "6":
                self.run_custom_command()
            else:
                print("Invalid choice. Please select 0-6.")

    def show_project_status(self):
        print("\n" + "="*60)
        print("PROJECT STATUS")
        print("="*60)


        venv_status = "Active" if self.check_venv() else "Not set up"
        print(f"Virtual Environment: {venv_status}")

        if self.venv_pip:
            try:
                result = subprocess.run([self.venv_pip, "list"], capture_output=True, text=True)
                installed_packages = len(result.stdout.strip().split('\n')) - 2  # Subtract header lines
                print(f"Installed Packages: {installed_packages} packages")
            except:
                print("Installed Packages: Unable to check")
        else:
            print("Installed Packages: N/A")

        datasets_found = []
        if (PROJECT_ROOT / "FaceForensics_Data").exists():
            datasets_found.append("FaceForensics++")
        if (PROJECT_ROOT / "Celeb-DF-v2").exists():
            datasets_found.append("Celeb-DF-v2")
        if (PROJECT_ROOT / "DFDC").exists():
            datasets_found.append("DFDC")

        print(f"Datasets Found: {', '.join(datasets_found) if datasets_found else 'None'}")

        checkpoints_dir = PROJECT_ROOT / "checkpoints"
        if checkpoints_dir.exists():
            models = list(checkpoints_dir.glob("*.pth"))
            print(f"Model Checkpoints: {len(models)} found")
        else:
            print("Model Checkpoints: None")

        print("-"*60)

    def run_custom_command(self):
        print("\n" + "="*60)
        print("CUSTOM COMMAND EXECUTOR")
        print("="*60)
        print("Enter any command to run in the virtual environment.")
        print("Examples:")
        print("  python eval/run_eval_v3.py")
        print("  python -c \"import torch; print(torch.cuda.is_available())\"")
        print("  pip install additional-package")
        print()

        if not self.venv_python:
            print("Virtual environment not available. Please run setup first.")
            return

        command = input("Command: ").strip()
        if not command:
            return

        command = command.split('#')[0].strip()
        print(f"\nExecuting: {command}")
        print("-"*60)

        try:
            if command.startswith("python "):
                cmd_parts = [self.venv_python] + command.split()[1:]
            elif command.startswith("pip "):
                cmd_parts = [self.venv_pip] + command.split()[1:]
            else:
                cmd_parts = command.split()

            subprocess.run(cmd_parts, check=True)
            print("Command executed successfully!")

        except subprocess.CalledProcessError as e:
            print(f"Command failed with exit code {e.returncode}")
        except FileNotFoundError:
            print("Command not found. Check if it's installed.")

        print("-"*60)
        input("Press Enter to continue...")


def main():
    print("AFAGNetV3")
    print("===========================")

    parser = argparse.ArgumentParser(description="AFAGNetV3 Project Manager")
    parser.add_argument("--setup-only", action="store_true",
                       help="Only perform setup and exit")
    parser.add_argument("--non-interactive", action="store_true",
                       help="Run in non-interactive mode")

    args = parser.parse_args()

    manager = ProjectManager()

    if args.setup_only:
        success = manager.setup_project()
        sys.exit(0 if success else 1)

    if args.non_interactive:
        manager.show_project_status()
        return

    manager.show_main_menu()


if __name__ == "__main__":
    main()