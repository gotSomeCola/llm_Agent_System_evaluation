#!/usr/bin/env python3
"""
Centralized configuration for directory structure and paths.
All scripts should use these constants instead of hardcoding paths.
"""

import os
from pathlib import Path

# Base directories
PROJECT_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = PROJECT_DIR / "results"
PROJECT_TEMP_DIR = PROJECT_DIR / "project"
DATA_DIR = PROJECT_DIR

# Dataset path
LEETCODE_DATASET_PATH = DATA_DIR / "leetcode_dataset.json"

# Results subdirectories - Evaluations (raw data)
EVAL_RESULTS_DIR = RESULTS_DIR / "evaluations"
BASELINE_EVAL_DIR = EVAL_RESULTS_DIR / "baseline"
REPAIRS_EVAL_DIR = EVAL_RESULTS_DIR / "repairs"
PASS_AT_K_EVAL_DIR = EVAL_RESULTS_DIR / "pass_at_k"

# Results subdirectories - Reports (formatted output)
REPORTS_DIR = RESULTS_DIR / "reports"
BASELINE_REPORTS_DIR = REPORTS_DIR / "baseline"
REPAIRS_REPORTS_DIR = REPORTS_DIR / "repairs"
PASS_AT_K_REPORTS_DIR = REPORTS_DIR / "pass_at_k"

# Backward compatibility mappings (for existing code)
# These can be updated in scripts gradually
CODER_TESTER_RESULTS = REPAIRS_EVAL_DIR
BASELINE_AT_K_RESULTS = PASS_AT_K_EVAL_DIR
BASELINE_RESULTS = BASELINE_REPORTS_DIR


def ensure_directories():
    """Create all necessary directories."""
    directories = [
        # Evaluation results
        BASELINE_EVAL_DIR,
        REPAIRS_EVAL_DIR,
        PASS_AT_K_EVAL_DIR,
        # Reports
        BASELINE_REPORTS_DIR,
        REPAIRS_REPORTS_DIR,
        PASS_AT_K_REPORTS_DIR,
        # Temporary
        PROJECT_TEMP_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory ready: {directory}")


def cleanup_temp_projects():
    """Clean up temporary project directories."""
    if PROJECT_TEMP_DIR.exists():
        import shutil
        try:
            shutil.rmtree(PROJECT_TEMP_DIR)
            PROJECT_TEMP_DIR.mkdir(parents=True, exist_ok=True)
            print(f"✓ Cleaned up temporary projects: {PROJECT_TEMP_DIR}")
        except Exception as e:
            print(f"Warning: Failed to cleanup {PROJECT_TEMP_DIR}: {e}")


def print_structure():
    """Print the current directory structure info."""
    print("\n" + "="*60)
    print("PROJECT STRUCTURE CONFIGURATION")
    print("="*60)
    print(f"Project Directory: {PROJECT_DIR}")
    print(f"\nRESULTS:")
    print(f"  Base: {RESULTS_DIR}")
    print(f"  Evaluations: {EVAL_RESULTS_DIR}")
    print(f"    - Baseline: {BASELINE_EVAL_DIR}")
    print(f"    - Repairs: {REPAIRS_EVAL_DIR}")
    print(f"    - Pass@K: {PASS_AT_K_EVAL_DIR}")
    print(f"  Reports: {REPORTS_DIR}")
    print(f"    - Baseline: {BASELINE_REPORTS_DIR}")
    print(f"    - Repairs: {REPAIRS_REPORTS_DIR}")
    print(f"    - Pass@K: {PASS_AT_K_REPORTS_DIR}")
    print(f"\nTEMPORARY:")
    print(f"  Project Temp: {PROJECT_TEMP_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    ensure_directories()
    print_structure()
