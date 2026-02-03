#!/usr/bin/env python3
"""
Batch testing script: Run multiple model evaluations sequentially.
Script can run automatically - set it and leave it running.
"""
import subprocess
import json
import time
import os
from datetime import datetime
from config import PASS_AT_K_EVAL_DIR, ensure_directories

# Configuration
# Models to test
MODELS_TO_TEST = [

    #"gemma3:12b",
    #"gemma3:27b",
    #"llama3.1:70b",
    #"llama3.3:70b",
    #"gpt-oss:20b",
    "gpt-oss:120b",
]

# Evaluation parameters
EVAL_CONFIG = {
    "min_count": 101,
    "max_count": 601,
    "workers": 2,
    "k": 5,
    "use_at_k": True
}

# Output directory - use config-managed path
OUTPUT_DIR = str(PASS_AT_K_EVAL_DIR)

# Main program

def ensure_output_dir():
    """Ensure output directory exists."""
    ensure_directories()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_output_filename(model_name, timestamp):
    """Generate output filename."""
    # Replace special characters with underscore
    safe_model_name = model_name.replace(":", "_").replace("-", "_")
    return os.path.join(OUTPUT_DIR, f"results_{safe_model_name}_{timestamp}.jsonl")

def run_evaluation(model_name, output_file, config):
    """Run evaluation for a single model."""
    print(f"\n{'='*70}")
    print(f"Starting evaluation: {model_name}")
    print(f"{'='*70}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output file: {output_file}")
    print(f"Config: {config}")
    
    # Select script based on evaluation mode
    script = "run_baseline_at_k.py" if config["use_at_k"] else "run_baseline.py"
    
    # Build command
    cmd = [
        "python", script,
        "--model_name", model_name,
        "--output_file", output_file,
        "--min_count", str(config["min_count"]),
        "--max_count", str(config["max_count"]),
        "--workers", str(config["workers"]),
    ]
    
    # Add k parameter if using pass@k
    if config["use_at_k"]:
        cmd.extend(["--k", str(config["k"])])
    
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        # Run evaluation script
        print("Running evaluation...")
        result = subprocess.run(cmd, check=True)
        
        # Verify output file was created
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"\n✓ Evaluation successful!")
            print(f"JSONL output: {output_file}")
            print(f"File size: {file_size / 1024:.2f} KB")
            
            # Auto-generate formatted report
            print(f"\nGenerating formatted report...")
            generate_report(model_name, output_file, config["use_at_k"])
            
            return True
        else:
            print(f"\n⚠ Warning: Output file not created")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Evaluation failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False

def generate_report(model_name, jsonl_file, use_at_k=False):
    """Generate formatted report from evaluation results."""
    try:
        # Select report generation script based on mode
        script = "generate_pass_at_k_report.py" if use_at_k else "generate_pass_at_1_report.py"
        
        cmd = [
            "python", script,
            "--input", jsonl_file,
        ]
        
        print(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print(f"✓ Report generated successfully!")
        print(f"Output files:")
        
        # Extract generated filenames
        base_path = os.path.splitext(jsonl_file)[0]
        output_dir = os.path.dirname(jsonl_file)
        
        if use_at_k:
            report_files = {
                "CSV Detail": f"{base_path}.csv",
                "TXT Summary": f"{base_path}_summary.txt",
            }
            summary_csv = os.path.join(output_dir, "summary_pass_at_k.csv")
        else:
            report_files = {
                "CSV": f"{base_path}.csv",
                "TXT Summary": f"{base_path}_summary.txt",
            }
            summary_csv = os.path.join(output_dir, "summary_pilot.csv")
        
        for file_type, file_path in report_files.items():
            if os.path.exists(file_path):
                print(f"  {file_type}: {file_path}")
        
        # Show comparison file if exists
        if os.path.exists(summary_csv):
            print(f"  Comparison: {summary_csv}")
        
    except subprocess.CalledProcessError as e:
        print(f"⚠ Report generation failed: {e}")
    except Exception as e:
        print(f"⚠ Report generation error: {e}")

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("Batch LLM Code Generation Evaluation")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models to test: {len(MODELS_TO_TEST)}")
    print(f"Model list: {', '.join(MODELS_TO_TEST)}")
    mode = f"Pass@{EVAL_CONFIG['k']} (multiple attempts)" if EVAL_CONFIG['use_at_k'] else "Pass@1 (single attempt)"
    print(f"Evaluation mode: {mode}")
    print("="*70 + "\n")
    
    ensure_output_dir()
    
    # Record timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_summary = {
        "start_time": datetime.now().isoformat(),
        "config": EVAL_CONFIG,
        "models": [],
        "end_time": None
    }
    
    # Run each model sequentially
    for i, model_name in enumerate(MODELS_TO_TEST, 1):
        output_file = generate_output_filename(model_name, timestamp)
        
        print(f"\n[{i}/{len(MODELS_TO_TEST)}] Processing: {model_name}")
        
        # Run evaluation
        success = run_evaluation(model_name, output_file, EVAL_CONFIG)
        
        # Record result
        model_result = {
            "model": model_name,
            "output_file": output_file,
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
        results_summary["models"].append(model_result)
        
        # Wait before next model if not the last one
        if i < len(MODELS_TO_TEST):
            print(f"\nWaiting 10s before next model...")
            time.sleep(10)
    
    # Complete
    results_summary["end_time"] = datetime.now().isoformat()
    
    # Save summary
    summary_file = os.path.join(OUTPUT_DIR, f"batch_summary_{timestamp}.json")
    with open(summary_file, "w") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    # Print final summary
    print(f"\n" + "="*70)
    print("Batch Testing Complete!")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Summary: {summary_file}")
    mode = f"Pass@{EVAL_CONFIG['k']} (multiple attempts)" if EVAL_CONFIG['use_at_k'] else "Pass@1 (single attempt)"
    print(f"Evaluation mode: {mode}\n")
    
    for i, model_result in enumerate(results_summary["models"], 1):
        status = "✓ Success" if model_result["success"] else "✗ Failed"
        print(f"{i}. {model_result['model']}: {status}")
        print(f"   Output: {model_result['output_file']}")
        
        # Show generated report files
        base_path = os.path.splitext(model_result['output_file'])[0]
        output_dir = os.path.dirname(model_result['output_file'])
        report_files = {
            "CSV": f"{base_path}.csv",
            "Summary": f"{base_path}_summary.txt",
        }
        
        print(f"   Reports:")
        for file_type, file_path in report_files.items():
            if os.path.exists(file_path):
                print(f"      {file_type}: {file_path}")
        
        # Show comparison file
        if EVAL_CONFIG["use_at_k"]:
            summary_file = os.path.join(output_dir, "summary_pass_at_k.csv")
        else:
            summary_file = os.path.join(output_dir, "summary_pilot.csv")
        
        if os.path.exists(summary_file):
            print(f"      Comparison: {summary_file}")
        print()
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
