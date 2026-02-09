#!/usr/bin/env python3
"""
Batch automation script to run coder_tester_system.py for multiple models.
Executes models sequentially, generates reports, and compiles results.
"""
import subprocess
import json
import time
import os
import shutil
from datetime import datetime
from config import REPAIRS_EVAL_DIR, ensure_directories

# Configuration
MODELS_TO_TEST = [
    #"gpt-oss:20b",
    #"gpt-oss:120b",
    "gemma3:12b",
    #"llama3.3:70b",
]

# Evaluation config (same for all models)
EVAL_CONFIG = {
    "min_count": 101,
    "max_count": 601,
    "workers": 3,
}

# Output directory - use config-managed path
OUTPUT_DIR = str(REPAIRS_EVAL_DIR)


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    ensure_directories()
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_output_filename(model_name, timestamp):
    """Generate output filename based on model name and timestamp."""
    safe_model_name = model_name.replace(":", "_").replace("-", "_").lower()
    return os.path.join(OUTPUT_DIR, f"results_{safe_model_name}_{timestamp}.jsonl")


def calculate_metrics(jsonl_file):
    """Calculate statistics from JSONL results file."""
    if not os.path.exists(jsonl_file):
        return None
    
    total = pass_count = compilable_count = pass_at_1_count = pass_at_k_count = 0
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if 'error' in record:
                        continue
                    
                    total += 1
                    if record.get('pass', False):
                        pass_count += 1
                    if record.get('compilable', False):
                        compilable_count += 1
                    if record.get('pass_at_1', 0.0) > 0:
                        pass_at_1_count += 1
                    if record.get('pass_at_k', 0.0) > 0:
                        pass_at_k_count += 1
                except json.JSONDecodeError:
                    continue
        
        if total == 0:
            return None
        
        return {
            "total": total,
            "pass": pass_count,
            "compilable": compilable_count,
            "pass_at_1": pass_at_1_count,
            "pass_at_k": pass_at_k_count,
            "pass_rate": round(pass_count / total * 100, 2),
            "compilable_rate": round(compilable_count / total * 100, 2),
            "pass_at_1_rate": round(pass_at_1_count / total * 100, 2),
            "pass_at_k_rate": round(pass_at_k_count / total * 100, 2)
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None



def run_coder_tester(model_name, output_file, config):
    """Execute coder_tester_system.py for a single model."""
    print(f"\n{'='*70}")
    print(f"Starting model: {model_name}")
    print(f"{'='*70}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Range: {config['min_count']} - {config['max_count']}")
    print(f"Workers: {config['workers']}")
    print(f"Output file: {output_file}")
    
    cmd = [
        "python", "coder_tester_system.py",
        "--model_name", model_name,
        "--output_file", output_file,
        "--min_count", str(config["min_count"]),
        "--max_count", str(config["max_count"]),
        "--workers", str(config["workers"]),
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        print("Running...")
        start_time = time.time()
        
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        elapsed = time.time() - start_time
        
        if "results_graph.jsonl" in result.stdout:
            if os.path.exists("results_graph.jsonl"):
                shutil.copy("results_graph.jsonl", output_file)
                print(f"\n✓ Test completed successfully")
                print(f"Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}m)")
                file_size = os.path.getsize(output_file)
                print(f"Output: {output_file}")
                print(f"Size: {file_size / 1024:.2f} KB")
                return True
        
        if os.path.exists(output_file):
            print(f"\n✓ Test completed successfully")
            print(f"Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}m)")
            file_size = os.path.getsize(output_file)
            print(f"Output: {output_file}")
            print(f"Size: {file_size / 1024:.2f} KB")
            return True
        else:
            print(f"\n✗ Warning: Output file not created")
            print(f"Stdout: {result.stdout}")
            print(f"Stderr: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Execution failed")
        print(f"Error code: {e.returncode}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def generate_report(jsonl_file):
    """Generate formatted report from JSONL results."""
    try:
        cmd = [
            "python", "generate_results_report.py",
            "--input", jsonl_file,
        ]
        
        print(f"\nGenerating report...")
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print(f"✓ Report generated")
        
        base_path = os.path.splitext(jsonl_file)[0]
        report_files = {
            "CSV": f"{base_path}.csv",
            "Summary": f"{base_path}_summary.txt",
        }
        
        print(f"Output files:")
        for file_type, file_path in report_files.items():
            if os.path.exists(file_path):
                print(f"  {file_type}: {file_path}")
        
    except subprocess.CalledProcessError as e:
        print(f"Warning: Report generation failed: {e}")
    except Exception as e:
        print(f"Warning: Report error: {e}")


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("Batch Model Evaluation")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models: {len(MODELS_TO_TEST)}")
    print("Model list:")
    for model in MODELS_TO_TEST:
        print(f"  - {model}")
    print(f"Config: min_count={EVAL_CONFIG['min_count']}, max_count={EVAL_CONFIG['max_count']}, workers={EVAL_CONFIG['workers']}")
    print("="*70 + "\n")
    
    ensure_output_dir()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_summary = {
        "start_time": datetime.now().isoformat(),
        "config": {
            "models": MODELS_TO_TEST,
            "eval_config": EVAL_CONFIG,
            "output_dir": OUTPUT_DIR,
        },
        "models_results": [],
        "end_time": None,
        "total_success": 0,
        "total_failed": 0,
    }
    
    # Run each model sequentially
    for i, model_name in enumerate(MODELS_TO_TEST, 1):
        output_file = generate_output_filename(model_name, timestamp)
        
        print(f"\n[{i}/{len(MODELS_TO_TEST)}] Processing: {model_name}")
        
        success = run_coder_tester(model_name, output_file, EVAL_CONFIG)
        
        model_result = {
            "model_name": model_name,
            "min_count": EVAL_CONFIG["min_count"],
            "max_count": EVAL_CONFIG["max_count"],
            "workers": EVAL_CONFIG["workers"],
            "output_file": output_file,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "metrics": None
        }
        
        if success and os.path.exists(output_file):
            metrics = calculate_metrics(output_file)
            model_result["metrics"] = metrics
        
        results_summary["models_results"].append(model_result)
        
        if success:
            results_summary["total_success"] += 1
            generate_report(output_file)
        else:
            results_summary["total_failed"] += 1
        
        # Wait before next model
        if i < len(MODELS_TO_TEST):
            wait_time = 5
            print(f"\nWaiting {wait_time}s before next model...")
            time.sleep(wait_time)
    
    results_summary["end_time"] = datetime.now().isoformat()
    
    # Save summary
    summary_file = os.path.join(OUTPUT_DIR, f"batch_summary_{timestamp}.json")
    with open(summary_file, "w") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n" + "="*70)
    print("Batch Evaluation Complete")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Summary: {summary_file}")
    print(f"Success: {results_summary['total_success']}/{len(MODELS_TO_TEST)}")
    print(f"Failed: {results_summary['total_failed']}/{len(MODELS_TO_TEST)}")
    print("\nModel details:")
    
    for i, model_result in enumerate(results_summary["models_results"], 1):
        status = "✓" if model_result["success"] else "✗"
        print(f"\n{i}. {model_result['model_name']}: {status}")
        print(f"   Range: [{model_result['min_count']}, {model_result['max_count']})")
        print(f"   Workers: {model_result['workers']}")
        print(f"   Output: {model_result['output_file']}")
        
        if model_result.get("metrics"):
            m = model_result["metrics"]
            print(f"   Metrics:")
            print(f"      Total: {m['total']}")
            print(f"      Pass Rate: {m['pass']}/{m['total']} ({m['pass_rate']}%)")
            print(f"      Compilable Rate: {m['compilable']}/{m['total']} ({m['compilable_rate']}%)")
            print(f"      Pass@1 Rate: {m['pass_at_1']}/{m['total']} ({m['pass_at_1_rate']}%)")
            print(f"      Pass@k Rate: {m['pass_at_k']}/{m['total']} ({m['pass_at_k_rate']}%)")
        
        base_path = os.path.splitext(model_result['output_file'])[0]
        report_files = {
            "CSV": f"{base_path}.csv",
            "Summary": f"{base_path}_summary.txt",
        }
        
        print(f"   Reports:")
        for file_type, file_path in report_files.items():
            if os.path.exists(file_path):
                print(f"      {file_type}: {file_path}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
