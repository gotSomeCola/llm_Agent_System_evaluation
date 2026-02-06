"""
Unified evaluation report generator for both pass@1 and pass@k results.
Automatically detects result type and generates appropriate reports.
"""
import json
import os
import argparse
import pandas as pd
from pathlib import Path
from config import BASELINE_REPORTS_DIR, PASS_AT_K_REPORTS_DIR, ensure_directories
from metrics import calculate_pass_at_k


def load_results(jsonl_file):
    """Load results from JSONL file"""
    results = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                results.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return results


def get_model_name_from_results(results):
    """Extract model name from results"""
    if results and len(results) > 0:
        return results[0].get('model_name', 'unknown_model')
    return 'unknown_model'


def detect_result_type(results):
    """
    Detect whether results are pass@1 or pass@k format.
    Returns: 'pass@k' if attempts field exists, 'pass@1' otherwise
    """
    if not results:
        return 'pass@1'
    
    # Check first result for attempts field
    if 'attempts' in results[0]:
        return 'pass@k'
    return 'pass@1'


def extract_metrics_from_attempts(attempts):
    """Extract metrics from attempts (for pass@k)"""
    passed_attempts = [a for a in attempts if a.get("pass", False)]
    
    if not passed_attempts:
        return {
            "num_passed": 0,
            "avg_code_bleu": 0.0,
            "avg_code_bert_f1": 0.0,
            "avg_runtime_ms": 0.0,
            "first_pass_attempt": None,
            "first_pass_code_bleu": 0.0,
            "first_pass_code_bert_f1": 0.0
        }
    
    first_pass = passed_attempts[0]
    avg_code_bleu = sum(a.get("code_bleu", 0) for a in passed_attempts) / len(passed_attempts)
    avg_code_bert_f1 = sum(a.get("code_bert_f1", 0) for a in passed_attempts) / len(passed_attempts)
    avg_runtime_ms = sum(a.get("runtime_ms", 0) for a in passed_attempts) / len(passed_attempts)
    
    return {
        "num_passed": len(passed_attempts),
        "avg_code_bleu": avg_code_bleu,
        "avg_code_bert_f1": avg_code_bert_f1,
        "avg_runtime_ms": avg_runtime_ms,
        "first_pass_attempt": first_pass.get("attempt", 0),
        "first_pass_code_bleu": first_pass.get("code_bleu", 0),
        "first_pass_code_bert_f1": first_pass.get("code_bert_f1", 0)
    }


def generate_csv_report_pass_at_1(results, output_file):
    """Generate CSV report for pass@1 baseline results"""
    rows = []
    for r in results:
        # Dynamically detect pass@k values
        pass_at_keys = [k for k in r.keys() if k.startswith('pass@')]
        
        row = {
            'ID': r.get('id', 'N/A'),
            'Title': r.get('title', 'N/A'),
            'Compilable': 'Yes' if r.get('compilable', False) else 'No',
            'Pass': 'Yes' if r.get('pass', False) else 'No',
            'Total Duration (s)': r.get('total_duration', 0),
            'Runtime (ms)': r.get('runtime_ms', 0),
            'Code BLEU': round(r.get('code_bleu', 0), 4),
            'CodeBERT F1': round(r.get('code_bert_f1', 0), 4),
        }
        
        # Add pass@k values if available
        for key in sorted(pass_at_keys):
            row[key] = f"{r.get(key, 0):.4f}"
        
        if r.get('error'):
            row['Error'] = r.get('error', '')
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✓ CSV report saved: {output_file}")
    return df


def generate_csv_report_pass_at_k(results, output_file):
    """Generate detailed CSV report for pass@k results"""
    rows = []
    
    for r in results:
        metrics = extract_metrics_from_attempts(r.get("attempts", []))
        
        # Dynamically build row with all pass@k values
        pass_at_keys = [k for k in r.keys() if k.startswith('pass@')]
        
        row = {
            'ID': r.get('id', 'N/A'),
            'Title': r.get('title', 'N/A'),
            'N_Attempts': r.get('n_attempts', r.get('n_samples', 0)),
            'C_Correct': r.get('c_correct', 0),
            'Pass@1_First': 'Yes' if r.get('pass@1_first', False) else 'No',
        }
        
        # Add all pass@k values
        for key in sorted(pass_at_keys):
            if key != 'pass@1_first':
                row[key] = f"{r.get(key, 0):.4f}"
        
        row.update({
            'First_Pass_Attempt': metrics['first_pass_attempt'] if metrics['first_pass_attempt'] is not None else 'N/A',
            'Avg_Code_BLEU': f"{metrics['avg_code_bleu']:.4f}",
            'Avg_CodeBERT_F1': f"{metrics['avg_code_bert_f1']:.4f}",
            'Avg_Runtime_ms': f"{metrics['avg_runtime_ms']:.2f}",
        })
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✓ CSV report saved: {output_file}")
    return df


def generate_summary_stats_pass_at_1(results, output_file, model_name):
    """Generate summary statistics for pass@1 baseline evaluation"""
    total = len(results)
    compilable = sum(1 for r in results if r.get('compilable', False))
    passed = sum(1 for r in results if r.get('pass', False))
    
    compile_rate = (compilable / total * 100) if total > 0 else 0
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    compilable_results = [r for r in results if r.get('compilable', False)]
    avg_runtime = sum(r.get('runtime_ms', 0) for r in compilable_results) / len(compilable_results) if compilable_results else 0
    avg_code_bleu = sum(r.get('code_bleu', 0) for r in results) / total if total > 0 else 0
    avg_codebert = sum(r.get('code_bert_f1', 0) for r in results) / total if total > 0 else 0
    
    summary = f"""
=== EVALUATION SUMMARY (PASS@1) ===
Model: {model_name}
Total Tasks: {total}

Success Rates:
- Compile Rate: {compile_rate:.2f}% ({compilable}/{total})
- Pass Rate: {pass_rate:.2f}% ({passed}/{total})

Average Metrics:
- Runtime (compilable only): {avg_runtime:.2f}ms
- Code BLEU: {avg_code_bleu:.4f}
- CodeBERT F1: {avg_codebert:.4f}
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"✓ Summary stats saved: {output_file}")
    print(summary)
    
    return {
        'model': model_name,
        'total': total,
        'compilable': compilable,
        'compile_rate': compile_rate,
        'passed': passed,
        'pass_rate': pass_rate,
        'avg_runtime': avg_runtime,
        'avg_code_bleu': avg_code_bleu,
        'avg_codebert': avg_codebert
    }


def generate_summary_stats_pass_at_k(results, output_file, model_name, k_override=None):
    """Generate summary statistics for pass@k evaluation"""
    total = len(results)
    if total == 0:
        print("No results to summarize.")
        return {}
    
    # Detect n_attempts from results
    n_attempts = results[0].get('n_attempts', results[0].get('n_samples', 1))
    
    # Determine k for reporting
    if k_override is not None:
        k = k_override
        print(f"Recalculating Pass@K metrics with k={k} (original n_attempts={n_attempts})")
    else:
        k = n_attempts
    
    # Count successes
    pass_first_count = sum(1 for r in results if r.get('pass@1_first', False))
    any_pass_count = sum(1 for r in results if r.get('c_correct', 0) > 0)

    # Calculate pass@k metrics (recalculate if k_override provided)
    if k_override is not None:
        pass_1_values = []
        pass_k_values = []
        for r in results:
            n = r.get('n_attempts', r.get('n_samples', 1))
            c = r.get('c_correct', 0)
            pass_1_values.append(calculate_pass_at_k(n, c, 1))
            pass_k_values.append(calculate_pass_at_k(n, c, k))
        avg_pass_k = sum(pass_k_values) / total

        if k >= 5 or n_attempts >= 5:
            pass_5_values = []
            for r in results:
                n = r.get('n_attempts', r.get('n_samples', 1))
                c = r.get('c_correct', 0)
                pass_5_values.append(calculate_pass_at_k(n, c, 5))
            avg_pass_5 = sum(pass_5_values) / total
        else:
            avg_pass_5 = None
    else:
        avg_pass_5 = sum(r.get('pass@5', 0) for r in results) / total if n_attempts >= 5 else None
        avg_pass_k = sum(r.get('pass@k', 0) for r in results) / total
    
    # Build summary
    summary = f"""
=== PASS@K EVALUATION SUMMARY ===
Model: {model_name}
Total Tasks: {total}
Attempts per Task: {n_attempts}

Success Rates:
- Pass@1 (First Attempt): {pass_first_count/total*100:.2f}% ({pass_first_count}/{total})
"""

    if avg_pass_5 is not None:
        summary += f"- Pass@5:     {avg_pass_5*100:.2f}%\n"
    summary += f"- Pass@{k}:     {avg_pass_k*100:.2f}%\n"
    
    summary += f"- Any Success in {n_attempts} tries: {any_pass_count/total*100:.2f}% ({any_pass_count}/{total})\n"
    
    # Code quality metrics (only for tasks with at least 1 success)
    successful_tasks = []
    for r in results:
        if r.get('c_correct', 0) > 0:
            metrics = extract_metrics_from_attempts(r.get("attempts", []))
            successful_tasks.append(metrics)
    
    if successful_tasks:
        avg_code_bleu = sum(m['avg_code_bleu'] for m in successful_tasks) / len(successful_tasks)
        avg_code_bert = sum(m['avg_code_bert_f1'] for m in successful_tasks) / len(successful_tasks)
        summary += f"""
Code Quality Metrics (Successful Tasks Only, n={len(successful_tasks)}):
- Avg Code BLEU:    {avg_code_bleu:.4f}
- Avg CodeBERT F1:  {avg_code_bert:.4f}
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"✓ Summary stats saved: {output_file}")
    print(summary)
    
    return {
        'model': model_name,
        'total_tasks': total,
        'k': k,
        'pass@1_first_count': pass_first_count,
        'pass@1_first_rate': round(pass_first_count / total * 100, 2),
        'pass@5': round(avg_pass_5 * 100, 2) if avg_pass_5 is not None else None,
        'pass@k': round(avg_pass_k * 100, 2),
        'any_pass_count': any_pass_count,
        'successful_tasks_count': len(successful_tasks),
        'avg_code_bleu': round(avg_code_bleu, 4),
        'avg_code_bert_f1': round(avg_code_bert, 4)
    }


def save_summary_jsonl(stats, output_file):
    """Save summary statistics to JSONL file (append mode for comparing multiple models)"""
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(stats) + '\n')
    
    print(f"✓ Summary JSONL saved: {output_file}")


def save_summary_csv(stats, output_file):
    """Save summary statistics to CSV (append mode for comparing multiple models)"""
    import csv
    
    file_exists = os.path.exists(output_file)
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'a', newline='', encoding='utf-8') as f:
        fieldnames = [
            'model', 'total_tasks', 'k',
            'pass@1_first_count', 'pass@1_first_rate_%',
            'pass@5_%', 'pass@k_%', 'any_pass_count',
            'successful_tasks_count', 'avg_code_bleu', 'avg_code_bert_f1'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        
        row = {
            'model': stats['model'],
            'total_tasks': stats['total_tasks'],
            'k': stats['k'],
            'pass@1_first_count': stats['pass@1_first_count'],
            'pass@1_first_rate_%': f"{stats['pass@1_first_rate']:.2f}",
            'pass@5_%': f"{stats['pass@5']:.2f}" if stats['pass@5'] is not None else 'N/A',
            'pass@k_%': f"{stats['pass@k']:.2f}",
            'any_pass_count': stats['any_pass_count'],
            'successful_tasks_count': stats['successful_tasks_count'],
            'avg_code_bleu': f"{stats['avg_code_bleu']:.4f}",
            'avg_code_bert_f1': f"{stats['avg_code_bert_f1']:.4f}"
        }
        writer.writerow(row)
    
    print(f"✓ Summary CSV saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Unified evaluation report generator')
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Output directory (auto-detected based on result type if not specified)')
    parser.add_argument('--k_values', type=str, default=None,
                       help='Comma-separated k values for recalculation (e.g. "1,3,5,10")')
    
    args = parser.parse_args()
    
    # Load results
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    results = load_results(args.input)
    if not results:
        print("Error: No results loaded from file")
        return
    
    model_name = get_model_name_from_results(results)
    result_type = detect_result_type(results)
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        if result_type == 'pass@k':
            output_dir = str(PASS_AT_K_REPORTS_DIR)
        else:
            output_dir = str(BASELINE_REPORTS_DIR)
    
    ensure_directories()
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate reports based on result type
    if result_type == 'pass@k':
        print(f"\n📊 Detected PASS@K results")
        csv_file = os.path.join(output_dir, f'{model_name}_pass_at_k.csv')
        summary_file = os.path.join(output_dir, f'{model_name}_pass_at_k_summary.txt')
        summary_jsonl_file = os.path.join(output_dir, 'summary_pass_at_k.jsonl')
        summary_csv_file = os.path.join(output_dir, 'summary_pass_at_k.csv')
        
        generate_csv_report_pass_at_k(results, csv_file)
        
        # Parse k_values for recalculation
        k_override = None
        if args.k_values:
            try:
                k_list = [int(k.strip()) for k in args.k_values.split(',')]
                if len(k_list) > 1:
                    print("Warning: multiple k values provided, using the first one")
                k_override = k_list[0]
            except ValueError:
                print("Warning: k_values must be comma-separated integers, using default")
        
        stats = generate_summary_stats_pass_at_k(results, summary_file, model_name, k_override)
        save_summary_jsonl(stats, summary_jsonl_file)
        save_summary_csv(stats, summary_csv_file)
    else:
        print(f"\n📊 Detected PASS@1 (baseline) results")
        csv_file = os.path.join(output_dir, f'{model_name}_baseline.csv')
        summary_file = os.path.join(output_dir, f'{model_name}_baseline_summary.txt')
        
        generate_csv_report_pass_at_1(results, csv_file)
        generate_summary_stats_pass_at_1(results, summary_file, model_name)
    
    print(f"\n✓ Reports generated successfully!")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
