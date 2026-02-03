"""
Generate pass@k results report from JSONL files with multiple attempts per task.
Calculates metrics based on successful attempts only.
"""
import json
import os
import argparse
from datetime import datetime
import pandas as pd
from pathlib import Path
from config import PASS_AT_K_REPORTS_DIR, ensure_directories


def load_pass_at_k_results(jsonl_file):
    """Load pass@k results from JSONL file"""
    results = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                results.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return results


def extract_metrics_from_attempts(attempts):
    """
    Extract metrics from attempts.
    Returns metrics from successful attempts only.
    
    Strategy: Calculate average of all successful attempts' code quality metrics.
    """
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
    
    # Find first successful attempt
    first_pass = passed_attempts[0]
    
    # Calculate averages for successful attempts
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


def generate_csv_report(results, output_file):
    """Generate detailed CSV report for pass@k results"""
    rows = []
    
    for r in results:
        metrics = extract_metrics_from_attempts(r.get("attempts", []))
        
        rows.append({
            'ID': r.get('id', 'N/A'),
            'Title': r.get('title', 'N/A'),
            'N_Samples': r.get('n_samples', 0),
            'C_Correct': r.get('c_correct', 0),
            'Pass@1_First': 'Yes' if r.get('pass@1_first', False) else 'No',
            'Pass@1': f"{r.get('pass@1', 0):.4f}",
            'Pass@5': f"{r.get('pass@5', 0):.4f}",
            'Pass@K': f"{r.get('pass@k', 0):.4f}",
            'First_Pass_Attempt': metrics['first_pass_attempt'] if metrics['first_pass_attempt'] is not None else 'N/A',
            'Avg_Code_BLEU': f"{metrics['avg_code_bleu']:.4f}",
            'Avg_CodeBERT_F1': f"{metrics['avg_code_bert_f1']:.4f}",
            'Avg_Runtime_ms': f"{metrics['avg_runtime_ms']:.2f}",
            'First_Pass_Code_BLEU': f"{metrics['first_pass_code_bleu']:.4f}",
            'First_Pass_CodeBERT_F1': f"{metrics['first_pass_code_bert_f1']:.4f}"
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✓ CSV report saved: {output_file}")
    return df


def generate_summary_stats(results, output_file, model_name):
    """Generate summary statistics for pass@k evaluation"""
    total = len(results)
    if total == 0:
        print("No results to summarize.")
        return {}
    
    # Extract k value (n_samples)
    k = results[0].get('n_samples', 1) if results else 1
    
    # Count successes
    pass_first_count = sum(1 for r in results if r.get('pass@1_first', False))
    any_pass_count = sum(1 for r in results if r.get('c_correct', 0) > 0)
    
    # Calculate average pass@k probabilities
    avg_pass_1 = sum(r.get('pass@1', 0) for r in results) / total
    avg_pass_5 = sum(r.get('pass@5', 0) for r in results) / total if k >= 5 else None
    avg_pass_k = sum(r.get('pass@k', 0) for r in results) / total
    
    # Calculate code quality metrics (only for tasks with at least 1 success)
    successful_tasks = []
    for r in results:
        if r.get('c_correct', 0) > 0:
            metrics = extract_metrics_from_attempts(r.get("attempts", []))
            successful_tasks.append(metrics)
    
    if successful_tasks:
        avg_code_bleu = sum(m['avg_code_bleu'] for m in successful_tasks) / len(successful_tasks)
        avg_code_bert = sum(m['avg_code_bert_f1'] for m in successful_tasks) / len(successful_tasks)
        avg_runtime = sum(m['avg_runtime_ms'] for m in successful_tasks) / len(successful_tasks)
        
        first_pass_avg_bleu = sum(m['first_pass_code_bleu'] for m in successful_tasks) / len(successful_tasks)
        first_pass_avg_bert = sum(m['first_pass_code_bert_f1'] for m in successful_tasks) / len(successful_tasks)
    else:
        avg_code_bleu = avg_code_bert = avg_runtime = 0.0
        first_pass_avg_bleu = first_pass_avg_bert = 0.0
    
    summary = f"""
=== PASS@K EVALUATION SUMMARY ===
Model: {model_name}
Total Tasks: {total}
Samples per Task (k): {k}

Success Rates:
- Pass@1 (First Attempt): {pass_first_count/total*100:.2f}% ({pass_first_count}/{total})
- Pass@1 (Estimated):     {avg_pass_1*100:.2f}%"""
    
    if avg_pass_5 is not None:
        summary += f"\n- Pass@5 (Estimated):     {avg_pass_5*100:.2f}%"
    
    summary += f"""
- Pass@{k} (Estimated):     {avg_pass_k*100:.2f}%
- Any Success in {k} tries:  {any_pass_count/total*100:.2f}% ({any_pass_count}/{total})

Code Quality Metrics (Successful Tasks Only, n={len(successful_tasks)}):
Average across all successful attempts:
- Avg Code BLEU:    {avg_code_bleu:.4f}
- Avg CodeBERT F1:  {avg_code_bert:.4f}
- Avg Runtime:      {avg_runtime:.2f}ms

First successful attempt only:
- Avg Code BLEU:    {first_pass_avg_bleu:.4f}
- Avg CodeBERT F1:  {first_pass_avg_bert:.4f}

Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"✓ Summary stats saved: {output_file}")
    print(summary)
    
    # Return stats as dictionary for JSONL export
    return {
        'model': model_name,
        'total_tasks': total,
        'k': k,
        'pass@1_first_count': pass_first_count,
        'pass@1_first_rate': round(pass_first_count / total * 100, 2),
        'pass@1_estimated': round(avg_pass_1 * 100, 2),
        'pass@5_estimated': round(avg_pass_5 * 100, 2) if avg_pass_5 is not None else None,
        'pass@k_estimated': round(avg_pass_k * 100, 2),
        'any_pass_count': any_pass_count,
        'any_pass_rate': round(any_pass_count / total * 100, 2),
        'successful_tasks_count': len(successful_tasks),
        'avg_code_bleu': round(avg_code_bleu, 4),
        'avg_code_bert_f1': round(avg_code_bert, 4),
        'avg_runtime_ms': round(avg_runtime, 2),
        'first_pass_avg_code_bleu': round(first_pass_avg_bleu, 4),
        'first_pass_avg_code_bert_f1': round(first_pass_avg_bert, 4),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }


def save_summary_jsonl(stats, output_file):
    """Save summary statistics to JSONL file (append mode for comparing multiple models)"""
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(stats) + '\n')
    
    print(f"✓ Summary JSONL saved: {output_file}")


def save_summary_csv(stats, output_file=None):
    """Save summary statistics to CSV (append mode for comparing multiple models)"""
    import csv
    
    if output_file is None:
        output_file = str(PASS_AT_K_REPORTS_DIR / 'summary_pass_at_k.csv')
    
    file_exists = os.path.exists(output_file)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'a', newline='', encoding='utf-8') as f:
        fieldnames = [
            'timestamp', 'model', 'total_tasks', 'k',
            'pass@1_first_count', 'pass@1_first_rate_%', 'pass@1_estimated_%', 
            'pass@5_estimated_%', 'pass@k_estimated_%', 'any_pass_count', 'any_pass_rate_%',
            'successful_tasks_count', 'avg_code_bleu', 'avg_code_bert_f1', 'avg_runtime_ms',
            'first_pass_avg_code_bleu', 'first_pass_avg_code_bert_f1'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        row = {
            'timestamp': stats['timestamp'],
            'model': stats['model'],
            'total_tasks': stats['total_tasks'],
            'k': stats['k'],
            'pass@1_first_count': stats['pass@1_first_count'],
            'pass@1_first_rate_%': f"{stats['pass@1_first_rate']:.2f}",
            'pass@1_estimated_%': f"{stats['pass@1_estimated']:.2f}",
            'pass@5_estimated_%': f"{stats['pass@5_estimated']:.2f}" if stats['pass@5_estimated'] is not None else 'N/A',
            'pass@k_estimated_%': f"{stats['pass@k_estimated']:.2f}",
            'any_pass_count': stats['any_pass_count'],
            'any_pass_rate_%': f"{stats['any_pass_rate']:.2f}",
            'successful_tasks_count': stats['successful_tasks_count'],
            'avg_code_bleu': f"{stats['avg_code_bleu']:.4f}",
            'avg_code_bert_f1': f"{stats['avg_code_bert_f1']:.4f}",
            'avg_runtime_ms': f"{stats['avg_runtime_ms']:.2f}",
            'first_pass_avg_code_bleu': f"{stats['first_pass_avg_code_bleu']:.4f}",
            'first_pass_avg_code_bert_f1': f"{stats['first_pass_avg_code_bert_f1']:.4f}"
        }
        writer.writerow(row)
    
    print(f"✓ Summary CSV saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate pass@k evaluation reports')
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file with pass@k results')
    parser.add_argument('--output_dir', type=str, default=str(PASS_AT_K_REPORTS_DIR), help='Output directory for reports')
    parser.add_argument('--model_name', type=str, default=None, help='Model name (auto-detected if not provided)')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.input}...")
    results = load_pass_at_k_results(args.input)
    print(f"Loaded {len(results)} task results")
    
    if not results:
        print("No results found!")
        return
    
    # Get model name
    model_name = args.model_name or results[0].get('model_name', 'unknown_model')
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    task_count = len(results)
    
    # Sanitize model name for filename (replace : with -)
    safe_model_name = model_name.replace(':', '-').replace('/', '-')
    
    # Generate reports
    csv_file = os.path.join(args.output_dir, f"{safe_model_name}_tasks{task_count}_{timestamp}.csv")
    summary_file = os.path.join(args.output_dir, f"{safe_model_name}_tasks{task_count}_{timestamp}_summary.txt")
    summary_jsonl_file = os.path.join(args.output_dir, 'summary_pass_at_k.jsonl')
    summary_csv_file = os.path.join(args.output_dir, 'summary_pass_at_k.csv')
    
    generate_csv_report(results, csv_file)
    stats = generate_summary_stats(results, summary_file, model_name)
    save_summary_jsonl(stats, summary_jsonl_file)
    save_summary_csv(stats, summary_csv_file)
    
    print("\n✓ All reports generated successfully!")


if __name__ == "__main__":
    main()
