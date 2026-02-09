import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PASS_AT_K_REPORTS_DIR


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

    Strategy: Use the first successful attempt's code quality metrics.
    """
    passed_attempts = [a for a in attempts if a.get("pass", False)]

    if not passed_attempts:
        return {
            "num_passed": 0,
            "avg_code_bleu": 0.0,
            "avg_code_bert_f1": 0.0,
            "first_pass_attempt": None,
            "first_pass_code_bleu": 0.0,
            "first_pass_code_bert_f1": 0.0
        }

    # Find first successful attempt
    first_pass = passed_attempts[0]

    # Use first successful attempt metrics only
    avg_code_bleu = first_pass.get("code_bleu", 0)
    avg_code_bert_f1 = first_pass.get("code_bert_f1", 0)

    return {
        "num_passed": len(passed_attempts),
        "avg_code_bleu": avg_code_bleu,
        "avg_code_bert_f1": avg_code_bert_f1,
        "first_pass_attempt": first_pass.get("attempt", 0),
        "first_pass_code_bleu": first_pass.get("code_bleu", 0),
        "first_pass_code_bert_f1": first_pass.get("code_bert_f1", 0)
    }


def generate_csv_report(results, output_file):
    """Generate detailed CSV report for pass@k results"""
    rows = []

    for r in results:
        metrics = extract_metrics_from_attempts(r.get("attempts", []))
        attempts = r.get("attempts", [])
        first_attempt_compilable = 1 if attempts and attempts[0].get("compilable", False) else 0
        pass_at_1_attempt = 1 if attempts and attempts[0].get("pass", False) else 0

        rows.append({
            'ID': r.get('id', 'N/A'),
            'Title': r.get('title', 'N/A'),
            'N_Samples': r.get('n_samples', 0),
            'Pass_At_1_Attempt': pass_at_1_attempt,
            'Num_Passed': metrics['num_passed'],
            'Compilable_Count': first_attempt_compilable,
            'Compilable_Rate': f"{float(first_attempt_compilable):.4f}",
            'First_Pass_Attempt': metrics['first_pass_attempt'] if metrics['first_pass_attempt'] is not None else 'N/A',
            'Avg_Code_BLEU': f"{metrics['avg_code_bleu']:.4f}",
            'Avg_CodeBERT_F1': f"{metrics['avg_code_bert_f1']:.4f}",
            'First_Pass_Code_BLEU': f"{metrics['first_pass_code_bleu']:.4f}",
            'First_Pass_CodeBERT_F1': f"{metrics['first_pass_code_bert_f1']:.4f}"
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✓ CSV report saved: {output_file}")
    return df


def generate_summary_stats(results, model_name):
    """Generate summary statistics for pass@1 attempt and num_passed."""
    total = len(results)
    if total == 0:
        print("No results to summarize.")
        return {}

    pass_at_1_attempt_count = 0
    compilable_at_1_attempt_count = 0
    at_least_1_pass = 0
    attempts_to_pass_total = 0
    total_code_bleu = 0.0
    total_code_bert = 0.0

    for r in results:
        attempts = r.get("attempts", [])
        if attempts:
            if attempts[0].get("pass", False):
                pass_at_1_attempt_count += 1
            if attempts[0].get("compilable", False):
                compilable_at_1_attempt_count += 1

        metrics = extract_metrics_from_attempts(attempts)
        attempts_count = len(attempts)
        if metrics["num_passed"] > 0:
            at_least_1_pass += 1
            first_pass_attempt = metrics.get("first_pass_attempt")
            if first_pass_attempt is None:
                attempts_needed = attempts_count
            else:
                attempts_needed = first_pass_attempt + 1
        else:
            attempts_needed = attempts_count

        attempts_to_pass_total += attempts_needed

        total_code_bleu += metrics.get("avg_code_bleu", 0.0)
        total_code_bert += metrics.get("avg_code_bert_f1", 0.0)

    avg_code_bleu = total_code_bleu / total if total > 0 else 0.0
    avg_code_bert = total_code_bert / total if total > 0 else 0.0

    return {
        'model': model_name,
        'total_tasks': total,
        'avg_att_to_pass*': round(attempts_to_pass_total / total, 4),
        'p1_cnt': pass_at_1_attempt_count,
        'p1_rate*': round(pass_at_1_attempt_count / total * 100, 2),
        'comp1_cnt': compilable_at_1_attempt_count,
        'comp1_rate*': round(compilable_at_1_attempt_count / total * 100, 2),
        'p_any_cnt': at_least_1_pass,
        'p_any_rate*': round(at_least_1_pass / total * 100, 2),
        'avg_code_bleu': round(avg_code_bleu, 4),
        'avg_code_bert_f1': round(avg_code_bert, 4)
    }


def save_summary_jsonl(stats, output_file):
    """Save summary statistics to JSONL file (append mode for comparing multiple models)"""
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

    ordered_keys = [
        'model', 'total_tasks', 'avg_att_to_pass*',
        'p1_cnt', 'p1_rate*',
        'comp1_cnt', 'comp1_rate*',
        'p_any_cnt', 'p_any_rate*',
        'avg_code_bleu', 'avg_code_bert_f1'
    ]
    ordered_stats = {k: stats.get(k) for k in ordered_keys}

    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(ordered_stats) + '\n')

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
            'model', 'total_tasks',
            'avg_att_to_pass*',
            'p1_cnt', 'p1_rate*',
            'comp1_cnt', 'comp1_rate*',
            'p_any_cnt', 'p_any_rate*',
            'avg_code_bleu', 'avg_code_bert_f1'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        row = {
            'model': stats['model'],
            'total_tasks': stats['total_tasks'],
            'avg_att_to_pass*': f"{stats['avg_att_to_pass*']:.4f}",  # avg_attempts_to_pass
            'p1_cnt': stats['p1_cnt'],  # pass@1 attempt count
            'p1_rate*': f"{stats['p1_rate*']:.2f}",  # pass@1 attempt rate
            'comp1_cnt': stats['comp1_cnt'],  # compilable at attempt 1 count
            'comp1_rate*': f"{stats['comp1_rate*']:.2f}",  # compilable at attempt 1 rate
            'p_any_cnt': stats['p_any_cnt'],  # at least one pass count
            'p_any_rate*': f"{stats['p_any_rate*']:.2f}",  # at least one pass rate
            'avg_code_bleu': f"{stats['avg_code_bleu']:.4f}",
            'avg_code_bert_f1': f"{stats['avg_code_bert_f1']:.4f}"
        }
        writer.writerow(row)

    print(f"✓ Summary CSV saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate pass@k evaluation reports')
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file with pass@k results')
    parser.add_argument('--output_dir', type=str, default=str(PASS_AT_K_REPORTS_DIR), help='Output directory for reports')
    parser.add_argument('--model_name', type=str, default=None, help='Model name (auto-detected if not provided)')
    parser.add_argument('--k', type=int, default=None, help='(deprecated) Not used. Pass@k is no longer calculated.')

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

    task_count = len(results)

    # Sanitize model name for filename (replace : with -)
    safe_model_name = model_name.replace(':', '-').replace('/', '-')

    # Generate reports
    csv_file = os.path.join(args.output_dir, f"{safe_model_name}_tasks{task_count}.csv")
    summary_jsonl_file = os.path.join(args.output_dir, 'summary_pass_at_k.jsonl')
    summary_csv_file = os.path.join(args.output_dir, 'summary_pass_at_k.csv')

    generate_csv_report(results, csv_file)
    stats = generate_summary_stats(results, model_name)
    save_summary_jsonl(stats, summary_jsonl_file)
    save_summary_csv(stats, summary_csv_file)

    print("\n✓ All reports generated successfully!")


if __name__ == "__main__":
    main()