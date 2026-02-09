#!/usr/bin/env python3
"""
Report generation script for repair round test results.
Processes JSONL result files with repair history and generates comprehensive reports.

This script analyzes the test results which contain:
- model_name: The LLM model used
- id: Problem identifier
- title: Problem title
- pass: Whether the problem was eventually solved
- pass_at_1, pass_at_k: Pass rate metrics
- compilable: Whether the final code compiles
- repair_rounds: Number of repair iterations needed
- code_bleu, code_bert_f1: Code quality metrics
- history: List of repair attempts with pass/fail status
"""

import csv
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def load_results(input_file):
    """
    Load JSONL results file.

    Args:
        input_file: Path to the JSONL file

    Returns:
        List of dictionaries, each containing one test result
    """
    results = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def extract_repair_metrics(result):
    """
    Extract repair-related metrics from a single result record.

    Args:
        result: Dictionary containing one test result

    Returns:
        Dictionary with extracted metrics for analysis
    """
    history = result.get('history', [])
    pass_at_1 = result.get('pass_at_1', 0.0)
    metrics = {
        'model_name': result.get('model_name', ''),
        'id': result.get('id', ''),
        'title': result.get('title', ''),
        'pass': result.get('pass', False),
        'pass_at_1': pass_at_1,
        'pass_at_k': result.get('pass_at_k', 0.0),
        'compilable': result.get('compilable', False),
        'repair_rounds': result.get('repair_rounds', 0),
        'code_bleu': result.get('code_bleu', 0.0),
        'code_bert_f1': result.get('code_bert_f1', 0.0),
    }

    return metrics


def generate_csv_report(results, output_file):
    """
    Generate CSV report with metrics for each problem.

    Args:
        results: List of result dictionaries
        output_file: Path to save CSV file

    Returns:
        List of metrics dictionaries
    """
    metrics_list = [extract_repair_metrics(r) for r in results]

    # Define CSV columns
    fieldnames = [
        'model_name', 'id', 'title', 'pass', 'pass_at_1', 'pass_at_k',
        'compilable', 'repair_rounds', 'code_bleu', 'code_bert_f1'
    ]

    # Write CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_list)

    return metrics_list


def generate_summary_stats(results, metrics_list):
    """
    Generate comprehensive summary statistics.

    Args:
        results: List of result dictionaries
        metrics_list: List of metrics dictionaries

    Returns:
        Dictionary containing all summary statistics
    """
    total_problems = len(results)

    total_passed = sum(1 for m in metrics_list if m['pass'])
    compilable_count_at_last = sum(1 for m in metrics_list if m['compilable'])
    pass_at_1_attempt_count = sum(1 for m in metrics_list if m.get('pass_at_1', 0.0) >= 1.0)

    max_attempts = 5
    attempts_to_pass_total = 0
    for m in metrics_list:
        repair_rounds = m.get('repair_rounds', 0)
        attempts_needed = min(repair_rounds + 1, max_attempts)
        attempts_to_pass_total += attempts_needed

    stats = {
        'total_problems': total_problems,
        'p1_cnt': pass_at_1_attempt_count,
        'p1_rate*': (pass_at_1_attempt_count / total_problems * 100) if total_problems > 0 else 0,
        'comp_last_cnt': compilable_count_at_last,
        'comp_last_rate*': (compilable_count_at_last / total_problems * 100) if total_problems > 0 else 0,
        'p_any_cnt': total_passed,
        'p_any_rate*': (total_passed / total_problems * 100) if total_problems > 0 else 0,
        'avg_att_to_pass*': (attempts_to_pass_total / total_problems) if total_problems > 0 else 0,
        'avg_code_bleu': sum(m['code_bleu'] for m in metrics_list) / total_problems if total_problems > 0 else 0,
        'avg_code_bert_f1': sum(m['code_bert_f1'] for m in metrics_list) / total_problems if total_problems > 0 else 0,
    }

    return stats


def save_summary_stats(stats, output_file):
    """
    Save summary statistics to JSON file.

    Args:
        stats: Dictionary of statistics
        output_file: Path to save JSON file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def print_summary_stats(stats):
    """
    Print summary statistics to console in human-readable format.

    Args:
        stats: Dictionary of statistics
    """
    print("\n" + "=" * 70)
    print("REPAIR ROUND TEST RESULTS - SUMMARY REPORT")
    print("=" * 70)

    print("\nSUMMARY METRICS:")
    print(f"  Total Problems:              {stats['total_problems']}")
    print(f"  Pass@1 (Attempt) Count:      {stats['p1_cnt']}")
    print(f"  Pass@1 (Attempt) Rate:       {stats['p1_rate*']:.2f}%")
    print(f"  Compilable Count (Last):     {stats['comp_last_cnt']}")
    print(f"  Compilable Rate (Last):      {stats['comp_last_rate*']:.2f}%")
    print(f"  At Least 1 Pass Count:       {stats['p_any_cnt']}")
    print(f"  At Least 1 Pass Rate:        {stats['p_any_rate*']:.2f}%")
    print(f"  Avg Attempts To Pass:        {stats['avg_att_to_pass*']:.2f}")

    print("\nCODE QUALITY METRICS:")
    print(f"  Average Code BLEU:           {stats['avg_code_bleu']:.4f}")
    print(f"  Average Code BERT F1:        {stats['avg_code_bert_f1']:.4f}")

    print("\n" + "=" * 70 + "\n")


def save_summary_csv(stats, model_name, output_file, append=True):
    """
    Save summary statistics to a single-row CSV file.
    """
    fieldnames = [
        'model', 'total_tasks',
        'avg_att_to_pass*',
        'p1_cnt', 'p1_rate*',
        'comp_last_cnt', 'comp_last_rate*',
        'p_any_cnt', 'p_any_rate*',
        'avg_code_bleu', 'avg_codebert_f1'
    ]

    row = {
        'model': model_name,
        'total_tasks': stats['total_problems'],
        'avg_att_to_pass*': f"{stats['avg_att_to_pass*']:.4f}",  # avg_attempts_to_pass
        'p1_cnt': stats['p1_cnt'],  # pass@1 attempt count
        'p1_rate*': f"{stats['p1_rate*']:.2f}",  # pass@1 attempt rate
        'comp_last_cnt': stats['comp_last_cnt'],  # compilable at last attempt count
        'comp_last_rate*': f"{stats['comp_last_rate*']:.2f}",  # compilable at last attempt rate
        'p_any_cnt': stats['p_any_cnt'],  # at least one pass count
        'p_any_rate*': f"{stats['p_any_rate*']:.2f}",  # at least one pass rate
        'avg_code_bleu': f"{stats['avg_code_bleu']:.4f}",
        'avg_codebert_f1': f"{stats['avg_code_bert_f1']:.4f}",
    }

    file_exists = os.path.exists(output_file)
    write_header = not file_exists or not append

    mode = 'a' if append else 'w'
    with open(output_file, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    """Main entry point for the report generation script."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate comprehensive reports for repair round test results'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Path to input JSONL file containing test results'
    )
    parser.add_argument(
        '--output_dir',
        default=str(config.REPAIRS_REPORTS_DIR),
        help='Output directory for generated reports (default: config.REPAIRS_REPORTS_DIR)'
    )

    args = parser.parse_args()

    # Create output directory
    config.ensure_directories()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get input file basename
    input_name = Path(args.input).stem

    # Load results
    print(f"Loading results from: {args.input}")
    results = load_results(args.input)
    print(f"Successfully loaded {len(results)} records\n")

    # Generate CSV report
    csv_file = output_dir / f"report_{input_name}.csv"
    print(f"Generating CSV report: {csv_file}")
    metrics_list = generate_csv_report(results, csv_file)

    # Generate summary statistics
    stats = generate_summary_stats(results, metrics_list)

    model_name = results[0].get('model_name', 'unknown') if results else 'unknown'

    # Save statistics to JSON
    stats_file = output_dir / f"summary_{input_name}.json"
    print(f"Saving summary statistics: {stats_file}")
    save_summary_stats(stats, stats_file)

    # Save summary CSV
    summary_csv_file = output_dir / "summary_repairs.csv"
    print(f"Saving summary CSV: {summary_csv_file}")
    save_summary_csv(stats, model_name, summary_csv_file, append=True)

    # Print summary to console
    print_summary_stats(stats)

    # Print completion message
    print(f"✓ All reports generated successfully in: {output_dir}")


if __name__ == '__main__':
    main()