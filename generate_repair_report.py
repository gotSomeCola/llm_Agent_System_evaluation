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

import json
import csv
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime

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
    
    # Calculate basic statistics
    stats = {
        'total_problems': total_problems,
        'total_passed': sum(1 for m in metrics_list if m['pass']),
        'total_failed': sum(1 for m in metrics_list if not m['pass']),
        'compilable_count': sum(1 for m in metrics_list if m['compilable']),
        'avg_repair_rounds': sum(m['repair_rounds'] for m in metrics_list) / total_problems if total_problems > 0 else 0,
        'avg_code_bleu': sum(m['code_bleu'] for m in metrics_list) / total_problems if total_problems > 0 else 0,
        'avg_code_bert_f1': sum(m['code_bert_f1'] for m in metrics_list) / total_problems if total_problems > 0 else 0,
        'avg_pass_at_1': sum(m['pass_at_1'] for m in metrics_list) / total_problems if total_problems > 0 else 0,
        'avg_pass_at_k': sum(m['pass_at_k'] for m in metrics_list) / total_problems if total_problems > 0 else 0,
    }
    
    # Analyze repair round distribution
    issues_by_repair_rounds = defaultdict(int)
    
    for m in metrics_list:
        if m['repair_rounds'] > 0:
            issues_by_repair_rounds[m['repair_rounds']] += 1
    
    stats['issues_by_repair_rounds'] = dict(issues_by_repair_rounds)
    
    # Calculate pass rates as percentages
    stats['pass_rate'] = (stats['total_passed'] / total_problems * 100) if total_problems > 0 else 0
    stats['compilable_rate'] = (stats['compilable_count'] / total_problems * 100) if total_problems > 0 else 0
    
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
    print("\n" + "="*70)
    print("REPAIR ROUND TEST RESULTS - SUMMARY REPORT")
    print("="*70)
    
    print(f"\nBASIC STATISTICS:")
    print(f"  Total Problems:              {stats['total_problems']}")
    print(f"  Passed:                      {stats['total_passed']}")
    print(f"  Failed:                      {stats['total_failed']}")
    print(f"  Pass Rate:                   {stats['pass_rate']:.2f}%")
    
    print(f"\nCOMPILABILITY:")
    print(f"  Compilable Problems:         {stats['compilable_count']}")
    print(f"  Compilable Rate:             {stats['compilable_rate']:.2f}%")
    
    print(f"\nREPAIR METRICS:")
    print(f"  Average Repair Rounds:       {stats['avg_repair_rounds']:.2f}")
    
    if stats['issues_by_repair_rounds']:
        print(f"  Repair Round Distribution:")
        for rounds in sorted(stats['issues_by_repair_rounds'].keys()):
            count = stats['issues_by_repair_rounds'][rounds]
            print(f"    {rounds} repair rounds: {count} problems")
    
    print(f"\nCODE QUALITY METRICS:")
    print(f"  Average Code BLEU:           {stats['avg_code_bleu']:.4f}")
    print(f"  Average Code BERT F1:        {stats['avg_code_bert_f1']:.4f}")
    
    print(f"\nPASS RATE METRICS:")
    print(f"  Average Pass@1:              {stats['avg_pass_at_1']:.4f}")
    print(f"  Average Pass@K:              {stats['avg_pass_at_k']:.4f}")
    
    print("\n" + "="*70 + "\n")


def save_summary_csv(stats, model_name, output_file, timestamp, append=True):
    """
    Save summary statistics to a single-row CSV file.
    """
    fieldnames = [
        'timestamp', 'model', 'total_tasks', 'compilable_count', 'compile_rate_%',
        'pass_count', 'pass_rate_%', 'avg_repair_rounds', 'avg_code_bleu',
        'avg_codebert_f1', 'avg_pass_at_1', 'avg_pass_at_k'
    ]

    row = {
        'timestamp': timestamp,
        'model': model_name,
        'total_tasks': stats['total_problems'],
        'compilable_count': stats['compilable_count'],
        'compile_rate_%': f"{stats['compilable_rate']:.2f}",
        'pass_count': stats['total_passed'],
        'pass_rate_%': f"{stats['pass_rate']:.2f}",
        'avg_repair_rounds': f"{stats['avg_repair_rounds']:.3f}",
        'avg_code_bleu': f"{stats['avg_code_bleu']:.4f}",
        'avg_codebert_f1': f"{stats['avg_code_bert_f1']:.4f}",
        'avg_pass_at_1': f"{stats['avg_pass_at_1']:.4f}",
        'avg_pass_at_k': f"{stats['avg_pass_at_k']:.4f}",
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
    
    # Get input file basename and generate timestamp
    input_name = Path(args.input).stem
    timestamp = datetime.now().strftime("%Y%m%d %H%M%S")
    
    # Load results
    print(f"Loading results from: {args.input}")
    results = load_results(args.input)
    print(f"Successfully loaded {len(results)} records\n")
    
    # Generate CSV report
    csv_file = output_dir / f"report_{input_name}_{timestamp}.csv"
    print(f"Generating CSV report: {csv_file}")
    metrics_list = generate_csv_report(results, csv_file)
    
    # Generate summary statistics
    stats = generate_summary_stats(results, metrics_list)

    model_name = results[0].get('model_name', 'unknown') if results else 'unknown'
    
    # Save statistics to JSON
    stats_file = output_dir / f"summary_{input_name}_{timestamp.replace(' ', '_')}.json"
    print(f"Saving summary statistics: {stats_file}")
    save_summary_stats(stats, stats_file)

    # Save summary CSV
    summary_csv_file = output_dir / "summary_repairs.csv"
    print(f"Saving summary CSV: {summary_csv_file}")
    save_summary_csv(stats, model_name, summary_csv_file, timestamp, append=True)
    
    # Print summary to console
    print_summary_stats(stats)
    
    # Print completion message
    print(f"✓ All reports generated successfully in: {output_dir}")


if __name__ == '__main__':
    main()
