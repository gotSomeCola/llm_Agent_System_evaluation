"""
Generate results report in CSV and visualizations from results_full_metrics.jsonl
"""
import json
import os
import argparse
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Set Chinese font support
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def get_model_name_from_results(results):
    """Extract model name from results"""
    if results and len(results) > 0:
        return results[0].get('model_name', 'unknown_model')
    return 'unknown_model'

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

def create_results_folder(base_path='./results'):
    """Create results folder if it doesn't exist"""
    Path(base_path).mkdir(exist_ok=True)
    return base_path

def generate_csv_report(results, output_file):
    """Generate CSV report"""
    df = pd.DataFrame([
        {
            'ID': r.get('id', 'N/A'),
            'Title': r.get('title', 'N/A'),
            'Compilable': 'Yes' if r.get('compilable', False) else 'No',
            'Pass': 'Yes' if r.get('pass', False) else 'No',
            'Total Duration (s)': r.get('total_duration', 0),
            'Runtime (ms)': r.get('runtime_ms', 0),
            'Code BLEU': round(r.get('code_bleu', 0), 4),
            'CodeBERT F1': round(r.get('code_bert_f1', 0), 4),
            'Error': r.get('error', '')
        }
        for r in results
    ])
    
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✓ CSV report saved: {output_file}")
    return df

def generate_summary_stats(results, output_file, model_name):
    """Generate summary statistics for large-scale tests"""
    total = len(results)
    compilable = sum(1 for r in results if r.get('compilable', False))
    passed = sum(1 for r in results if r.get('pass', False))
    
    # Calculate percentages
    compile_rate = (compilable / total * 100) if total > 0 else 0
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    # Calculate averages (only for compilable code)
    compilable_results = [r for r in results if r.get('compilable', False)]
    avg_duration = sum(r.get('total_duration', 0) for r in results) / total if total > 0 else 0
    avg_runtime = sum(r.get('runtime_ms', 0) for r in compilable_results) / len(compilable_results) if compilable_results else 0
    avg_code_bleu = sum(r.get('code_bleu', 0) for r in results) / total if total > 0 else 0
    avg_codebert = sum(r.get('code_bert_f1', 0) for r in results) / total if total > 0 else 0
    
    summary = f"""
=== EVALUATION SUMMARY ===
Model: {model_name}
Total Tasks: {total}

Success Rates:
- Compile Rate: {compile_rate:.2f}% ({compilable}/{total})
- Pass Rate: {pass_rate:.2f}% ({passed}/{total})

Average Metrics:
- Total Duration: {avg_duration:.2f}s
- Runtime (compilable only): {avg_runtime:.2f}ms
- Code BLEU: {avg_code_bleu:.4f}
- CodeBERT F1: {avg_codebert:.4f}

Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
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
        'avg_duration': avg_duration,
        'avg_runtime': avg_runtime,
        'avg_code_bleu': avg_code_bleu,
        'avg_codebert': avg_codebert
    }

def save_summary_csv(stats, timestamp, output_file='results/summary_all.csv'):
    """Save summary statistics to CSV (append mode for comparing multiple runs)"""
    import csv
    import os

    file_exists = os.path.exists(output_file)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

    with open(output_file, 'a', newline='', encoding='utf-8') as f:
        fieldnames = [
            'timestamp', 'model', 'total_tasks',
            'compilable_count', 'compile_rate_%', 'pass_count', 'pass_rate_%',
            'avg_duration_s', 'avg_runtime_ms', 'avg_code_bleu', 'avg_codebert_f1'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        row = {
            'timestamp': timestamp,
            'model': stats['model'],
            'total_tasks': stats['total'],
            'compilable_count': stats['compilable'],
            'compile_rate_%': f"{stats['compile_rate']:.2f}",
            'pass_count': stats['passed'],
            'pass_rate_%': f"{stats['pass_rate']:.2f}",
            'avg_duration_s': f"{stats['avg_duration']:.2f}",
            'avg_runtime_ms': f"{stats['avg_runtime']:.2f}",
            'avg_code_bleu': f"{stats['avg_code_bleu']:.4f}",
            'avg_codebert_f1': f"{stats['avg_codebert']:.4f}"
        }
        writer.writerow(row)

    print(f"✓ Summary CSV saved: {output_file}")

def generate_visualizations(results, output_dir):
    """Generate simple pie chart for pass rate (suitable for large-scale tests)"""
    
    total = len(results)
    passed = sum(1 for r in results if r.get('pass', False))
    failed_compilable = sum(1 for r in results if r.get('pass', False) is False and r.get('compilable', False))
    not_compilable = sum(1 for r in results if r.get('compilable', False) is False)
    
    # Single pie chart for results
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sizes = [passed, failed_compilable, not_compilable]
    labels = [f'Passed\n{passed} ({100*passed/total:.1f}%)', 
              f'Failed (Compilable)\n{failed_compilable} ({100*failed_compilable/total:.1f}%)', 
              f'Not Compilable\n{not_compilable} ({100*not_compilable/total:.1f}%)']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    explode = (0.05, 0, 0)
    
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
            startangle=90, explode=explode, textprops={'fontsize': 11, 'weight': 'bold'})
    ax.set_title(f'Test Results Distribution (n={total})', fontsize=14, weight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'results_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Results distribution chart saved: {output_dir}/results_distribution.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate results report from evaluation results')
    parser.add_argument('--input', type=str, default='results_full_metrics.jsonl',
                       help='Input JSONL file')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for reports')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Model name for the report (auto-detected from agents.py if not provided)')
    
    args = parser.parse_args()
    
    # Load results
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        return
    
    results = load_results(args.input)
    print(f"Loaded {len(results)} results")
    
    # Auto-detect model name from results
    model_name_from_results = get_model_name_from_results(results)
    if args.model_name is None:
        args.model_name = model_name_from_results
    print(f"Model: {args.model_name}")
    print(f"Total Tasks: {len(results)}")
    
    # Create output directory
    output_dir = create_results_folder(args.output_dir)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    task_count = len(results)
    
    # Sanitize model name for filename (replace : with -)
    safe_model_name = args.model_name.replace(':', '-').replace('/', '-')
    
    # Generate reports
    csv_file = os.path.join(output_dir, f'{safe_model_name}_tasks{task_count}_{timestamp}.csv')
    stats_file = os.path.join(output_dir, f'{safe_model_name}_tasks{task_count}_{timestamp}_summary.txt')
    
    # Generate CSV
    df = generate_csv_report(results, csv_file)
    
    # Generate summary
    stats = generate_summary_stats(results, stats_file, args.model_name)
    
    # Generate visualizations
    generate_visualizations(results, output_dir)
    
    print(f"\n✓ All reports generated in: {output_dir}/")
    
    # Save summary to CSV (for comparing multiple runs)
    summary_csv_file = os.path.join(output_dir, 'summary_all.csv')
    save_summary_csv(stats, timestamp.replace('_', ' '), summary_csv_file)

if __name__ == '__main__':
    main()
