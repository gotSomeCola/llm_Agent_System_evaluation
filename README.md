# LLM Agent System Evaluation

This project evaluates multiple LLMs for Java code generation and repair. It covers:
- baseline / pass_at_k: multiple generation attempts (pass@k calculation disabled; attempts still evaluated)
- repair-loop: multi-iteration repair based on test feedback
- reporting & visualization: CSV/JSONL summaries and comparison plots

## Quick Start

### Python Dependencies
```bash
pip install -r requirements.txt
```

### Run Evaluations
- baseline / pass_at_k:
```bash
python run_baseline_at_k.py --model_name "gpt-oss:20b" --n_attempts 5
```

- repair-loop:
```bash
python coder_tester_system.py --model_name "gpt-oss:20b" --min_count 0 --max_count 5
```

### Generate Reports
- pass_at_k report:
```bash
python scripts/reports/generate_pass_at_k_report.py --input results/evaluations/pass_at_k/<file>.jsonl
```

- repair-loop report:
```bash
python scripts/reports/generate_repair_report.py --input results/evaluations/repairs/<file>.jsonl
```

- unified report (auto-detects type):
```bash
python scripts/reports/generate_evaluation_report.py --input results/evaluations/<file>.jsonl
```

### Plotting
- pass_at_k vs repair-loop:
```bash
python scripts/plots/plot_model_comparison.py
```

- pilot vs tests baseline:
```bash
python scripts/plots/plot_baseline_comparison.py
```

## Directory Layout
- Raw evaluation JSONL: [results/evaluations](results/evaluations)
- Report outputs (CSV/JSONL): [results/reports](results/reports)
- Temporary Java projects: [project](project)
- Scripts:
  - Reports: [scripts/reports](scripts/reports)
  - Plots: [scripts/plots](scripts/plots)

See [DIRECTORY_STRUCTURE.md](DIRECTORY_STRUCTURE.md) for the full layout.

## Metrics
- Results metrics: [RESULTS_EXPLANATION.md](RESULTS_EXPLANATION.md)
- Metric definitions: [METRICS_EXPLANATION.md](METRICS_EXPLANATION.md)

## Notes
- Maven is not listed in requirements.txt. Install it separately if you run Java tests.
- Pass@k calculation is disabled; attempts are still generated and used for evaluation.
