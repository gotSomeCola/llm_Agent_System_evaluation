# 📁 Project Directory Structure Guide

## Overview

This project evaluates multiple LLMs for Java code generation. The directory layout is split into two main areas: **evaluation data** (raw JSONL) and **reports** (formatted outputs).

## Directory Tree

```
results/
├── evaluations/                    # Raw evaluation data (JSONL)
│   ├── baseline/                   # Pass@1 baseline
│   │   ├── results_gemma3_27b_20260201_120000.jsonl
│   │   ├── results_gpt-oss:20b_20260201_120100.jsonl
│   │   └── batch_summary_20260201_120000.json
│   │
│   ├── repairs/                    # Code repair evaluation (multi-iteration)
│   │   ├── results_gemma3_27b_20260201_130000.jsonl
│   │   ├── results_llama3.3_70b_20260201_130100.jsonl
│   │   └── batch_summary_20260201_130000.json
│   │
│   └── pass_at_k/                  # Pass@K evaluation (multiple attempts)
│       ├── results_gemma3_27b_20260201_140000.jsonl
│       ├── results_gpt-oss:120b_20260201_140100.jsonl
│       └── batch_summary_20260201_140000.json
│
└── reports/                        # Formatted analysis reports
  ├── baseline/                   # Pass@1 reports
    │   ├── gemma3-27b_tasks500_20260201_120000.csv
    │   ├── gemma3-27b_tasks500_20260201_120000_summary.txt
    │   ├── gpt-oss-20b_tasks500_20260201_120100.csv
    │   ├── gpt-oss-20b_tasks500_20260201_120100_summary.txt
    │   └── summary_pilot.csv                      # Cross-model comparison
    │
    ├── repairs/                    # Code repair reports
    │   ├── gemma3-27b_tasks500_20260201_130000.csv
    │   ├── gemma3-27b_tasks500_20260201_130000_summary.txt
    │   └── summary_repair.csv
    │
    └── pass_at_k/                  # Pass@K reports
        ├── gemma3-27b_tasks500_20260201_140000.csv
        ├── gemma3-27b_tasks500_20260201_140000_summary.txt
        └── summary_pass_at_k.csv
```

## File Format Notes

### Evaluation Data (JSONL)

每行一条 JSON 记录：
```json
{
  "id": "1234",
  "model_name": "gemma3:27b",
  "pass": true,
  "compilable": true,
  "pass_at_1": 1.0,
  "pass_at_k": 1.0,
  "code_bleu": 0.8234,
  "code_bert_f1": 0.7123,
  "repair_rounds": 2,
  "history": ["Initial Draft", "Repair 1", "Test PASSED"]
}
```

### Report Files (CSV)

```
ID,Title,Compilable,Pass,Code BLEU,CodeBERT F1
1234,Two Sum,Yes,Yes,0.8234,0.7123
5678,LRU Cache,No,No,0.0,0.0
```

### Summary (TXT)

```
=== EVALUATION SUMMARY ===
Model: gemma3:27b
Total Tasks: 500

Success Rates:
- Compile Rate: 95.20% (476/500)
- Pass Rate: 82.40% (412/500)

Average Metrics:
- Code BLEU: 0.7845
- CodeBERT F1: 0.6923
```

## Script-to-Directory Mapping

| Script | Evaluation Directory | Report Directory | Output Files |
|------|---------|---------|---------|
| run_baseline.py | evaluations/baseline/ | reports/baseline/ | results_*.jsonl |
| run_baseline_at_k.py | evaluations/pass_at_k/ | reports/pass_at_k/ | results_*.jsonl |
| batch_coder_tester.py | evaluations/repairs/ | reports/repairs/ | results_*.jsonl |
| batch_test_models.py | evaluations/pass_at_k/ | reports/pass_at_k/ | results_*.jsonl |
| generate_pass_at_1_report.py | read | reports/baseline/ | *.csv, *_summary.txt |
| generate_pass_at_k_report.py | read | reports/pass_at_k/ | *.csv, *_summary.txt |

## Using the Configuration File

All directory constants are centralized in config.py:

```python
from config import (
    BASELINE_EVAL_DIR,      # 基准评估输出
    REPAIRS_EVAL_DIR,       # 修复评估输出
    PASS_AT_K_EVAL_DIR,     # Pass@K 评估输出
    BASELINE_REPORTS_DIR,   # 基准报告输出
    REPAIRS_REPORTS_DIR,    # 修复报告输出
    PASS_AT_K_REPORTS_DIR,  # Pass@K 报告输出
)

# Use config
output_dir = BASELINE_EVAL_DIR
```

## Cleanup and Maintenance

### Clean Temporary Projects

```bash
python -c "from config import cleanup_temp_projects; cleanup_temp_projects()"
```

### Initialize Directory Structure

```bash
python -c "from config import ensure_directories; ensure_directories()"
```

### View Structure

```bash
python config.py
```

## Naming Conventions

### Filename Patterns

**Evaluation JSONL:**
```
results_{model}_{timestamp}.jsonl
示例: results_gemma3_27b_20260201_120000.jsonl
      results_gpt-oss_120b_20260201_130000.jsonl
```

**Report CSV:**
```
{safe_model}_tasks{count}_{timestamp}.csv
示例: gemma3-27b_tasks500_20260201_120000.csv
```

**Summary TXT:**
```
{safe_model}_tasks{count}_{timestamp}_summary.txt
```

**Batch Summary:**
```
batch_summary_{timestamp}.json
```

### Model Name Mapping

| Original Model Name | Filename-safe Version |
|---------|---------------|
| `gemma3:27b` | `gemma3-27b` |
| `gpt-oss:120b` | `gpt-oss-120b` |
| `llama3.3:70b` | `llama3.3-70b` |

## Best Practices

✅ **Do:**
- Use constants from config.py
- Call ensure_directories() at the start of scripts
- Add timestamps where appropriate
- Use clear model name mappings

❌ **Don't:**
- Hardcode path strings
- Create unnecessary directories
- Mix evaluation data and report directories
- Use unclear filenames

