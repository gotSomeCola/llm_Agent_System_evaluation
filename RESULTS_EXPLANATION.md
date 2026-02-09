# Results Explanation

## Pass_at_k Report (scripts/reports/generate_pass_at_k_report.py)

### Single-task CSV metrics
- Pass_At_1_Attempt: 1 if attempts[0].pass is true, else 0.
- Num_Passed: number of attempts with pass == true.
- Compilable_Count / Compilable_Rate: 1/0 based on attempts[0].compilable.
- First_Pass_Attempt: attempt index (0-based) of the first pass; N/A if none.
- Avg_Code_BLEU / Avg_CodeBERT_F1: metrics from the first successful attempt only.
- First_Pass_Code_BLEU / First_Pass_CodeBERT_F1: same as above, explicitly labeled.

### Summary metrics (JSONL / CSV)
- avg_att_to_pass*: per task, use first pass attempt index + 1. If no pass, use len(attempts). Then average across tasks.
- p1_cnt / p1_rate*: number and percent of tasks where attempts[0].pass is true.
- comp1_cnt / comp1_rate**: number and percent of tasks where attempts[0].compilable is true.
- p_any_cnt / p_any_rate*: number and percent of tasks with at least one pass.
- avg_code_bleu / avg_code_bert_f1: average over all tasks (no pass contributes 0).

## Repair-loop Report (scripts/reports/generate_repair_report.py)

### Single-task CSV metrics
- pass: final test result after the repair loop.
- pass_at_1: 1.0 if success with repair_rounds == 0, else 0.0.
- compilable: final compilability result from the last run.
- repair_rounds: number of repair iterations (initial draft not included).
- code_bleu / code_bert_f1: set only if final pass and reference exists; else 0.

### Summary metrics
- p1_cnt / p1_rate*: number and percent of tasks where pass_at_1 == 1.0.
- comp_last_cnt / comp_last_rate**: number and percent of tasks where final compilable == true.
- p_any_cnt / p_any_rate*: number and percent of tasks where final pass == true.
- avg_att_to_pass*: per task, use min(repair_rounds + 1, 5). Then average across tasks.
- avg_code_bleu / avg_codebert_f1: average over all tasks (failed tasks contribute 0).

## How to compare pass_at_k vs repair-loop
- Align attempts: pass_at_k uses external attempts; repair-loop uses internal attempts (repair_rounds + 1).
- Best cross-system metrics: pass_at_1_attempt_rate, at_least_1_pass_rate, avg_attempts_to_pass.
- Note on compilable: pass_at_k uses attempt 1; repair-loop uses final attempt.
- Note on BLEU/BERT: pass_at_k uses first successful attempt per task and averages over all tasks (failures are 0); repair-loop averages over all tasks (failures are 0).
