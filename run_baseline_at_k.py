# run_baseline.py
import argparse
import json
import os
import shutil # Zum Löschen von Ordnern
import time
import metrics
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 禁用 tokenizers 并行处理，避免 fork 警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Importiere unsere Module
from agents import get_llm, implementer_prompt, LLM_MODEL_NAME
from utils import create_solution_file
from tools import run_mvn_test
from langchain_core.output_parsers import StrOutputParser
 

# --- KONFIGURATION ---
PROJECT_BASE_DIR = "./project"
DATASET_PATH = './leetcode_dataset.json'

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_leetcode_dataset():
    data = []
    if not os.path.exists(DATASET_PATH):
        print(f"Warnung: {DATASET_PATH} nicht gefunden.")
        return []
        
    with open(DATASET_PATH, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data

def setup_project_env(task_id):
    """
    Erstellt eine ISOLIERTE Umgebung für genau einen Task.
    Pfad: ./project/proj_{task_id}
    
    Dies ermöglicht echtes Multithreading ohne Dateikonkurrenz.
    """
    path = os.path.join(PROJECT_BASE_DIR, f"proj_{task_id}")
    src = os.path.join(path, "src/main/java/referenz")
    test = os.path.join(path, "src/test/java/referenz")
    
    # Sicherheits-Check: Falls der Ordner noch existiert (z.B. alter Absturz), löschen
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except OSError:
            pass # Ignorieren, falls gerade gesperrt
    
    # Ordnerstruktur erstellen
    os.makedirs(src, exist_ok=True)
    os.makedirs(test, exist_ok=True)
    
    # pom.xml kopieren (Zwingend notwendig für Maven)
    pom_dest = os.path.join(path, "pom.xml")
    if not os.path.exists(pom_dest):
        if os.path.exists("./pom.xml"):
            shutil.copy("./pom.xml", pom_dest)
        else:
            print(f"ACHTUNG: Keine pom.xml gefunden! Test für {task_id} wird scheitern.")
        
    return path, src, test

def cleanup_project_env(path):
    """
    Löscht den temporären Projektordner komplett, um Speicherplatz freizugeben.
    
    Dies verhindert, dass sich alte Test-Daten bei vielen parallelen Ausführungen aufbauen.
    """
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except Exception as e:
            print(f"Warning: Could not delete {path}: {e}")

def evaluate_single_attempt(problem, model_chain, model_name, task_id, attempt_num=0, temperature=0.0):
    """
    单次尝试的评估函数（用于 pass@k）
    """
    def get_val(key): return problem.get(key) or problem.get('prompt', {}).get(key)
    description = get_val('description') or "No Desc"
    title = problem.get('prompt', {}).get('problem', task_id)
    rahmen_code = get_val('rahmen_code') 
    referenz_code = get_val('referenz_code')
    test_content = get_val('test_content') or problem.get('test_code')

    # Setup Environment (每次尝试使用独立的目录)
    proj_dir, src_dir, test_dir = setup_project_env(f"{task_id}_attempt{attempt_num}")
    
    input_data = {
        "task_description": description,
        "rahmen_code": rahmen_code,
        "plan": "None" 
    }
    
    start_time = time.time()
    try:
        # A. 使用指定温度生成代码
        llm_response = model_chain.invoke(input_data)
        final_code = create_solution_file("// unused", llm_response)
        
        with open(os.path.join(src_dir, "Solution.java"), "w", encoding="utf-8") as f:
            f.write(final_code)
        if "package referenz" not in test_content:
            test_content = "package referenz\n" + test_content
        with open(os.path.join(test_dir, "SolutionTest.java"), "w", encoding="utf-8") as f:
            f.write(test_content)
            
        # B. Run Test
        return_code, logs = run_mvn_test(proj_dir)
        total_duration = time.time() - start_time
        
        # C. Metrics
        eval_result = metrics.evaluate_test_results(return_code, logs)
        runtime_ms = metrics.extract_runtime_from_logs(logs) * 1000
        
        code_bleu_res = {"codebleu": 0.0}
        c_bert_f1 = 0.0
        if referenz_code and final_code:
            code_bleu_res = metrics.calculate_code_bleu(referenz_code, final_code)
            _, _, c_bert_f1 = metrics.evaluate_code_with_codeBert_score(referenz_code, final_code)
        
        return {
            "attempt": attempt_num,
            "temperature": temperature,
            "compilable": eval_result["compilable"],
            "pass": eval_result["pass"],
            "total_duration": round(total_duration, 2),
            "runtime_ms": runtime_ms,
            "code_bleu": round(code_bleu_res.get('codebleu', 0.0), 4),
            "code_bert_f1": round(c_bert_f1, 4),
            "generated_code": final_code,
            "logs": logs[:500] if eval_result["pass"] else logs[:2000]
        }
        
    except Exception as e:
        return {
            "attempt": attempt_num,
            "temperature": temperature,
            "compilable": False,
            "pass": False,
            "error": str(e)
        }
    finally:
        cleanup_project_env(proj_dir)

def evaluate_single_problem(problem, model_chain, model_name, k=1):
    """
    Pass@k 评估：对同一问题尝试 k 次生成
    """
    # 1. ID & Data Extraction
    raw_id = problem.get('task_id', 'unknown')
    task_id = str(raw_id).strip('/').split('/')[-1]
    
    def get_val(key): return problem.get(key) or problem.get('prompt', {}).get(key)
    description = get_val('description') or "No Desc"
    title = problem.get('prompt', {}).get('problem', task_id)
    rahmen_code = get_val('rahmen_code') 
    referenz_code = get_val('referenz_code')
    test_content = get_val('test_content') or problem.get('test_code')
    
    # 2. 进行 k 次尝试
    attempts = []
    first_success_idx = -1
    
    for attempt_num in range(k):
        # 温度策略：第一次用 0.0（确定性），后续增加随机性
        temp = 0.0 if attempt_num == 0 else min(0.9, 0.5 + 0.1 * attempt_num)
        
        # 为每次尝试创建新的 LLM（使用不同温度）
        llm = get_llm(model_name=model_name, temperature=temp)
        chain = implementer_prompt | llm | StrOutputParser()
        
        result = evaluate_single_attempt(
            problem, chain, model_name, task_id, attempt_num, temp
        )
        attempts.append(result)
        
        # 记录第一次成功的索引
        if first_success_idx == -1 and result.get("pass"):
            first_success_idx = attempt_num
    
    # 3. 计算 pass@k 和 pass@1
    pass_at_k = any(a.get("pass") for a in attempts)
    pass_at_1 = attempts[0].get("pass") if attempts else False
    
    # 4. 构建最终结果
    result_payload = {
        "model_name": model_name,
        "id": task_id,
        "title": title,
        "k": k,
        "pass@1": pass_at_1,
        "pass@k": pass_at_k,
        "first_success_idx": first_success_idx,
        "total_attempts": k,
        "attempts": attempts
    }
    
    return result_payload

def evaluate_single_problem_legacy(problem, model_chain, model_name):
    """
    原始的 pass@1 评估函数（保留用于向后兼容）
    """
    # 1. ID & Data Extraction (Same as before)
    raw_id = problem.get('task_id', 'unknown')
    task_id = str(raw_id).strip('/').split('/')[-1]
    
    def get_val(key): return problem.get(key) or problem.get('prompt', {}).get(key)
    description = get_val('description') or "No Desc"
    title = problem.get('prompt', {}).get('problem', task_id)
    rahmen_code = get_val('rahmen_code') 
    referenz_code = get_val('referenz_code')
    test_content = get_val('test_content') or problem.get('test_code')

    # 2. Setup Environment
    proj_dir, src_dir, test_dir = setup_project_env(task_id)
    
    input_data = {
        "task_description": description,
        "rahmen_code": rahmen_code,
        "plan": "None" 
    }
    
    start_time = time.time()
    try:
        # A. Generate & Save
        llm_response = model_chain.invoke(input_data)
        final_code = create_solution_file("// unused", llm_response)
        
        with open(os.path.join(src_dir, "Solution.java"), "w", encoding="utf-8") as f:
            f.write(final_code)
        if "package referenz" not in test_content:
            test_content = "package referenz;\n" + test_content
        with open(os.path.join(test_dir, "SolutionTest.java"), "w", encoding="utf-8") as f:
            f.write(test_content)
            
        # B. Run Test (Get RAW results)
        return_code, logs = run_mvn_test(proj_dir)
        total_duration = time.time() - start_time
        
        # C. METRICS EVALUATION (Centralized in metrics.py)
        # 1. Determine Status (Compilable? Pass?)
        eval_result = metrics.evaluate_test_results(return_code, logs)
        
        # 2. Extract Runtime
        runtime_ms = metrics.extract_runtime_from_logs(logs) * 1000
        
        # 3. Code Similarity Metrics
        code_bleu_res = {"codebleu": 0.0}
        c_bert_f1 = 0.0
        
        if referenz_code and final_code:
            code_bleu_res = metrics.calculate_code_bleu(referenz_code, final_code)
            _, _, c_bert_f1 = metrics.evaluate_code_with_codeBert_score(referenz_code, final_code)
        
        # D. Build Payload
        result_payload = {
            "model_name": model_name,
            "id": task_id,
            "title": title,
            # Core Metrics
            "compilable": eval_result["compilable"],
            "pass": eval_result["pass"],
            "repair_rounds": 0,
            # Performance Metrics
            "total_duration": round(total_duration, 2),
            "runtime_ms": runtime_ms,
            # Similarity Metrics
            "code_bleu": round(code_bleu_res.get('codebleu', 0.0), 4),
            "code_bert_f1": round(c_bert_f1, 4),
            "generated_code": final_code,
            "logs": logs[:500] if eval_result["pass"] else logs[:2000]
        }

        return result_payload
        
    except Exception as e:
        return {
            "model_name": model_name,
            "id": task_id,
            "title": title,
            "compilable": False,
            "pass": False,
            "error": str(e)
        }
        
    finally:
        # F. CLEANUP (WICHTIG!)
        # Löscht den Ordner sofort nach Abschluss dieses Tasks.
        # Das erlaubt Multithreading ohne Speicher-Überlauf.
        cleanup_project_env(proj_dir)

def main_runner(model_name, output_file, min_count, max_count, concurrency=1, k=1):
    full_data = load_leetcode_dataset()
    end_index = len(full_data) if max_count <= 0 else max_count
    start_index = max(0, min_count)
    end_index = min(len(full_data), end_index)
    data_batch = full_data[start_index:end_index]
    
    print(f"--- Starte Pass@{k} Evaluation: {model_name} ---")
    print(f"Worker Threads: {concurrency}")
    print(f"Aufgaben Bereich: {start_index} bis {end_index} ({len(data_batch)} Tasks)")
    print(f"Attempts per Task: {k}")
    
    if len(data_batch) == 0:
        print("Keine Aufgaben im gewählten Bereich.")
        return

    # 注意：对于 pass@k，每次尝试会使用不同的温度，在 evaluate_single_problem 中处理
    llm = get_llm(model_name=model_name, temperature=0.0)  # 默认 LLM（会在函数内部被覆盖）
    chain = implementer_prompt | llm | StrOutputParser()
    
    success_count = 0
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_problem = {
            executor.submit(evaluate_single_problem, p, chain, model_name, k): p 
            for p in data_batch
        }
        
        for future in tqdm(as_completed(future_to_problem), total=len(data_batch), desc="Pass@k Eval"):
            res = future.result()
            # 对于 pass@k，统计 pass@k 的成功率
            if res.get("pass@k"):
                success_count += 1
                
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(res) + "\n")

    pass_at_k_rate = (success_count / len(data_batch)) * 100
    print(f"\nEvaluation Abgeschlossen!")
    print(f"Final Pass@{k} Rate: {pass_at_k_rate:.2f}% ({success_count}/{len(data_batch)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=LLM_MODEL_NAME,
                       help=f'Model name (default: {LLM_MODEL_NAME})')
    parser.add_argument('--output_file', type=str, default="results_full_metrics.jsonl")
    parser.add_argument('--min_count', type=int, default=0)
    parser.add_argument('--max_count', type=int, default=5)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--k', type=int, default=1, help='Number of attempts per task (pass@k)')
    
    args = parser.parse_args()
    ensure_directory_exists("./output")
    
    main_runner(
        model_name=args.model_name, 
        output_file=args.output_file, 
        min_count=args.min_count, 
        max_count=args.max_count, 
        concurrency=args.workers,
        k=args.k
    )