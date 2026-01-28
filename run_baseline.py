import argparse
import json
import os
import shutil # Zum Löschen von Ordnern
import time
import metrics
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Importiere unsere Module
from agents import get_llm, implementer_prompt
from utils import create_solution_file
from tools import run_mvn_test
from langchain_core.output_parsers import StrOutputParser
 

# --- KONFIGURATION ---
PROJECT_BASE_DIR = "./project"
DATASET_PATH = '../leetcode_dataset.json'

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
    # Pfad für DIESE Aufgabe
    path = os.path.join(PROJECT_BASE_DIR, f"proj_{task_id}")
    src = os.path.join(path, "src/main/java/referenz")
    test = os.path.join(path, "src/test/java/referenz")
    
    # Sicherstellen, dass der Ordner sauber ist (falls alter Lauf abgebrochen)
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except:
            pass # Ignorieren, falls gerade Zugriffsproblem

    os.makedirs(src, exist_ok=True)
    os.makedirs(test, exist_ok=True)
    
    # Kopiere pom.xml
    if os.path.exists("./pom.xml"):
        os.system(f"cp ./pom.xml {path}/")
    else:
        print(f"ACHTUNG: Keine pom.xml gefunden! Test für {task_id} wird scheitern.")
        
    return path, src, test

def cleanup_project_env(path):
    """Löscht den Projektordner, um Speicher zu sparen."""
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except Exception as e:
            print(f"Warning: Could not delete {path}: {e}")

def evaluate_single_problem(problem, model_chain):
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
            "id": task_id,
            "title": title,
            "compilable": False,
            "pass": False,
            "error": str(e)
        }
    finally:
        cleanup_project_env(proj_dir)

def main_runner(model_name, output_file, min_count, max_count, concurrency=1):
    full_data = load_leetcode_dataset()
    end_index = len(full_data) if max_count <= 0 else max_count
    start_index = max(0, min_count)
    end_index = min(len(full_data), end_index)
    data_batch = full_data[start_index:end_index]
    
    print(f"--- Starte Evaluation: {model_name} ---")
    print(f"Range: {start_index} bis {end_index} ({len(data_batch)} Tasks)")
    
    if len(data_batch) == 0: return

    llm = get_llm(temperature=0.0)
    chain = implementer_prompt | llm | StrOutputParser()
    
    success_count = 0
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_problem = {
            executor.submit(evaluate_single_problem, p, chain): p 
            for p in data_batch
        }
        
        for future in tqdm(as_completed(future_to_problem), total=len(data_batch), desc="Eval"):
            res = future.result()
            if res.get("pass"):
                success_count += 1
                
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(res) + "\n")

    pass_rate = (success_count / len(data_batch)) * 100
    print(f"\nFinal Pass Rate: {pass_rate:.2f}% ({success_count}/{len(data_batch)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="deepseek-r1:14b")
    parser.add_argument('--output_file', type=str, default="results_full_metrics.jsonl")
    parser.add_argument('--min_count', type=int, default=0)
    parser.add_argument('--max_count', type=int, default=5)
    parser.add_argument('--workers', type=int, default=1)
    
    args = parser.parse_args()
    ensure_directory_exists("./output")
    
    main_runner(
        model_name=args.model_name, 
        output_file=args.output_file, 
        min_count=args.min_count, 
        max_count=args.max_count, 
        concurrency=args.workers
    )