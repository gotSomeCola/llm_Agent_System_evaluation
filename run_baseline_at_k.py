import argparse
import json
import os
import shutil
import time
import metrics
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import modules
from agents import get_llm, implementer_prompt, LLM_MODEL_NAME
from utils import create_solution_file
from tools import run_mvn_test
from langchain_core.output_parsers import StrOutputParser


# --- CONFIGURATION ---
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
    """Creates an ISOLATED environment for a single task/attempt."""
    path = os.path.join(PROJECT_BASE_DIR, f"proj_{task_id}")
    src = os.path.join(path, "src/main/java/referenz")
    test = os.path.join(path, "src/test/java/referenz")
    
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except OSError:
            pass 
    
    os.makedirs(src, exist_ok=True)
    os.makedirs(test, exist_ok=True)
    
    pom_dest = os.path.join(path, "pom.xml")
    if not os.path.exists(pom_dest):
        if os.path.exists("./pom.xml"):
            shutil.copy("./pom.xml", pom_dest)
        else:
            print(f"ACHTUNG: Keine pom.xml gefunden! Test für {task_id} wird scheitern.")
        
    return path, src, test

def cleanup_project_env(path):
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except Exception as e:
            print(f"Warning: Could not delete {path}: {e}")

def extract_problem_data(problem):
    raw_id = problem.get('task_id', 'unknown')
    task_id = str(raw_id).strip('/').split('/')[-1]
    
    def get_val(key): return problem.get(key) or problem.get('prompt', {}).get(key)
    
    return {
        "task_id": task_id,
        "description": get_val('description') or "No Desc",
        "title": problem.get('prompt', {}).get('problem', task_id),
        "rahmen_code": get_val('rahmen_code'),
        "referenz_code": get_val('referenz_code'),
        "test_content": get_val('test_content') or problem.get('test_code')
    }

# --- PASS@K CALCULATION FUNCTION ---
def calculate_pass_at_k(n, c, k):
    """
    Calculates pass@k metric using the estimator from the Codex paper.
    pass@k = 1 - E[ (n-c choose k) / (n choose k) ]
    
    n: total attempts (samples)
    c: correct attempts
    k: k-metric (e.g., 1, 10)
    """
    if n < k:
        return 0.0 # Not enough samples to estimate pass@k
    
    if n - c < k:
        return 1.0

    # Numerically stable calculation: 1 - product((n-c-i)/(n-i))
    prob_all_fail = 1.0
    for i in range(k):
        prob_all_fail *= (n - c - i) / (n - i)
    
    return 1.0 - prob_all_fail

def evaluate_single_attempt(problem_data, model_chain, model_name, attempt_num=0, temperature=0.0):
    """Evaluates a single attempt."""
    task_id = problem_data["task_id"]
    
    # Unique directory for this attempt
    proj_dir, src_dir, test_dir = setup_project_env(f"{task_id}_attempt{attempt_num}")
    
    input_data = {
        "task_description": problem_data["description"],
        "rahmen_code": problem_data["rahmen_code"],
        "plan": "None" 
    }
    
    start_time = time.time()
    try:
        llm_response = model_chain.invoke(input_data)
        final_code = create_solution_file("// unused", llm_response)
        
        with open(os.path.join(src_dir, "Solution.java"), "w", encoding="utf-8") as f:
            f.write(final_code)
        
        test_content = problem_data["test_content"]
        if "package referenz" not in test_content:
            test_content = "package referenz;\n" + test_content
        with open(os.path.join(test_dir, "SolutionTest.java"), "w", encoding="utf-8") as f:
            f.write(test_content)
            
        return_code, logs = run_mvn_test(proj_dir)
        total_duration = time.time() - start_time
        
        eval_result = metrics.evaluate_test_results(return_code, logs)
        runtime_ms = metrics.extract_runtime_from_logs(logs) * 1000
        
        # Calculate heavy metrics only if passed (saves time) or always if you prefer
        code_bleu_res = {"codebleu": 0.0}
        c_bert_f1 = 0.0
        
        # Optional: Only calc metrics if passed to speed up pass@k
        if problem_data["referenz_code"] and final_code and eval_result["pass"]:
            code_bleu_res = metrics.calculate_code_bleu(problem_data["referenz_code"], final_code)
            _, _, c_bert_f1 = metrics.evaluate_code_with_codeBert_score(problem_data["referenz_code"], final_code)
        
        return {
            "attempt": attempt_num,
            "temperature": temperature,
            "pass": eval_result["pass"],
            "compilable": eval_result["compilable"],
            "total_duration": round(total_duration, 2),
            "runtime_ms": runtime_ms,
            "code_bleu": round(code_bleu_res.get('codebleu', 0.0), 4),
            "code_bert_f1": round(c_bert_f1, 4),
            "generated_code": final_code,
            "logs": logs[:300] if eval_result["pass"] else logs[:800] # Save space
        }
        
    except Exception as e:
        return { "attempt": attempt_num, "pass": False, "compilable": False, "error": str(e) }
    finally:
        cleanup_project_env(proj_dir)

def evaluate_single_problem(problem, model_chain, model_name, n_samples=1):
    """
    Runs n_samples attempts and calculates the unbiased Pass@k metrics.
    """
    problem_data = extract_problem_data(problem)
    
    attempts = []
    
    for i in range(n_samples):
        # Temp strategy: 0.0 for first (greedy), 0.8 for rest (diverse)
        # If n_samples is 1, strictly use 0.0
        if n_samples == 1:
            temp = 0.0
        else:
            temp = 0.0 if i == 0 else 0.8
        
        # Create a fresh LLM instance with the correct temperature
        llm = get_llm(model_name=model_name, temperature=temp)
        chain = implementer_prompt | llm | StrOutputParser()
        
        res = evaluate_single_attempt(problem_data, chain, model_name, i, temp)
        attempts.append(res)
    
    # --- STATISTICS ---
    # c: number of correct attempts
    c = sum(1 for a in attempts if a.get("pass"))
    n = n_samples
    
    # Calculate Pass@k metrics
    # Pass@1: Probability of finding a solution in 1 try
    pass_1 = calculate_pass_at_k(n, c, 1)
    
    # Pass@k: Probability of finding a solution in k tries (where k=n)
    pass_k = calculate_pass_at_k(n, c, n)
    
    # Pass@5: If we generated enough samples (>=5), calculate Pass@5
    pass_5 = calculate_pass_at_k(n, c, 5) if n >= 5 else 0.0

    # First attempt result (temperature 0.0)
    pass_first = attempts[0].get("pass") if attempts else False

    return {
        "model_name": model_name,
        "id": problem_data["task_id"],
        "title": problem_data["title"],
        "n_samples": n,
        "c_correct": c,
        # Metrics
        "pass@1_first": pass_first,
        "pass@1": pass_1,
        "pass@5": pass_5,
        "pass@k": pass_k, # This is Pass@n_samples
        # Details
        "attempts": attempts
    }

def main_runner(model_name, output_file, min_count, max_count, concurrency=1, k=1):
    full_data = load_leetcode_dataset()
    end_index = len(full_data) if max_count <= 0 else max_count
    start_index = max(0, min_count)
    end_index = min(len(full_data), end_index)
    data_batch = full_data[start_index:end_index]
    
    print(f"--- Starting Pass@k Evaluation: {model_name} ---")
    print(f"Tasks: {len(data_batch)} | Samples per Task (n): {k}")
    
    if len(data_batch) == 0: return

    # Dummy LLM just to pass into the function (will be overridden inside loop)
    llm = get_llm(temperature=0.0)
    chain = implementer_prompt | llm | StrOutputParser()
    
    stats = {
        "pass@1_first": 0,
        "pass@1": 0.0,
        "pass@5": 0.0,
        "pass@k": 0.0
    }
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_problem = {
            # We pass 'k' as the number of samples to generate (n_samples)
            executor.submit(evaluate_single_problem, p, chain, model_name, k): p 
            for p in data_batch
        }
        
        for future in tqdm(as_completed(future_to_problem), total=len(data_batch), desc="Eval"):
            res = future.result()
            
            # Aggregate Probabilities (Sum them up)
            if res.get("pass@1_first"): stats["pass@1_first"] += 1
            stats["pass@1"] += res.get("pass@1", 0.0)
            stats["pass@5"] += res.get("pass@5", 0.0)
            stats["pass@k"] += res.get("pass@k", 0.0)
            
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(res) + "\n")

    total = len(data_batch)
    if total == 0:
        print("No tasks evaluated.")
        return
        
    print(f"\nEvaluation Finished!")
    print(f"Pass@1 (First):    {stats['pass@1_first']/total*100:.2f}%")
    print(f"Pass@1:            {stats['pass@1']/total*100:.2f}%")
    if k >= 5:
        print(f"Pass@5:            {stats['pass@5']/total*100:.2f}%")
    print(f"Pass@{k}:            {stats['pass@k']/total*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=LLM_MODEL_NAME)
    parser.add_argument('--output_file', type=str, default="results_pass_k.jsonl")
    parser.add_argument('--min_count', type=int, default=0)
    parser.add_argument('--max_count', type=int, default=5)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--k', type=int, default=1, help='Samples per task (n) to generate')
    
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