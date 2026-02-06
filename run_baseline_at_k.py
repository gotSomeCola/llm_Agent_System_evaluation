import argparse
import json
import os
import shutil
import time
import metrics
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from metrics import calculate_pass_at_k
import config
# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import modules
from agents import get_llm, implementer_gen_prompt, LLM_MODEL_NAME, retry_on_rate_limit
from utils import create_solution_file
from tools import run_mvn_test
from langchain_core.output_parsers import StrOutputParser


# --- CONFIGURATION ---
PROJECT_BASE_DIR = str(config.PROJECT_TEMP_DIR)
DATASET_PATH = str(config.LEETCODE_DATASET_PATH)

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
            print(f"Warning: pom.xml not found! Test for {task_id} will fail.")
        
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
        @retry_on_rate_limit(max_retries=10, wait_seconds=10)
        def _invoke_llm(chain, payload):
            return chain.invoke(payload)

        llm_response = _invoke_llm(model_chain, input_data)
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
        
        if problem_data["referenz_code"] and final_code:
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
            "logs": logs[:500] if eval_result["pass"] else logs[:2000] # Save space
        }
        
    except Exception as e:
        return { "attempt": attempt_num, "pass": False, "compilable": False, "error": str(e) }
    finally:
        cleanup_project_env(proj_dir)

def evaluate_single_problem(problem, model_chain, model_name, n_attempts=1, k_values=None):
    """
    Runs n_attempts times and calculates Pass@k for specified k values.
    
    Args:
        problem: Problem data
        model_chain: LLM chain
        model_name: Model identifier
        n_attempts: Number of times to generate attempts (n in pass@k formula)
        k_values: List of k values to calculate pass@k for (e.g. [1,3,5,10])
    """
    if k_values is None:
        k_values = [1, 5, n_attempts]
    
    problem_data = extract_problem_data(problem)
    
    attempts = []
    
    for i in range(n_attempts):
        # Temp strategy: 0.0 for first (greedy), 0.8 for rest (diverse)
        # If n_attempts is 1, strictly use 0.0
        if n_attempts == 1:
            temp = 0.0
        else:
            temp = 0.0 if i == 0 else min(0.3 + i * 0.1, 0.9)
        
        # Create a fresh LLM instance with the correct temperature
        llm = get_llm(model_name=model_name, temperature=temp)
        chain = implementer_gen_prompt | llm | StrOutputParser()
        
        res = evaluate_single_attempt(problem_data, chain, model_name, i, temp)
        attempts.append(res)
    
    # --- STATISTICS ---
    # n = total attempts, c = number of correct attempts
    c = sum(1 for a in attempts if a.get("pass"))
    n = n_attempts
    
    # Calculate Pass@k metrics for all specified k values
    pass_at_k = {}
    for k in k_values:
        if k <= n:
            pass_at_k[f"pass@{k}"] = calculate_pass_at_k(n, c, k)
        else:
            pass_at_k[f"pass@{k}"] = 0.0

    # First attempt result (temperature 0.0)
    first_attempt = attempts[0] if attempts else {}
    pass_first = first_attempt.get("pass", False)

    result = {
        "model_name": model_name,
        "id": problem_data["task_id"],
        "title": problem_data["title"],
        "n_samples": n,
        "c_correct": c,
        # Pass@k metrics for all specified k values
        **pass_at_k,
        # Pass@1 equals first attempt success (override)
        "pass@1": 1.0 if pass_first else 0.0,
        # Details
        "attempts": attempts
    }
    
    return result

def main_runner(model_name, output_file, min_count, max_count, concurrency=1, n_attempts=1, k_values=None):
    full_data = load_leetcode_dataset()
    end_index = len(full_data) if max_count <= 0 else max_count
    start_index = max(0, min_count)
    end_index = min(len(full_data), end_index)
    data_batch = full_data[start_index:end_index]
    
    if k_values is None:
        k_values = [1, 5, n_attempts]
    
    print(f"--- Starting Pass@k Evaluation: {model_name} ---")
    print(f"Tasks: {len(data_batch)} | Attempts per Task (n): {n_attempts}")
    print(f"Calculate Pass@k for k values: {k_values}")
    
    if len(data_batch) == 0: return

    # Dummy LLM just to pass into the function (will be overridden inside loop)
    llm = get_llm(temperature=0.0)
    chain = implementer_gen_prompt | llm | StrOutputParser()
    
    # Initialize stats for k values
    stats = {}
    for k in k_values:
        stats[f"pass@{k}"] = 0.0
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_problem = {
            executor.submit(evaluate_single_problem, p, chain, model_name, n_attempts, k_values): p 
            for p in data_batch
        }
        
        for future in tqdm(as_completed(future_to_problem), total=len(data_batch), desc="Eval"):
            res = future.result()
            
            # Aggregate results
            for k in k_values:
                key = f"pass@{k}"
                stats[key] += res.get(key, 0.0)
            
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(res) + "\n")

    total = len(data_batch)
    if total == 0:
        print("No tasks evaluated.")
        return
        
    print(f"\nEvaluation Finished!")
    for k in k_values:
        key = f"pass@{k}"
        print(f"Pass@{k}:            {stats[key]/total*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=LLM_MODEL_NAME)
    default_output_file = config.PASS_AT_K_EVAL_DIR / "results_pass_k.jsonl"
    parser.add_argument(
        '--output_file',
        type=str,
        default=str(default_output_file)
    )
    parser.add_argument('--min_count', type=int, default=0)
    parser.add_argument('--max_count', type=int, default=5)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--n_attempts', type=int, default=1, 
                       help='Number of times to generate code per task (n in pass@k formula)')
    parser.add_argument('--k_values', type=str, default=None,
                       help='Comma-separated list of k values to evaluate (e.g. "1,3,5,10"). '
                            'If not specified, defaults to [1, 5, n_attempts]')
    
    args = parser.parse_args()
    config.ensure_directories()
    
    # If output_file not specified, include model name in filename
    if args.output_file == str(default_output_file):
        safe_model_name = args.model_name.replace(':', '_').replace('/', '_')
        args.output_file = str(config.PASS_AT_K_EVAL_DIR / f"results_{safe_model_name}_pass_k.jsonl")

    # Parse k_values
    k_values = None
    if args.k_values:
        try:
            k_values = [int(k.strip()) for k in args.k_values.split(",")]
        except ValueError:
            print("Error: k_values must be comma-separated integers (e.g. '1,3,5,10')")
            exit(1)
    
    main_runner(
        model_name=args.model_name, 
        output_file=args.output_file, 
        min_count=args.min_count, 
        max_count=args.max_count, 
        concurrency=args.workers,
        n_attempts=args.n_attempts,
        k_values=k_values
    )