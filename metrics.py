# metrics.py
import logging
import re

# Library checks (CodeBERT / CodeBLEU)
try:
    import code_bert_score
    CODEBERT_AVAILABLE = True
except ImportError:
    print("WARNING: 'code-bert-score' not installed. Metrics will be 0.")
    CODEBERT_AVAILABLE = False

try:
    from codebleu import calc_codebleu
    CODEBLEU_AVAILABLE = True
except ImportError:
    print("WARNING: 'codebleu' not installed. Metrics will be 0.")
    CODEBLEU_AVAILABLE = False

def evaluate_test_results(return_code: int, logs: str) -> dict:
    """
    Analyzes Maven logs to determine compilability and test success.
    
    Args:
        return_code (int): The exit code of the subprocess (0 = success, non-0 = failure).
        logs (str): The combined stdout/stderr from Maven.
        
    Returns:
        dict: {
            "compilable": bool,
            "pass": bool,
            "reason": str  # Helpful for debugging (e.g. 'Compilation Error', 'Tests Failed')
        }
    """
    result = {
        "compilable": False,
        "pass": False,
        "reason": "Unknown Failure"
    }

    # 1. Check for COMPILATION SUCCESS
    # If "COMPILATION ERROR" is in logs, it definitely failed to compile.
    if "COMPILATION ERROR" in logs:
        result["compilable"] = False
        result["reason"] = "Compilation Error"
        return result
    
    # If build failed but no compilation error, it likely compiled but tests failed
    if "BUILD FAILURE" in logs:
        # Double check phases to be sure
        if "Compilation failure" in logs or "[ERROR] /" in logs: # Source file error
             result["compilable"] = False
             result["reason"] = "Compilation Failure"
        else:
             # It compiled, but tests failed (e.g. assertion error)
             result["compilable"] = True
             result["pass"] = False
             result["reason"] = "Tests Failed"
        return result

    # 2. Check for FULL SUCCESS
    # Return code 0 AND "BUILD SUCCESS" means everything passed
    if return_code == 0 and "BUILD SUCCESS" in logs:
        result["compilable"] = True
        result["pass"] = True
        result["reason"] = "Success"
        return result

    # Fallback for timeouts or system errors
    if return_code == -1: # Our custom code for timeout/system error
        result["reason"] = "System Error / Timeout"
    
    return result

def evaluate_code_with_codeBert_score(reference_code, candidate_code):
    """Calculates Precision, Recall, F1 using CodeBERT."""
    if not CODEBERT_AVAILABLE: return 0.0, 0.0, 0.0
    try:
        scores = code_bert_score.score(cands=[candidate_code], refs=[reference_code], lang='java')
        return scores[0].mean().item(), scores[1].mean().item(), scores[2].mean().item()
    except Exception as e:
        logging.warning(f"CodeBERT failed: {e}")
        return 0.0, 0.0, 0.0

def calculate_code_bleu(reference_code, candidate_code):
    """Calculates CodeBLEU score."""
    if not CODEBLEU_AVAILABLE: return {"codebleu": 0.0}
    try:
        return calc_codebleu([reference_code], [candidate_code], lang="java", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
    except Exception as e:
        logging.warning(f"CodeBLEU failed: {e}")
        return {"codebleu": 0.0}

def extract_runtime_from_logs(logs: str) -> float:
    """Extracts execution time from logs."""
    match = re.search(r'Time elapsed:\s+([0-9.]+)\s+s', logs)
    if match:
        try: return float(match.group(1))
        except ValueError: pass
    return 0.0