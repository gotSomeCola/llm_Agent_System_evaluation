import argparse
import json
import os
import shutil
from typing import TypedDict, Optional, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# LangGraph & LangChain Imports
from langgraph.graph import StateGraph, END, START
from langchain_core.output_parsers import StrOutputParser

# Custom modules
import metrics
from agents import (
    get_llm, 
    implementer_gen_prompt, 
    implementer_repair_prompt, 
    tester_feedback_prompt, 
    retry_on_rate_limit,
    LLM_MODEL_NAME
)
from utils import create_solution_file
from tools import run_mvn_test
from run_baseline_at_k import load_leetcode_dataset, setup_project_env, cleanup_project_env
from config import REPAIRS_EVAL_DIR

# Global variable for model name
GLOBAL_MODEL_NAME = LLM_MODEL_NAME

# State Definition
class AgentState(TypedDict):
    # Static data
    task_id: str
    task_description: str
    rahmen_code: str
    test_content: str
    referenz_code: Optional[str]
    
    # Dynamic data
    code: str             
    logs: str             
    feedback: str         
    iterations: int       
    success: bool         
    compilable: bool
    trace: list[str]

    # Environment paths
    project_dir: str
    src_dir: str
    test_dir: str

# Node Definitions

def setup_node(state: AgentState):
    """Initialize the environment and set up project directories."""
    proj, src, test = setup_project_env(state["task_id"])
    return {
        "project_dir": proj, 
        "src_dir": src, 
        "test_dir": test,
        "iterations": 0,
        "success": False,
        "compilable": False,
        "trace": []
    }

@retry_on_rate_limit(max_retries=10, wait_seconds=10)
def _invoke_llm_chain(chain, input_data):
    """Invoke LLM chain with automatic retry on rate limit errors."""
    return chain.invoke(input_data)

def implementer_node(state: AgentState):
    """Implementer node generates or repairs code based on feedback."""
    llm = get_llm(model_name=GLOBAL_MODEL_NAME, temperature=0.0)
    current_iter = state.get("iterations", 0)
    
    if current_iter == 0:
        # Initial code generation
        chain = implementer_gen_prompt | llm | StrOutputParser()
        generated_text = _invoke_llm_chain(chain, {
            "task_description": state["task_description"],
            "rahmen_code": state["rahmen_code"]
        })
    else:
        # Code repair mode
        chain = implementer_repair_prompt | llm | StrOutputParser()
        generated_text = _invoke_llm_chain(chain, {
            "task_description": state["task_description"],
            "code": state["code"],
            "feedback": state["feedback"]
        })
    
    clean_code = create_solution_file("// unused", generated_text)
    
    action = "Initial Draft" if current_iter == 0 else "Repair"
    msg = f"Iter {current_iter}: Implementer performed {action}"
    
    return {
        "code": clean_code, 
        "iterations": current_iter + 1,
        "trace": state["trace"] + [msg]
    }

def executor_node(state: AgentState):
    """Execute Maven compilation and test to verify code correctness."""
    # Ensure correct package declaration
    solution_code = state["code"]
    if not solution_code.strip().startswith("package referenz;"):
        lines = solution_code.splitlines()
        lines = [l for l in lines if not l.strip().startswith("package ")]
        solution_code = "package referenz;\n\n" + "\n".join(lines)

    with open(os.path.join(state["src_dir"], "Solution.java"), "w", encoding="utf-8") as f:
        f.write(solution_code)
        
    t_content = state["test_content"]
    if "package referenz" not in t_content:
        t_content = "package referenz;\n" + t_content
    with open(os.path.join(state["test_dir"], "SolutionTest.java"), "w", encoding="utf-8") as f:
        f.write(t_content)
        
    # Run Maven tests
    return_code, logs = run_mvn_test(state["project_dir"])
    eval_result = metrics.evaluate_test_results(return_code, logs)
    
    status = "PASSED" if eval_result["pass"] else "FAILED"
    msg = f"Iter {state['iterations']-1}: Test {status}"
    
    return {
        "success": eval_result["pass"], 
        "compilable": eval_result["compilable"],
        "logs": logs,
        "trace": state["trace"] + [msg]
    }

def tester_node(state: AgentState):
    """Tester node generates feedback on failed code execution."""
    llm = get_llm(model_name=GLOBAL_MODEL_NAME, temperature=0.0)
    chain = tester_feedback_prompt | llm | StrOutputParser()
    
    short_logs = state["logs"][:8000]
    
    feedback = _invoke_llm_chain(chain, {
        "task_description": state["task_description"],
        "code": state["code"],
        "error_log": short_logs
    })
    
    snippet = feedback[:100].replace("\n", " ") + "..."
    msg = f"Iter {state['iterations']-1}: Tester feedback: '{snippet}'"
    
    return {
        "feedback": feedback,
        "trace": state["trace"] + [msg]
    }

# Edge routing logic

def should_continue(state: AgentState) -> Literal["tester", END]:
    if state["success"]:
        return END
    
    # Max iterations: 5 (1 initial draft + 4 repair attempts)
    if state["iterations"] >= 5:
        return END
        
    return "tester"

# Graph builder

def build_repair_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("setup", setup_node)
    workflow.add_node("implementer", implementer_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("tester", tester_node)
    
    workflow.add_edge(START, "setup")
    workflow.add_edge("setup", "implementer")
    workflow.add_edge("implementer", "executor")
    
    workflow.add_conditional_edges(
        "executor",
        should_continue,
        {
            "tester": "tester",
            END: END
        }
    )
    
    workflow.add_edge("tester", "implementer")
    
    return workflow.compile()

# Task evaluation logic

def evaluate_single_task_graph(problem):
    raw_id = problem.get('task_id', 'unknown')
    task_id = str(raw_id).strip('/').split('/')[-1]
    
    def get_val(key): return problem.get(key) or problem.get('prompt', {}).get(key)
    
    initial_inputs = {
        "task_id": task_id,
        "task_description": get_val('description') or "No Desc",
        "rahmen_code": get_val('rahmen_code'),
        "referenz_code": get_val('referenz_code'),
        "test_content": get_val('test_content') or problem.get('test_code'),
        "iterations": 0,
        "success": False,
        "compilable": False,
        "code": "",
        "logs": "",
        "feedback": "",
        "trace": []
    }
    
    app = build_repair_graph()
    
    try:
        final_state = app.invoke(initial_inputs)
        cleanup_project_env(final_state["project_dir"])
        
        c_bleu = 0.0
        c_bert = 0.0
        if final_state["success"] and final_state.get("referenz_code"):
            c_bleu = metrics.calculate_code_bleu(final_state["referenz_code"], final_state["code"]).get("codebleu", 0.0)
            _, _, c_bert = metrics.evaluate_code_with_codeBert_score(final_state["referenz_code"], final_state["code"])

        # Calculate repair rounds: iterations - 1 (first iteration is initial draft)
        repair_rounds = final_state["iterations"] - 1

        # Pass@1: Success on first attempt (0 repair rounds)
        pass_at_1 = 1.0 if (final_state["success"] and repair_rounds == 0) else 0.0
        
        # Pass@k: Success within iteration limit
        pass_at_k = 1.0 if final_state["success"] else 0.0

        return {
            "model_name": GLOBAL_MODEL_NAME,
            "id": task_id,
            "title": problem.get('prompt', {}).get('problem', task_id),
            "pass": final_state["success"],
            "pass_at_1": pass_at_1,
            "pass_at_k": pass_at_k,
            "compilable": final_state.get("compilable", False),
            "repair_rounds": repair_rounds,
            "final_code": final_state["code"],
            "code_bleu": round(c_bleu, 4),
            "code_bert_f1": round(c_bert, 4),
            "logs": final_state["logs"][:500],
            "history": final_state.get("trace", [])
        }

    except Exception as e:
        cleanup_project_env(os.path.join("./project", f"proj_{task_id}"))
        return {"id": task_id, "pass": False, "error": str(e)}

def main_runner_graph(output_file, min_count, max_count, concurrency=1):
    full_data = load_leetcode_dataset()
    start_index = max(0, min_count)
    end_index = min(len(full_data), max_count if max_count > 0 else len(full_data))
    data_batch = full_data[start_index:end_index]
    
    print(f"Starting repair evaluation with graph-controlled workflow")
    print(f"Tasks: {len(data_batch)} | Workers: {concurrency}")
    
    success_count = 0
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_problem = {
            executor.submit(evaluate_single_task_graph, p): p 
            for p in data_batch
        }
        
        for future in tqdm(as_completed(future_to_problem), total=len(data_batch), desc="Graph Eval"):
            res = future.result()
            if res.get("pass"):
                success_count += 1
                
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(res) + "\n")

    print(f"Final Pass Rate: {(success_count/len(data_batch))*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default=str(REPAIRS_EVAL_DIR / "temp.jsonl"))
    parser.add_argument('--min_count', type=int, default=0)
    parser.add_argument('--max_count', type=int, default=5)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--model_name', type=str, default=LLM_MODEL_NAME)
    
    args = parser.parse_args()
    
    # Set global model name
    GLOBAL_MODEL_NAME = args.model_name
    
    main_runner_graph(
        output_file=args.output_file,
        min_count=args.min_count,
        max_count=args.max_count,
        concurrency=args.workers
    )
