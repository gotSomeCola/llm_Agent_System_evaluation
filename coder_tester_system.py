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

# Eigene Module
import metrics
from agents import (
    get_llm, 
    implementer_gen_prompt, 
    implementer_repair_prompt, 
    tester_feedback_prompt, 
    LLM_MODEL_NAME
)
from utils import create_solution_file
from tools import run_mvn_test
from run_baseline import load_leetcode_dataset, setup_project_env, cleanup_project_env

# --- 1. STATE DEFINITION ---
class AgentState(TypedDict):
    # Statische Daten
    task_id: str
    task_description: str
    rahmen_code: str
    test_content: str
    referenz_code: Optional[str]
    
    # Dynamische Daten
    code: str             
    logs: str             
    feedback: str         
    iterations: int       
    success: bool         
    compilable: bool      # Syntax-Status
    trace: list[str]      # Verlauf der Konversation

    # Env Paths
    project_dir: str
    src_dir: str
    test_dir: str

# --- 2. NODES ---

def setup_node(state: AgentState):
    """Initialisiert die Umgebung."""
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

def implementer_node(state: AgentState):
    """Implementer Agent (Coder)."""
    llm = get_llm(temperature=0.0)
    current_iter = state.get("iterations", 0)
    
    if current_iter == 0:
        # Initial Draft
        chain = implementer_gen_prompt | llm | StrOutputParser()
        generated_text = chain.invoke({
            "task_description": state["task_description"],
            "rahmen_code": state["rahmen_code"]
        })
    else:
        # Repair Mode
        chain = implementer_repair_prompt | llm | StrOutputParser()
        generated_text = chain.invoke({
            "task_description": state["task_description"],
            "code": state["code"],
            "feedback": state["feedback"]
        })
    
    clean_code = create_solution_file("// unused", generated_text)
    
    # TRACE LOGGING
    action = "Initial Draft" if current_iter == 0 else "Repair"
    msg = f"Iter {current_iter}: Implementer performed {action}"
    
    return {
        "code": clean_code, 
        "iterations": current_iter + 1,
        "trace": state["trace"] + [msg]
    }

def executor_node(state: AgentState):
    """Executor Tool (Maven)."""
    # 1. Package Fix (Robustheit)
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
        
    # 2. Maven Run
    return_code, logs = run_mvn_test(state["project_dir"])
    eval_result = metrics.evaluate_test_results(return_code, logs)
    
    # TRACE LOGGING
    status = "PASSED" if eval_result["pass"] else "FAILED"
    msg = f"Iter {state['iterations']-1}: Test {status}"
    
    return {
        "success": eval_result["pass"], 
        "compilable": eval_result["compilable"],
        "logs": logs,
        "trace": state["trace"] + [msg]
    }

def tester_node(state: AgentState):
    """Tester Agent (Feedback)."""
    llm = get_llm(temperature=0.0)
    chain = tester_feedback_prompt | llm | StrOutputParser()
    
    # Logs kürzen
    short_logs = state["logs"][:8000]
    
    feedback = chain.invoke({
        "task_description": state["task_description"],
        "code": state["code"],
        "error_log": short_logs
    })
    
    # TRACE LOGGING
    snippet = feedback[:100].replace("\n", " ") + "..."
    msg = f"Iter {state['iterations']-1}: Tester feedback: '{snippet}'"
    
    return {
        "feedback": feedback,
        "trace": state["trace"] + [msg]
    }

# --- 3. EDGES ---

def should_continue(state: AgentState) -> Literal["tester", END]:
    """Entscheidet über den nächsten Schritt."""
    # 1. Abbruch bei Erfolg
    if state["success"]:
        return END
    
    # 2. Abbruch bei Limit (1 Draft + 4 Repairs = 5 Iterationen)
    if state["iterations"] >= 5:
        return END
        
    # 3. Weiter zum Tester
    return "tester"

# --- 4. GRAPH BUILDER ---

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

# --- 5. RUNNER ---

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
        
        # --- METRICS CALCULATION ---
        c_bleu = 0.0
        c_bert = 0.0
        if final_state["success"] and final_state.get("referenz_code"):
            c_bleu = metrics.calculate_code_bleu(final_state["referenz_code"], final_state["code"]).get("codebleu", 0.0)
            _, _, c_bert = metrics.evaluate_code_with_codeBert_score(final_state["referenz_code"], final_state["code"])

        # Berechne Repair Rounds (Iterations - 1, da 1. Iteration der Initial Draft ist)
        # Wenn iterations=1 -> Rounds=0
        repair_rounds = final_state["iterations"] - 1

        # --- PASS AT K LOGIC (REPAIR CONTEXT) ---
        # Pass@1: Erfolg im Initial Draft (0 Repair Rounds)
        pass_at_1 = 1.0 if (final_state["success"] and repair_rounds == 0) else 0.0
        
        # Pass@k: Erfolg innerhalb des Limits (Egal wann)
        pass_at_k = 1.0 if final_state["success"] else 0.0

        return {
            "model_name": LLM_MODEL_NAME,
            "id": task_id,
            "title": problem.get('prompt', {}).get('problem', task_id),
            "pass": final_state["success"],
            "pass_at_1": pass_at_1,       # <--- NEU
            "pass_at_k": pass_at_k,       # <--- NEU
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
    
    print(f"--- Starting Graph-Controlled Repair Eval ---")
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
    parser.add_argument('--output_file', type=str, default="results_graph.jsonl")
    parser.add_argument('--min_count', type=int, default=0)
    parser.add_argument('--max_count', type=int, default=5)
    parser.add_argument('--workers', type=int, default=1)
    
    args = parser.parse_args()
    
    main_runner_graph(
        output_file=args.output_file,
        min_count=args.min_count,
        max_count=args.max_count,
        concurrency=args.workers
    )