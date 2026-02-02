import argparse
import json
import os
import shutil
from typing import TypedDict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# LangChain / LangGraph
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# Import Metrics & Agents
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
    # Static Data
    task_id: str
    task_description: str
    rahmen_code: str
    test_content: str
    referenz_code: Optional[str]
    
    # Dynamic Data
    code: str             # Current Java Code
    logs: str             # Current Maven Logs
    feedback: str         # Tester Feedback
    iterations: int       # Attempt Counter
    success: bool         # Pass Status
    
    # Env Paths
    project_dir: str
    src_dir: str
    test_dir: str

# --- 2. NODES ---

def setup_node(state: AgentState):
    """Creates isolated folder structure."""
    proj, src, test = setup_project_env(state["task_id"])
    return {
        "project_dir": proj, 
        "src_dir": src, 
        "test_dir": test,
        "iterations": 0,
        "success": False
    }

def implementer_node(state: AgentState):
    """
    Role: Developer
    Logic: Chooses generation vs. repair prompt based on iteration count.
    """
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
    
    # Parse Markdown
    clean_code = create_solution_file("// unused", generated_text)
    
    return {"code": clean_code, "iterations": current_iter + 1}

def executor_node(state: AgentState):
    """
    Role: Tool Executor (File System + Maven)
    Logic: Enforces package declarations and runs tests.
    """
    # 1. Write Solution.java (Force Package Declaration)
    solution_code = state["code"]
    if not solution_code.strip().startswith("package referenz;"):
        # Remove existing package lines if they exist (to avoid duplicates)
        lines = solution_code.splitlines()
        lines = [l for l in lines if not l.strip().startswith("package ")]
        solution_code = "package referenz;\n\n" + "\n".join(lines)

    with open(os.path.join(state["src_dir"], "Solution.java"), "w", encoding="utf-8") as f:
        f.write(solution_code)
        
    # 2. Write Test File
    t_content = state["test_content"]
    if "package referenz" not in t_content:
        t_content = "package referenz;\n" + t_content
    with open(os.path.join(state["test_dir"], "SolutionTest.java"), "w", encoding="utf-8") as f:
        f.write(t_content)
        
    # 3. Execute Maven
    return_code, logs = run_mvn_test(state["project_dir"])
    eval_result = metrics.evaluate_test_results(return_code, logs)
    
    return {
        "success": eval_result["pass"], 
        "logs": logs
    }

def tester_node(state: AgentState):
    """
    Role: QA Engineer
    Logic: Analyzes logs and writes feedback.
    """
    llm = get_llm(temperature=0.0)
    chain = tester_feedback_prompt | llm | StrOutputParser()
    
    # Give Tester enough context (8k chars), but don't overflow context window
    short_logs = state["logs"][:8000]
    
    feedback = chain.invoke({
        "task_description": state["task_description"],
        "code": state["code"],
        "error_log": short_logs
    })
    
    return {"feedback": feedback}

# --- 3. EDGES ---

def should_continue(state: AgentState):
    """Decides the workflow path."""
    if state["success"]:
        return "end"
    
    # Stop after Max Retries (e.g., 1 Initial + 3 Repairs = 4 Total)
    if state["iterations"] >= 4:
        return "end"
        
    return "repair"

# --- 4. GRAPH BUILDER ---

def build_repair_graph():
    """Builds a fresh graph instance."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("setup", setup_node)
    workflow.add_node("implementer", implementer_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("tester", tester_node)
    
    workflow.set_entry_point("setup")
    workflow.add_edge("setup", "implementer")
    workflow.add_edge("implementer", "executor")
    
    workflow.add_conditional_edges(
        "executor",
        should_continue,
        {
            "end": END,
            "repair": "tester"
        }
    )
    
    workflow.add_edge("tester", "implementer")
    
    return workflow.compile()

# --- 5. RUNNER ---

def evaluate_single_task_graph(problem):
    """Runs the full workflow for a single task."""
    # 1. Prepare Data
    raw_id = problem.get('task_id', 'unknown')
    task_id = str(raw_id).strip('/').split('/')[-1]
    
    def get_val(key): return problem.get(key) or problem.get('prompt', {}).get(key)
    
    initial_inputs = {
        "task_id": task_id,
        "task_description": get_val('description') or "No Desc",
        "rahmen_code": get_val('rahmen_code'),
        "referenz_code": get_val('referenz_code'),
        "test_content": get_val('test_content') or problem.get('test_code'),
        # Dynamic fields start empty
        "iterations": 0,
        "success": False,
        "code": "",
        "logs": "",
        "feedback": ""
    }
    
    # 2. Build & Run Graph (Per-Thread Instance!)
    app = build_repair_graph()
    
    try:
        final_state = app.invoke(initial_inputs)
        
        # 3. Cleanup
        cleanup_project_env(final_state["project_dir"])
        
        # 4. Metrics
        c_bleu = 0.0
        c_bert = 0.0
        if final_state["success"] and final_state.get("referenz_code"):
            # Calculate metrics only on success to save time/compute
            # Or calculate always if you want to analyze failed attempts too
            c_bleu = metrics.calculate_code_bleu(final_state["referenz_code"], final_state["code"]).get("codebleu", 0.0)
            _, _, c_bert = metrics.evaluate_code_with_codeBert_score(final_state["referenz_code"], final_state["code"])

        return {
            "model_name": LLM_MODEL_NAME,
            "id": task_id,
            "title": problem.get('prompt', {}).get('problem', task_id),
            "pass": final_state["success"],
            "repair_rounds": final_state["iterations"] - 1, # 0 means First Try
            "final_code": final_state["code"],
            "code_bleu": round(c_bleu, 4),
            "code_bert_f1": round(c_bert, 4),
            "logs": final_state["logs"][:500] # Truncated for storage
        }

    except Exception as e:
        # Fallback Cleanup
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