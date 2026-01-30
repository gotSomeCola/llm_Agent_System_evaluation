# agents.py
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# --- LLM CONFIGURATION ---
# Wir nutzen Llama oder DeepSeek.
# WICHTIG: Wenn du DeepSeek nutzt, stelle sicher, dass utils.py den <think> Block entfernt.
LLM_MODEL_NAME = "gemma3:27b" 
API_BASE = "https://f2ki-h100-1.f2.htw-berlin.de:11435/v1"
API_KEY = "not-needed"

def get_llm(model_name=None, temperature=0.0):
    return ChatOpenAI(
        model=model_name or LLM_MODEL_NAME,
        openai_api_key=API_KEY,
        openai_api_base=API_BASE,
        temperature=temperature,
        max_tokens=8192 # Wichtig für Reasoning Models!
    )

# --- PROMPTS (THE "CONTROLLED CONSTANT") ---

# 1. IMPLEMENTER AGENT
# Strategie: Role-Based + Constraint-Based.
# Wir geben ihm die Signatur und verbieten 'main'.
implementer_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an expert Java Developer. Your task is to implement a solution for an algorithmic problem.\n"
     "Rules:\n"
     "1. DO NOT output <think> tags or explanations. Output ONLY Java code.\n"
     "2. Use EXACTLY the method signature from the provided template.\n"
     "3. Do NOT write a 'public static void main' method.\n"
     "4. Implement the complete 'public class Solution' with ALL necessary imports.\n"
     "5. Your code MUST compile and MUST pass the test cases.\n"
     "6. Write ONLY Java code, NEVER Python or pseudocode."
    ),
    ("human", 
     "Create java code function for this problem: {task_description}."
     "Use this class as the framework:{rahmen_code}\nAdd the function into this frame and give me the class Solution back."
     "Implement the Solution class in Java."
     "Notice: Don't forget to add import if it needed!"
    )
])

# 2. PLANNER AGENT (Für spätere Phasen)
# Strategie: Pure Logic. Kein Code.
planner_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a Software Architect. You analyze algorithmic problems and design logical solutions.\n"
     "POLICY:\n"
     "1. Analyze the problem constraints and edge cases.\n"
     "2. Describe the algorithm step-by-step.\n"
     "3. Do NOT write Java code. Write in clear English (or logical pseudocode)."
    ),
    ("human", "Problem: {problem_description}")
])

# 3. TESTER AGENT (Für spätere Phasen)
# Strategie: Error Analysis.
tester_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a QA Engineer. You analyze Java compilation errors and test failures.\n"
     "POLICY:\n"
     "1. Read the error log and the source code.\n"
     "2. Identify exactly why the test failed (Logic error? Syntax error?).\n"
     "3. Provide a corrected version of the code or specific instructions to fix it."
    ),
    ("human", 
     "Source Code:\n{code}\n\n"
     "Error Log:\n{error_log}\n\n"
     "Why did it fail and how to fix it?"
    )
])