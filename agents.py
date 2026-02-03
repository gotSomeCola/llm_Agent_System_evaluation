# agents.py
import os
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# LLM Configuration
LLM_MODEL_NAME = "gemma3:27b" 
API_BASE = "https://f2ki-h100-1.f2.htw-berlin.de:11435/v1"
API_KEY = "not-needed"

def get_llm(model_name=None, temperature=0.0):
    return ChatOpenAI(
        model=model_name or LLM_MODEL_NAME,
        openai_api_key=API_KEY,
        openai_api_base=API_BASE,
        temperature=temperature,
        max_tokens=8192,
        request_timeout=180
    )


# Automatic retry decorator for API rate limiting
def retry_on_rate_limit(max_retries=10, wait_seconds=10):
    """
    Decorator to automatically retry on 429 errors and other retryable errors.
    
    Args:
        max_retries: Maximum number of retry attempts
        wait_seconds: Wait time between retries (seconds), fixed value
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            wait_time = wait_seconds
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        print(f"✓ Retry successful (attempt {attempt + 1})")
                    return result
                    
                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    
                    # Check if error is retryable
                    is_retryable = any(
                        keyword in error_str.lower()
                        for keyword in ["429", "too many requests", "rate limit", "500", "503", 
                                       "timeout", "timed out", "connection", "unexpectedly stopped"]
                    )
                    
                    if attempt < max_retries and is_retryable:
                        print(f"⚠ API error (attempt {attempt + 1}/{max_retries + 1}): {error_str[:150]}")
                        print(f"  Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        raise last_error
            
            raise last_error
        return wrapper
    return decorator

# System prompts and prompt templates
IMPLEMENTER_SYSTEM_MESSAGE = (
    "You are an expert Java Developer. Your task is to implement a solution for an algorithmic problem.\n"
    "Rules:\n"
    "1. DO NOT output <think> tags. Output ONLY Java code.\n"
    "2. Use EXACTLY the method signature from the provided template.\n"
    "3. Do NOT write a 'public static void main' method.\n"
    "4. Implement the complete 'public class Solution' with ALL necessary imports.\n"
    "5. Your code MUST compile and MUST pass the test cases.\n"
    "6. Write ONLY Java code, NEVER Python or pseudocode."
)

# Initial code generation prompt
implementer_gen_prompt = ChatPromptTemplate.from_messages([
    ("system", IMPLEMENTER_SYSTEM_MESSAGE),
    ("human", 
     "Create java code function for this problem: {task_description}."
     "Use this class as the framework:{rahmen_code}\nAdd the function into this frame and give me the class Solution back."
     "Implement the Solution class in Java."
     "Notice: Don't forget to add import if it needed!"
    )
])

# Code repair prompt (used in feedback loop)
implementer_repair_prompt = ChatPromptTemplate.from_messages([
    ("system", IMPLEMENTER_SYSTEM_MESSAGE),
    ("human", 
     "We have a problem with your previous code.\n\n"
     "Task Description: {task_description}\n"
     "Your Previous Code:\n{code}\n\n"
     "Feedback from QA Tester:\n{feedback}\n\n"
     "Please fix the code based on the feedback. Return the full corrected class Solution."
    )
])

# Tester feedback prompt
tester_feedback_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a Senior QA Engineer. Your goal is to help the Developer fix bugs.\n"
     "POLICY:\n"
     "1. Read the 'Task Description' to understand the requirements.\n"
     "2. Analyze the 'Error Log' and 'Source Code'.\n"
     "3. Explain WHY the code fails based on the requirements and the error.\n"
     "4. Provide clear instructions for the Developer.\n"
     "5. DO NOT write the full fixed code yourself."
    ),
    ("human", 
     "Task Description:\n{task_description}\n\n"
     "Source Code:\n{code}\n\n"
     "Error Log:\n{error_log}\n\n"
     "Analyze the error and tell the Developer what to fix."
    )
])
