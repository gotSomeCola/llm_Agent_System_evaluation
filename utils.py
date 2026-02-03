# utils.py

import re

def parse_llm_response(full_response: str):
    """
    Clean up LLM response and extract code from markdown blocks.
    """
    text = full_response.strip()
    
    # Extract markdown code block (```java or ```)
    code_blocks = re.findall(r'```(?:java)?\s*\n?(.*?)```', text, re.DOTALL)
    
    if code_blocks:
        # Use the longest block (usually the code, not the explanation)
        clean_code = max(code_blocks, key=len)
    else:
        # No markdown, use text directly
        clean_code = text

    return clean_code.strip()

def create_solution_file(framework_code_unused: str, llm_response: str) -> str:
    """
    Create the final Solution.java file from LLM response.
    """
    raw_code = parse_llm_response(llm_response)
    
    lines = raw_code.splitlines()
    final_lines = []
    
    # Set package header
    final_lines.append("package referenz;")
    final_lines.append("") 
    
    class_found = False
    
    for line in lines:
        stripped = line.strip()
        
        # Remove package declaration from LLM response
        if stripped.startswith("package "):
            continue
            
        # Rename class if needed (e.g., public class Main -> Solution)
        if "class " in line and "Solution" not in line:
            if "public class" in line:
                 line = re.sub(r'public class \w+', 'public class Solution', line)
            elif not class_found:
                 line = line.replace("class ", "public class ")
                 line = re.sub(r'class \w+', 'class Solution', line)
        
        if "class " in line:
            class_found = True
            
        final_lines.append(line)
        
    if not final_lines:
        return "// ERROR: No code extracted from LLM response"

    return "\n".join(final_lines)