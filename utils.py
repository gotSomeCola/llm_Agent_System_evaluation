# utils.py

import re

def parse_llm_response(full_response: str):
    """
    Bereinigt die Antwort vom LLM.
    Extrahiert den Code aus Markdown Code-Blöcken.
    """
    text = full_response.strip()
    
    # Markdown Code Block extrahieren
    # Suche nach ```java oder einfach nur ```
    code_blocks = re.findall(r'```(?:java)?\s*\n?(.*?)```', text, re.DOTALL)
    
    if code_blocks:
        # Nimm den längsten Block (meistens der Code, nicht die Erklärung)
        clean_code = max(code_blocks, key=len)
    else:
        # Kein Markdown? Versuche direkten Text
        clean_code = text

    return clean_code.strip()

def create_solution_file(framework_code_unused: str, llm_response: str) -> str:
    """
    Erstellt die finale Solution.java Datei.
    """
    raw_code = parse_llm_response(llm_response)
    
    lines = raw_code.splitlines()
    final_lines = []
    
    # Header setzen
    final_lines.append("package referenz;")
    final_lines.append("") 
    
    class_found = False
    
    for line in lines:
        stripped = line.strip()
        
        # Package vom LLM entfernen
        if stripped.startswith("package "):
            continue
            
        # Klasse umbenennen falls nötig (z.B. public class Main -> Solution)
        if "class " in line and "Solution" not in line:
            if "public class" in line:
                 line = re.sub(r'public class \w+', 'public class Solution', line)
            elif not class_found: # Die erste gefundene Klasse wird Solution
                 line = line.replace("class ", "public class ")
                 line = re.sub(r'class \w+', 'class Solution', line)
        
        if "class " in line:
            class_found = True
            
        final_lines.append(line)
        
    if not final_lines:
        return "// ERROR: No code extracted from LLM response"

    return "\n".join(final_lines)