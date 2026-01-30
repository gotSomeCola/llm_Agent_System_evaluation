# tools.py
import subprocess
import os
import shutil

def run_mvn_test(project_dir: str = "./project"):
    """
    Executes 'mvn clean test'.
    Returns:
        return_code (int): 0 for success, non-0 for failure.
        logs (str): Captured output.
    """
    if not shutil.which("mvn"):
        return -1, "CRITICAL ERROR: 'mvn' command not found."

    if not os.path.exists(project_dir):
        return -1, f"CRITICAL ERROR: Directory '{project_dir}' not found."

    try:
        result = subprocess.run(
            ["mvn", "clean", "test", "-Dtest=referenz.SolutionTest"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=45
        )
        return result.returncode, result.stdout + "\n" + result.stderr

    except subprocess.TimeoutExpired:
        return -1, "ERROR: Execution timed out."
    except Exception as e:
        return -1, f"SYSTEM ERROR: {str(e)}"