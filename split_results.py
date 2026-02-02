#!/usr/bin/env python3
"""
Split test results into pilot and real test sets.
- First 100 tasks from the first 600: pilot results
- Remaining 500 tasks: real tests results
"""

import json
import os
from pathlib import Path

def split_results(input_file, output_dir="./results/baseline_results"):
    """
    Split JSONL results into pilot and real tests.
    
    Args:
        input_file: Path to input JSONL file
        output_dir: Directory to save output files
    """
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Read first line to extract model name
    model_name = None
    with open(input_file, 'r') as infile:
        first_line = infile.readline()
        if first_line:
            first_record = json.loads(first_line)
            model_name = first_record.get('model_name', 'unknown')
    
    if not model_name:
        model_name = "unknown"
    
    # Create output filenames with model name
    pilot_file = os.path.join(output_dir, f"results_pilot_{model_name}.jsonl")
    real_tests_file = os.path.join(output_dir, f"results_tests_{model_name}.jsonl")
    
    pilot_count = 0
    real_tests_count = 0
    total_read = 0
    
    with open(input_file, 'r') as infile, \
         open(pilot_file, 'w') as pilot_out, \
         open(real_tests_file, 'w') as real_tests_out:
        
        for line in infile:
            if total_read >= 600:
                # Stop after processing 600 tasks
                break
            
            if total_read < 100:
                # First 100 tasks go to pilot
                pilot_out.write(line)
                pilot_count += 1
            else:
                # Next 500 tasks go to real tests
                real_tests_out.write(line)
                real_tests_count += 1
            
            total_read += 1
    
    print(f"✓ Split complete!")
    print(f"  Model: {model_name}")
    print(f"  Input file: {input_file}")
    print(f"  Total tasks processed: {total_read}")
    print(f"  Pilot results: {pilot_count} tasks -> {pilot_file}")
    print(f"  Real tests results: {real_tests_count} tasks -> {real_tests_file}")
    
    return pilot_file, real_tests_file


if __name__ == "__main__":
    # Use the results_tests_51-1051.jsonl file as input
    input_file = "results_gemma3_27b_20260130_014934.jsonl"
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        exit(1)
    
    pilot_file, real_tests_file = split_results(input_file)
    print(f"\n✓ Files created successfully!")
