import os
from pathlib import Path

def get_project_root():
    return Path(__file__).resolve().parent.parent.parent

def save_to_results(filename, subdirectory=''):
    project_root = get_project_root()
    results_dir = project_root / 'handwritten_digit_generation' / 'results' / subdirectory
    results_dir.mkdir(parents=True, exist_ok=True)
    return str(results_dir / filename)

