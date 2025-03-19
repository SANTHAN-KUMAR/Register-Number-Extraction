import subprocess
import os

# Ensure the working directory is the project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# List of scripts to run in sequence
steps = [
    'finetune_model.py',
    'finalize_model.py',
    'test_model.py'
]

if __name__ == "__main__":
    for step in steps:
        print(f"\nRunning {step}...")
        try:
            subprocess.run(['python', f'scripts/{step}'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error in {step}: {e}")
            break