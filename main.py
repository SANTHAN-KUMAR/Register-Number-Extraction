import subprocess
import os

# Ensure the working directory is the project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# List of scripts to run in sequence
steps = [
    'setup_and_prepare.py',
    'split_data.py',
    'convert_to_tiff.py',
    'preprocess_images.py',
    'visualize_comparison.py',
    'verify_pixels.py',
    'create_ground_truth.py',
    'generate_box_files.py',
    'generate_lstmf_files.py',
    'create_file_lists.py',
    'extract_eng_lstm.py',
    'finetune_model.py',
    'finalize_model.py',
    'test_model.py',
]

if __name__ == "__main__":
    for step in steps:
        print(f"\nRunning {step}...")
        try:
            subprocess.run(['python', f'scripts/{step}'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error in {step}: {e}")
            break
