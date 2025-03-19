# scripts/finetune_model.py
import subprocess
import os
import sys

if __name__ == "__main__":
    # Check if input files exist
    required_files = [
        'data/training_data/eng.lstm',
        '/usr/local/share/tessdata/eng.traineddata',
        'data/train_preprocessed/train_files.txt',
        'data/val_preprocessed/val_files.txt'
    ]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    # Ensure output directory exists
    os.makedirs('data/training_data', exist_ok=True)
    
    # Determine the number of CPU cores
    num_threads = os.cpu_count()
    if num_threads is None:
        num_threads = 4  # Fallback to 4 if cpu_count() fails
        print("Warning: Could not determine the number of CPU cores. Defaulting to 4 threads.")
    else:
        print(f"Detected {num_threads} CPU cores. Setting --num_threads to {num_threads}.")
    
    # Command for lstmtraining
    command = [
        'lstmtraining',
        '--model_output', 'data/training_data/handwritten_model',
        '--continue_from', 'data/training_data/eng.lstm',
        '--traineddata', '/usr/local/share/tessdata/eng.traineddata',
        '--train_listfile', 'data/train_preprocessed/train_files.txt',
        '--eval_listfile', 'data/val_preprocessed/val_files.txt',
        '--max_iterations', '5000',
        '--target_error_rate', '0.01',
        '--debug_interval', '100',
        '--num_threads', str(num_threads)  # Use all available CPU cores
    ]
    
    print("Starting fine-tuning...")
    print("Command:", " ".join(command))
    
    # Use subprocess.Popen to stream output in real-time
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # Line-buffered
        universal_newlines=True
    )
    
    # Stream stdout and stderr in real-time
    while True:
        # Read stdout
        stdout_line = process.stdout.readline()
        if stdout_line:
            print(stdout_line, end='', flush=True)
        
        # Read stderr
        stderr_line = process.stderr.readline()
        if stderr_line:
            print(stderr_line, end='', flush=True, file=sys.stderr)
        
        # Check if the process has finished
        if process.poll() is not None:
            break
    
    # Ensure all output is flushed
    stdout_remaining = process.stdout.read()
    stderr_remaining = process.stderr.read()
    if stdout_remaining:
        print(stdout_remaining, end='', flush=True)
    if stderr_remaining:
        print(stderr_remaining, end='', flush=True, file=sys.stderr)
    
    # Check the return code
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command, output=stdout_remaining, stderr=stderr_remaining)
    
    print("\nFine-tuning completed successfully.")