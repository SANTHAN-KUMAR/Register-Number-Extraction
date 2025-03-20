# scripts/finetune_model.py
import subprocess
import os

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
    
    try:
        result = subprocess.run([
            'lstmtraining',
            '--model_output', 'data/training_data/handwritten_model',
            '--continue_from', 'data/training_data/eng.lstm',
            '--traineddata', '/usr/local/share/tessdata/eng.traineddata',
            '--train_listfile', 'data/train_preprocessed/train_files.txt',
            '--eval_listfile', 'data/val_preprocessed/val_files.txt',
            '--max_iterations', '5000'
        ], check=True, capture_output=True, text=True)
        print("Fine-tuning completed successfully.")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error during fine-tuning: {e.stderr}")
        print(f"Command output: {e.stdout}")
        raise
