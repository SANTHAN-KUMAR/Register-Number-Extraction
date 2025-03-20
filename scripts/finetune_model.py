# scripts/finetune_model.py
import subprocess

if __name__ == "__main__":
    try:
        subprocess.run([
            'lstmtraining',
            '--model_output', 'data/training_data/handwritten_model',
            '--continue_from', 'data/training_data/eng.lstm',
            '--traineddata', '/usr/local/share/tessdata/eng.traineddata',
            '--train_listfile', 'data/train_preprocessed/train_files.txt',
            '--eval_listfile', 'data/val_preprocessed/val_files.txt',
            '--max_iterations', '1000',  # Increased to 5000
            '--target_error_rate', '0.01',  # Stop if error rate drops below 1%
            '--debug_interval', '100'  # Print training progress every 100 iterations
        ], check=True, capture_output=True, text=True)
        print("Fine-tuning completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during fine-tuning: {e.stderr}")
        raise
