# scripts/finalize_model.py
import subprocess

if __name__ == "__main__":
    try:
        subprocess.run([
            'lstmtraining',
            '--stop_training',
            '--continue_from', 'data/training_data/handwritten_model_checkpoint',
            '--traineddata', '/usr/local/share/tessdata/eng.traineddata',
            '--model_output', 'data/training_data/handwritten_model.traineddata'
        ], check=True, capture_output=True, text=True)
        print("Model finalized successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during model finalization: {e.stderr}")
        raise