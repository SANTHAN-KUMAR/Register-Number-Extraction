# scripts/extract_eng_lstm.py
import subprocess
import os

if __name__ == "__main__":
    os.makedirs('data/training_data', exist_ok=True)
    
    # Update the path to eng.traineddata based on your Tesseract 5.3.4 installation
    subprocess.run([
        'cp',
        '/usr/local/share/tessdata/eng.traineddata',  # Updated path for Tesseract 5.3.4
        'data/training_data/'
    ], check=True)
    
    subprocess.run([
        'combine_tessdata',
        '-u',
        'data/training_data/eng.traineddata',
        'data/training_data/eng'
    ], check=True)
    
    print("Files in data/training_data/:")
    print(os.listdir('data/training_data'))