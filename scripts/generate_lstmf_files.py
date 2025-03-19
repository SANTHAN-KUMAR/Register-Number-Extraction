# scripts/generate_lstmf_files.py
import pickle
import subprocess
import os

def generate_lstmf_files(tiff_pairs, output_dir):
    for tiff_path, label in tiff_pairs:
        # Verify the image, ground truth, and box files exist
        gt_path = tiff_path.replace('.tif', '.gt.txt')
        box_path = tiff_path.replace('.tif', '.box')
        if not os.path.exists(tiff_path):
            print(f"Skipping {tiff_path}: Image file not found")
            continue
        if not os.path.exists(gt_path):
            print(f"Skipping {tiff_path}: Ground truth file {gt_path} not found")
            continue
        if not os.path.exists(box_path):
            print(f"Skipping {tiff_path}: Box file {box_path} not found")
            continue
        
        base = os.path.splitext(tiff_path)[0]
        try:
            result = subprocess.run([
                'tesseract',
                tiff_path,
                base,
                '--psm', '6',
                'lstm.train'
            ], check=True, capture_output=True, text=True)
            print(f"Generated .lstmf for {tiff_path}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to generate .lstmf for {tiff_path}: {e.stderr}")

if __name__ == "__main__":
    # Load preprocessed pairs
    with open('data/preprocessed_pairs.pkl', 'rb') as f:
        preprocessed_pairs = pickle.load(f)
    
    train_preprocessed = preprocessed_pairs['train_preprocessed']
    val_preprocessed = preprocessed_pairs['val_preprocessed']
    
    # Generate .lstmf files
    generate_lstmf_files(train_preprocessed, 'data/train_preprocessed')
    generate_lstmf_files(val_preprocessed, 'data/val_preprocessed')