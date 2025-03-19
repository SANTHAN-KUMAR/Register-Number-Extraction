# scripts/create_ground_truth.py
import pickle
import os

def create_ground_truth_files(tiff_pairs, output_dir):
    for tiff_path, label in tiff_pairs:
        gt_path = os.path.join(output_dir, os.path.basename(tiff_path).replace('.tif', '.gt.txt'))
        with open(gt_path, 'w') as f:
            f.write(label)
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Failed to create ground truth file: {gt_path}")
        print(f"Created ground truth file: {gt_path}")

if __name__ == "__main__":
    # Load preprocessed pairs
    with open('data/preprocessed_pairs.pkl', 'rb') as f:
        preprocessed_pairs = pickle.load(f)
    
    train_preprocessed = preprocessed_pairs['train_preprocessed']
    val_preprocessed = preprocessed_pairs['val_preprocessed']
    
    # Create ground truth files
    create_ground_truth_files(train_preprocessed, 'data/train_preprocessed')
    create_ground_truth_files(val_preprocessed, 'data/val_preprocessed')
    
    # Verify all .gt.txt files exist
    for tiff_path, _ in train_preprocessed + val_preprocessed:
        gt_path = tiff_path.replace('.tif', '.gt.txt')
        if not os.path.exists(gt_path):
            print(f"Missing ground truth file: {gt_path}")