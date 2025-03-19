# scripts/preprocess_images.py
import pickle
import cv2
import os

def proper_preprocess(input_pairs, input_dir, output_dir):
    processed = []
    for tiff_path, label in input_pairs:
        # Read image
        img = cv2.imread(tiff_path, cv2.IMREAD_GRAYSCALE)
        
        # CORRECT ORDER: Blur first, then threshold
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Save to new location
        fname = os.path.basename(tiff_path)
        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, binary)
        processed.append((out_path, label))
    return processed

if __name__ == "__main__":
    # Load TIFF pairs
    with open('data/tiff_pairs.pkl', 'rb') as f:
        tiff_pairs = pickle.load(f)
    
    train_tiff = tiff_pairs['train_tiff']
    val_tiff = tiff_pairs['val_tiff']
    test_tiff = tiff_pairs['test_tiff']
    
    # Apply preprocessing
    train_preprocessed = proper_preprocess(train_tiff, 'data/train_tiff', 'data/train_preprocessed')
    val_preprocessed = proper_preprocess(val_tiff, 'data/val_tiff', 'data/val_preprocessed')
    test_preprocessed = proper_preprocess(test_tiff, 'data/test_tiff', 'data/test_preprocessed')
    
    # Save preprocessed pairs
    preprocessed_pairs = {
        'train_preprocessed': train_preprocessed,
        'val_preprocessed': val_preprocessed,
        'test_preprocessed': test_preprocessed
    }
    with open('data/preprocessed_pairs.pkl', 'wb') as f:
        pickle.dump(preprocessed_pairs, f)
    
    print(f"Preprocessed: Train: {len(train_preprocessed)}, Val: {len(val_preprocessed)}, Test: {len(test_preprocessed)}")