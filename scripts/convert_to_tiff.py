# scripts/convert_to_tiff.py
import pickle
from PIL import Image
import os

def convert_to_tiff(pairs, output_dir):
    converted = []
    for png_path, label in pairs:
        img = Image.open(png_path)
        tiff_name = os.path.basename(png_path).replace('.png', '.tif')
        tiff_path = os.path.join(output_dir, tiff_name)
        img.save(tiff_path, 'TIFF')
        converted.append((tiff_path, label))
    return converted

if __name__ == "__main__":
    # Load splits
    with open('data/splits.pkl', 'rb') as f:
        splits = pickle.load(f)
    
    train = splits['train']
    val = splits['val']
    test = splits['test']
    
    # Convert to TIFF
    train_tiff = convert_to_tiff(train, 'data/train_tiff')
    val_tiff = convert_to_tiff(val, 'data/val_tiff')
    test_tiff = convert_to_tiff(test, 'data/test_tiff')
    
    # Save TIFF pairs
    tiff_pairs = {
        'train_tiff': train_tiff,
        'val_tiff': val_tiff,
        'test_tiff': test_tiff
    }
    with open('data/tiff_pairs.pkl', 'wb') as f:
        pickle.dump(tiff_pairs, f)
    
    print(f"Converted to TIFF: Train: {len(train_tiff)}, Val: {len(val_tiff)}, Test: {len(test_tiff)}")