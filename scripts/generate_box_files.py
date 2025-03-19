# scripts/generate_box_files.py
import pickle
from PIL import Image
import os

def create_box_file(tiff_path, label):
    # Open the image to get dimensions
    img = Image.open(tiff_path)
    width, height = img.size
    
    # Assume equal spacing for 12 digits
    char_width = width // 12
    
    # Generate box coordinates for each character
    box_content = []
    for i, char in enumerate(label):
        left = i * char_width
        right = (i + 1) * char_width
        # Tesseract box format: <char> <left> <bottom> <right> <top> <page>
        box_line = f"{char} {left} 0 {right} {height} 0"
        box_content.append(box_line)
    
    # Write to .box file
    box_path = tiff_path.replace('.tif', '.box')
    with open(box_path, 'w') as f:
        f.write('\n'.join(box_content))
    print(f"Created box file: {box_path}")

if __name__ == "__main__":
    # Load preprocessed pairs
    with open('data/preprocessed_pairs.pkl', 'rb') as f:
        preprocessed_pairs = pickle.load(f)
    
    train_preprocessed = preprocessed_pairs['train_preprocessed']
    val_preprocessed = preprocessed_pairs['val_preprocessed']
    
    # Generate .box files
    for tiff_path, label in train_preprocessed:
        create_box_file(tiff_path, label)
    for tiff_path, label in val_preprocessed:
        create_box_file(tiff_path, label)