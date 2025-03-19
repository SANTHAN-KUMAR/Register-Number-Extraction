# scripts/setup_and_prepare.py
import os
from PIL import Image
import cv2
import numpy as np
import random
import pickle

# Set up directories
data_dir = 'data/Register Numbers/'
os.makedirs('data/train_tiff', exist_ok=True)
os.makedirs('data/val_tiff', exist_ok=True)
os.makedirs('data/test_tiff', exist_ok=True)
os.makedirs('data/train_preprocessed', exist_ok=True)
os.makedirs('data/val_preprocessed', exist_ok=True)
os.makedirs('data/test_preprocessed', exist_ok=True)

# Load image paths and labels
image_pairs = [
    (os.path.join(data_dir, f), f.split('.')[0]) 
    for f in os.listdir(data_dir) 
    if f.endswith('.png')
]

# Validate labels
def validate_label(label):
    if len(label) != 12 or not label.isdigit():
        raise ValueError(f"Invalid register number: {label}")
    return label

if __name__ == "__main__":
    cleaned_pairs = [(path, validate_label(label[:12])) for path, label in image_pairs]
    
    # Save cleaned_pairs for the next script
    with open('data/cleaned_pairs.pkl', 'wb') as f:
        pickle.dump(cleaned_pairs, f)
    
    print(f"Total images: {len(cleaned_pairs)}")