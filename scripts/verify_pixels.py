# scripts/verify_pixels.py
import pickle
import cv2
import numpy as np
import os

def check_binarization(pairs):
    for path, label in pairs[:3]:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        unique = np.unique(img)
        print(f"{os.path.basename(path)}: {unique}")

if __name__ == "__main__":
    # Load TIFF and preprocessed pairs
    with open('data/tiff_pairs.pkl', 'rb') as f:
        tiff_pairs = pickle.load(f)
    with open('data/preprocessed_pairs.pkl', 'rb') as f:
        preprocessed_pairs = pickle.load(f)
    
    train_tiff = tiff_pairs['train_tiff']
    train_preprocessed = preprocessed_pairs['train_preprocessed']
    
    print("Original TIFF values:")
    check_binarization(train_tiff)
    
    print("\nPreprocessed values:")
    check_binarization(train_preprocessed)