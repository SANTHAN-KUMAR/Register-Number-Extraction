# scripts/visualize_comparison.py
import pickle
import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load TIFF and preprocessed pairs
    with open('data/tiff_pairs.pkl', 'rb') as f:
        tiff_pairs = pickle.load(f)
    with open('data/preprocessed_pairs.pkl', 'rb') as f:
        preprocessed_pairs = pickle.load(f)
    
    train_tiff = tiff_pairs['train_tiff']
    train_preprocessed = preprocessed_pairs['train_preprocessed']
    
    # Compare first 3 samples
    for i in range(min(3, len(train_tiff))):
        orig_img = cv2.imread(train_tiff[i][0], cv2.IMREAD_GRAYSCALE)
        proc_img = cv2.imread(train_preprocessed[i][0], cv2.IMREAD_GRAYSCALE)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(121), plt.imshow(orig_img, cmap='gray'), plt.title('Original')
        plt.subplot(122), plt.imshow(proc_img, cmap='gray'), plt.title('Processed')
        plt.show()