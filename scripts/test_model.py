# scripts/test_model.py
import pickle
import pytesseract
import cv2
import os

if __name__ == "__main__":
    # Load test_preprocessed
    with open('data/preprocessed_pairs.pkl', 'rb') as f:
        preprocessed_pairs = pickle.load(f)
    test_preprocessed = preprocessed_pairs['test_preprocessed']
    
    # Set TESSDATA_PREFIX
    os.environ['TESSDATA_PREFIX'] = os.path.abspath('data/training_data')
    
    correct = 0
    total = len(test_preprocessed)
    
    for img_path, true_label in test_preprocessed:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        custom_config = r'--oem 1 --psm 6 -l handwritten_model'
        predicted_label = pytesseract.image_to_string(img, config=custom_config).strip()
        if predicted_label == true_label:
            correct += 1
        print(f"True: {true_label}, Predicted: {predicted_label}")
    
    accuracy = correct / total
    print(f"Tesseract Accuracy: {accuracy:.2%} ({correct}/{total})")