import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO
import cv2
import numpy as np
import argparse

# Define the CRNN class (must match exactly the original definition)
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        # CNN component with dropout
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (N, 64, 16, W/2)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (N, 128, 8, W/4)
            nn.Dropout2d(0.3),  # Dropout after second maxpool
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),  # (N, 256, 4, W/4)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),  # (N, 512, 2, W/4)
            nn.Dropout2d(0.3),  # Dropout after fourth maxpool
            nn.Conv2d(512, 512, kernel_size=(2, 1)),  # (N, 512, 1, W/4)
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        # LSTM with dropout between layers
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True, dropout=0.3)
        # Dropout before the fully connected layer
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)  # 512 because bidirectional (256 * 2)

    def forward(self, x):
        # Pass through CNN
        x = self.cnn(x)  # (N, 512, 1, W/4)
        x = x.squeeze(2)  # (N, 512, W/4)
        x = x.permute(2, 0, 1)  # (W/4, N, 512) for LSTM
        # Pass through LSTM
        x, _ = self.rnn(x)  # (W/4, N, 512)
        # Apply dropout before FC
        x = self.dropout(x)
        # Fully connected layer for classification
        x = self.fc(x)  # (W/4, N, num_classes)
        return x

class AnswerSheetExtractor:
    def __init__(self, yolo_weights_path, crnn_model_path):
        # Create output directories
        os.makedirs("cropped_register_numbers", exist_ok=True)
        os.makedirs("cropped_subject_codes", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load YOLO model
        print(f"Loading YOLO model from {yolo_weights_path}...")
        self.yolo_model = YOLO(yolo_weights_path)
        
        # Load CRNN model
        print(f"Loading CRNN model from {crnn_model_path}...")
        self.crnn_model = CRNN(num_classes=11)  # 10 digits + blank
        self.crnn_model.to(self.device)
        
        try:
            checkpoint = torch.load(crnn_model_path, map_location=self.device)
            self.crnn_model.load_state_dict(checkpoint['model_state_dict'])
            self.crnn_model.eval()
            print(f"CRNN model loaded successfully from epoch {checkpoint['epoch']} with Val Acc: {checkpoint.get('val_accuracy', 'N/A')}")
        except Exception as e:
            print(f"Error loading CRNN model: {e}")
            raise
        
        # Define transform for CRNN
        self.transform = transforms.Compose([
            transforms.Resize((32, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def detect_regions(self, image_path):
        """Detect register number and subject code regions using YOLO"""
        print(f"Processing image: {image_path}")
        
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Run inference with YOLO
        results = self.yolo_model(image)
        
        # Get detection details
        detections = results[0].boxes
        classes = results[0].names
        
        # Lists to store cropped regions
        register_regions = []
        subject_regions = []
        
        # Process each detection and crop the regions
        for i, box in enumerate(detections):
            # Get coordinates, confidence, and class
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = classes[class_id]
            
            # Crop the region from the original image
            cropped_region = image[y1:y2, x1:x2]
            
            # Save and store the cropped region based on the class
            if label == "RegisterNumber" and confidence > 0.5:
                save_path = f"cropped_register_numbers/register_number_{i}.jpg"
                cv2.imwrite(save_path, cropped_region)
                register_regions.append((save_path, confidence))
                print(f"Saved RegisterNumber region: {save_path} (confidence: {confidence:.2f})")
                
            elif label == "SubjectCode" and confidence > 0.5:
                save_path = f"cropped_subject_codes/subject_code_{i}.jpg"
                cv2.imwrite(save_path, cropped_region)
                subject_regions.append((save_path, confidence))
                print(f"Saved SubjectCode region: {save_path} (confidence: {confidence:.2f})")
        
        print(f"Total RegisterNumber regions detected: {len(register_regions)}")
        print(f"Total SubjectCode regions detected: {len(subject_regions)}")
        
        return register_regions, subject_regions
    
    def extract_register_number(self, image_path):
        """Extract register number from cropped image using CRNN"""
        try:
            # Open image and convert to grayscale
            image = Image.open(image_path).convert('L')
            
            # Apply transformations
            image_tensor = self.transform(image)
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
            
            # Inference
            with torch.no_grad():
                image_tensor = image_tensor.to(self.device)
                output = self.crnn_model(image_tensor)
                output = output.squeeze(1)
                output = output.softmax(1).argmax(1)
                seq = output.cpu().numpy()
                
                # Process sequence (CTC-like decoding)
                prev = -1
                result = []
                for s in seq:
                    if s != 0 and s != prev:  # 0 is blank class
                        result.append(s - 1)
                    prev = s
            
            # Convert to string
            predicted_number = ''.join(map(str, result))
            return predicted_number
        except Exception as e:
            print(f"Error extracting register number from {image_path}: {e}")
            return "ERROR"
    
    def process_answer_sheet(self, image_path, visualize=True):
        """Process an entire answer sheet image"""
        # Detect regions
        register_regions, subject_regions = self.detect_regions(image_path)
        
        results = []
        
        # Process register number regions (using the most confident one if multiple)
        if register_regions:
            # Sort by confidence (highest first)
            register_regions.sort(key=lambda x: x[1], reverse=True)
            best_region_path = register_regions[0][0]
            
            # Extract register number
            register_number = self.extract_register_number(best_region_path)
            print(f"Extracted Register Number: {register_number}")
            results.append(("Register Number", register_number))
            
            # Visualize if requested
            if visualize:
                self._visualize_result(image_path, best_region_path, register_number)
        else:
            print("No register number regions detected")
        
        # For now, we're skipping subject code extraction as mentioned
        if subject_regions:
            print(f"Subject code region detected but extraction not implemented")
            results.append(("Subject Code", "Not implemented"))
        
        return results
    
    def _visualize_result(self, original_image_path, cropped_image_path, prediction):
        """Create a visualization of the detection and extraction"""
        # Create a figure with 2 subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image with bounding box (simplified version)
        original = cv2.imread(original_image_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        axs[0].imshow(original)
        axs[0].set_title("Original Answer Sheet")
        axs[0].axis('off')
        
        # Cropped region
        cropped = cv2.imread(cropped_image_path)
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        axs[1].imshow(cropped)
        axs[1].set_title(f"Extracted Register Number: {prediction}")
        axs[1].axis('off')
        
        # Save and show
        result_path = f"results/result_{os.path.basename(original_image_path)}"
        plt.tight_layout()
        plt.savefig(result_path)
        plt.close()
        print(f"Visualization saved to {result_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract register numbers from answer sheets")
    parser.add_argument("--image", required=True, help="CIA test.png")
    parser.add_argument("--yolo_weights", default="weights.pt", help="weights.pt")
    parser.add_argument("--crnn_model", default="best_crnn_model.pth", help="best_crnn_model(model).pth")
    parser.add_argument("--no_viz", action="store_true", help="Disable visualization")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = AnswerSheetExtractor(args.yolo_weights, args.crnn_model)
    
    # Process the answer sheet
    results = extractor.process_answer_sheet(args.image, visualize=not args.no_viz)
    
    # Print final results
    print("\n=== EXTRACTION RESULTS ===")
    for label, value in results:
        print(f"{label}: {value}")

if __name__ == "__main__":
    main()
