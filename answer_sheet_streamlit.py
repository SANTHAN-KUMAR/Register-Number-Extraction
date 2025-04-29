import streamlit as st
import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt

# Define the CRNN model class
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Dropout2d(0.3),
            nn.Conv2d(512, 512, kernel_size=(2, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True, dropout=0.3)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)
        x, _ = self.rnn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Define the AnswerSheetExtractor class
class AnswerSheetExtractor:
    def __init__(self, yolo_weights_path, register_crnn_model_path, subject_crnn_model_path):
        os.makedirs("cropped_register_numbers", exist_ok=True)
        os.makedirs("cropped_subject_codes", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo_model = YOLO(yolo_weights_path)
        
        self.register_crnn_model = CRNN(num_classes=11)  # 10 digits + blank
        self.register_crnn_model.to(self.device)
        checkpoint = torch.load(register_crnn_model_path, map_location=self.device)
        self.register_crnn_model.load_state_dict(checkpoint['model_state_dict'])
        self.register_crnn_model.eval()
        
        self.subject_crnn_model = CRNN(num_classes=37)  # blank + 0-9 + A-Z
        self.subject_crnn_model.to(self.device)
        self.subject_crnn_model.load_state_dict(torch.load(subject_crnn_model_path, map_location=self.device))
        self.subject_crnn_model.eval()
        
        self.register_transform = transforms.Compose([
            transforms.Resize((32, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.subject_transform = transforms.Compose([
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        self.char_map = {i: str(i-1) for i in range(1, 11)}
        self.char_map.update({i: chr(i - 11 + ord('A')) for i in range(11, 37)})

    def detect_regions(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        results = self.yolo_model(image)
        detections = results[0].boxes
        classes = results[0].names
        
        register_regions = []
        subject_regions = []
        
        for i, box in enumerate(detections):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = classes[class_id]
            cropped_region = image[y1:y2, x1:x2]
            
            if label == "RegisterNumber" and confidence > 0.5:
                save_path = f"cropped_register_numbers/register_number_{i}.jpg"
                cv2.imwrite(save_path, cropped_region)
                register_regions.append((save_path, confidence))
            elif label == "SubjectCode" and confidence > 0.5:
                save_path = f"cropped_subject_codes/subject_code_{i}.jpg"
                cv2.imwrite(save_path, cropped_region)
                subject_regions.append((save_path, confidence))
        
        return register_regions, subject_regions
    
    def extract_register_number(self, image_path):
        try:
            image = Image.open(image_path).convert('L')
            image_tensor = self.register_transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.register_crnn_model(image_tensor).squeeze(1)
                output = output.softmax(1).argmax(1)
                seq = output.cpu().numpy()
                prev = -1
                result = []
                for s in seq:
                    if s != 0 and s != prev:
                        result.append(s - 1)
                    prev = s
            return ''.join(map(str, result))
        except:
            return "ERROR"
    
    def extract_subject_code(self, image_path):
        try:
            image = Image.open(image_path).convert('L')
            image_tensor = self.subject_transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.subject_crnn_model(image_tensor).squeeze(1)
                output = output.softmax(1).argmax(1)
                seq = output.cpu().numpy()
                prev = 0
                result = []
                for s in seq:
                    if s != 0 and s != prev:
                        result.append(self.char_map.get(s, ''))
                    prev = s
            return ''.join(result)
        except:
            return "ERROR"
    
    def process_answer_sheet(self, image_path):
        register_regions, subject_regions = self.detect_regions(image_path)
        results = []
        register_cropped_path = None
        subject_cropped_path = None
        
        if register_regions:
            best_region = max(register_regions, key=lambda x: x[1])
            register_cropped_path = best_region[0]
            register_number = self.extract_register_number(register_cropped_path)
            results.append(("Register Number", register_number))
        
        if subject_regions:
            best_subject = max(subject_regions, key=lambda x: x[1])
            subject_cropped_path = best_subject[0]
            subject_code = self.extract_subject_code(subject_cropped_path)
            results.append(("Subject Code", subject_code))
        
        return results, register_cropped_path, subject_cropped_path

# Streamlit app
def main():
    st.title("Answer Sheet Extractor")
    
    # Load models
    with st.spinner("Loading models..."):
        try:
            extractor = AnswerSheetExtractor(
                "weights.pt",
                "best_crnn_model(git).pth",
                "best_subject_model_final.pth"
            )
            st.success("Models loaded successfully")
        except Exception as e:
            st.error(f"Failed to load models: {e}")
            return
    
    # Upload image
    uploaded_file = st.file_uploader("Upload Answer Sheet Image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # Save uploaded image
        image_path = "uploaded_image.png"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display uploaded image
        st.image(image_path, caption="Uploaded Image", use_column_width=True)
        
        # Process image
        if st.button("Extract Information"):
            with st.spinner("Processing image..."):
                try:
                    results, register_cropped, subject_cropped = extractor.process_answer_sheet(image_path)
                    st.success("Extraction complete")
                    
                    # Display results
                    for label, value in results:
                        st.write(f"**{label}:** {value}")
                    
                    # Display cropped images
                    if register_cropped:
                        st.image(register_cropped, caption="Cropped Register Number", width=200)
                    if subject_cropped:
                        st.image(subject_cropped, caption="Cropped Subject Code", width=200)
                except Exception as e:
                    st.error(f"Failed to process image: {e}")

if __name__ == "__main__":
    main()
