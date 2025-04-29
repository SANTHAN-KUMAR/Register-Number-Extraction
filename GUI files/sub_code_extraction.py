import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# ----------------------
# Define your CRNN Model
# ----------------------
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
            nn.ReLU()
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

# -------------------------------
# Load Model and Utility Functions
# -------------------------------

def load_subject_code_model(model_path, device):
    model = CRNN(num_classes=37)  # 0-9, A-Z + blank (index 0)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0)  # (1, 1, 32, 128)

def decode_predictions(predictions):
    char_map = {i: str(i-1) for i in range(1, 11)}
    char_map.update({i: chr(i - 11 + ord('A')) for i in range(11, 37)})

    preds = predictions.softmax(2).argmax(2)
    preds = preds.squeeze(1).cpu().numpy()

    text = []
    prev = 0
    for p in preds:
        if p != 0 and p != prev:
            text.append(char_map.get(p, ''))
        prev = p

    return ''.join(text)

# -------------------------------
# Main Function to Extract Subject Code
# -------------------------------

def extract_subject_code_from_image(image_np, model, device):
    """
    Args:
        image_np: Cropped Subject Code region (numpy array).
        model: Loaded CRNN model.
        device: cuda or cpu.
    Returns:
        subject_code (str)
    """
    image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)).convert('L')  # Convert BGR->RGB->GRAY
    image_tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        output = model(image_tensor)
    
    subject_code = decode_predictions(output)
    return subject_code

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    model_path = "D:/reg_no_gui/best_subject_code_model.pth"  # << Your subject code model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_subject_code_model(model_path, device)

    # Example: Load a sample cropped image
    image_np = cv2.imread("subject_code_crops/subject_code_crop_0.png")  # Cropped subject image
    subject_code = extract_subject_code_from_image(image_np, model, device)
    print(f"Extracted Subject Code: {subject_code}")
