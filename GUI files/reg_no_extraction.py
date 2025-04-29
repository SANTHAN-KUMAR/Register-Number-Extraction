import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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
            nn.MaxPool2d((2,1), (2,1)),  # (N, 256, 4, W/4)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2,1), (2,1)),  # (N, 512, 2, W/4)
            nn.Dropout2d(0.3),  # Dropout after fourth maxpool
            nn.Conv2d(512, 512, kernel_size=(2,1)),  # (N, 512, 1, W/4)
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

# --- Load Register Number Model ---
def load_reg_no_model(model_path, device):
    """
    Load the Register Number CRNN model from the specified path.
    """
    model = CRNN(num_classes=11)  # Adjust the number of classes based on your model
    model.to(device)

    # Load the trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"[INFO] Register number model loaded from {model_path}")
    return model

# --- Extract Register Number ---
def extract_reg_no_from_image(cropped_image, model, device):
    """
    Extract the register number from a cropped image using the CRNN model.
    """
    transform = transforms.Compose([
        transforms.Resize((32, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Convert to PIL image and apply transformation
    image = Image.fromarray(cropped_image).convert('L')  # Convert to grayscale
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output = model(image)  # Get the model's output
        output = output.squeeze(1)  # Remove the channel dimension
        output = output.softmax(1).argmax(1)  # Get predicted class labels
        seq = output.cpu().numpy()
        
        prev = -1
        result = []
        for s in seq:
            if s != 0 and s != prev:
                result.append(s - 1)  # Convert class labels back to register number
            prev = s

    return ''.join(map(str, result))

# Example usage
if __name__ == "__main__":
    # Define device (CUDA or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model_path = 'path_to_model.pth'  # Replace with your model path
    model = load_reg_no_model(model_path, device)

    # Example image (replace with your cropped image)
    cropped_image = cv2.imread('s8.jpeg', cv2.IMREAD_GRAYSCALE)  # Read as grayscale

    # Extract register number
    reg_number = extract_reg_no_from_image(cropped_image, model, device)
    print(f"Predicted Register Number: {reg_number}")
