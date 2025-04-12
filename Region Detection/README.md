
# 📌 Region Detection Phase

This repository contains code and resources for **region detection tasks**, focusing on identifying specific areas (e.g., text regions like **register numbers** and **subject codes**) in images. It uses a blend of traditional techniques and deep learning, including a **custom YOLOv5 model**.

---

## 📁 Project Structure

```
Region-Detection-Project/
├── Region Detection/
│   └── Region_Detection.ipynb         # Main notebook with all methods
├── aug_data/
│   ├── train/
│   │   ├── images/
│   │   ├── labels/
│   │   ├── labels_yolo/
│   │   └── labels.cache
│   ├── val/
│   │   ├── images/
│   │   ├── labels/
│   │   └── labels.cache
│   └── test/
│       ├── images/
│       └── labels/
├── yolov5/                            # Cloned Ultralytics YOLOv5 repo
└── README.md
```

---

## 🔍 Region Detection Methods

Implemented inside the notebook `Region_Detection.ipynb`:

### 1. **Fixed Bounding Box Cropping**
- Uses manually specified bounding box coordinates.
- Crops the region directly and sends it to the **Register Number Extraction Model**.
- Simple and fast, but less adaptive to layout changes.
### 2. **Custom CNN with VGG16 Backbone**
- A custom model built using **VGG16** as the base network.
- Includes top layers and a **custom loss function** for bounding box regression and classification.
- Useful for fine-grained control and experimentation.
- - ---

## 3. Custom YOLOv5 Model (Recommended)
- Trained on annotated images for detecting regions like register numbers.
- Includes dataset loading, training, validation, and inference.
- Offers the best tradeoff between accuracy and speed.
- Supports robust performance across various document layouts.
### 🔍 Explanation of the YOLOv5 Program

**1. Setup and Model Configuration**  
- The device is selected (`CUDA` or `CPU`) for training.  
- The YOLOv5 model is loaded with `yolov5s.yaml` and configured for **1 class** (register number region).  
- Image size is set to `640` and original image size (`2552x3302`) is specified for label scaling.

**2. Label Adjustment**  
- A helper function `adjust_labels()` rescales the bounding box coordinates from the original image dimensions to the YOLO model's input size (640x640).  
- It processes both training and validation labels.

**3. Dataset Preparation**  
- Images and labels are loaded using `create_dataloader()` for both training and validation sets.  
- Data is normalized, and batches are created with size 16.

**4. Training Loop**  
- A loss function (`ComputeLoss`) is used to optimize bounding box and classification accuracy.  
- `SGD` optimizer and `CosineAnnealingLR` scheduler help in gradual learning.  
- The model is trained for 50 epochs with live progress via `tqdm`.

**5. Saving the Model**  
- After training, the best model weights are saved to `runs/train/exp/weights/best.pt`.

**6. Inference**  
- The `infer_image()` function runs prediction on a test image (`page_1.png`).  
- It processes output with `non_max_suppression()` and prints **normalized bounding box coordinates** (center x/y, width, height).  
- This allows you to extract the register number region confidently for post-processing.

## ⚙️ Prerequisites

- **Python 3.x**
- **Install required packages**:
```bash
pip install torch numpy opencv-python matplotlib tqdm kagglehub
```

---

## 🛠️ Setup Instructions

### 1. Clone YOLOv5 Repository

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```
> Move the cloned `yolov5/` folder to the root of this project.

---

### 2. Download Dataset from Kaggle

Use `kagglehub` to download:

```python
import kagglehub

path = kagglehub.dataset_download("kavinraja1612/register-number-images-and-their-yolo-labels")
print("Path to dataset files:", path)
```

- After download, move contents to:
```
aug_data/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
```

> Ensure **YOLO format** is preserved (`.txt` label files with normalized coordinates).

---

## 🧪 Usage

### ✅ Training the Custom YOLOv5 Model
- Training is initiated in the notebook.
- Progress is shown via `tqdm`.
- Best model saved at:
```
runs/train/exp/weights/best.pt
```

### 🔍 Inference
- Run detection on a test image (e.g., `page_1.jpg`).
- Outputs bounding box predictions and class labels for regions of interest.

---

## 📦 Dataset Details

- **Source**: [Kaggle Dataset by @kavinraja1612](https://www.kaggle.com/datasets/kavinraja1612/register-number-images-and-their-yolo-labels)
- **Original Image Size**: 2552×3302 px  
- **Training Resize**: 640×640 px  
- **Content**:  
  - Register numbers and subject codes in scanned exam papers.
  - Labels in YOLO format (`class x_center y_center width height`, normalized).

---

## 🤝 Contributing

Want to help improve region detection or enhance dataset utilities? Fork the repo and submit a PR!

---

## 📬 Contact

For questions, suggestions, or support:  
📧 [kavinrajad.student@saveetha.ac.in](mailto:kavinrajad.student@saveetha.ac.in)

---

## 📝 Notes

- Ensure `Region_Detection.ipynb` includes all 3 methods described.
- Update `test_image_path` in the notebook if you're using different test images.
- For private Kaggle datasets, set up your API key properly.
