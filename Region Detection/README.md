
# 📌 Register Number Region Detection using YOLOv11

This project phase detects register number regions in student documents using a custom-trained YOLOv11 model.

---

## 🚀 Project Summary

- **Model**: YOLOv11s and YOLOv11n (Ultralytics)
- **Dataset**: Custom-labeled via [Roboflow](https://roboflow.com)
- **Classes**: `Register-Number` only
- **Images**: 50 training, 10 validation, 5 test
- **Annotations**: Created with LabelMe, converted to YOLO format via Roboflow
- **Framework**: Ultralytics `YOLOv11` (PyTorch-based)

---

## 🔧 Setup Instructions

1. **Install dependencies**:
   ```bash
   pip install ultralytics roboflow
   ```

2. **Download dataset securely (DO NOT expose API key in code)**:
   ```python
    
    from roboflow import Roboflow
    rf = Roboflow(api_key="i0jt76QKedNzuxBlhPjT")
    project = rf.workspace("reg-number-region-detection").project("register-number-detetction")
    version = project.version(2)
    dataset = version.download("yolov11")
   ```

3. **Train model**:
   ```bash
   yolo task=detect mode=train model=yolo11s.pt data={dataset.location}/data.yaml epochs=50 imgsz=640 plots=True
   ```

4. **Validate model**:
   ```bash
   yolo task=detect mode=val model=runs/detect/train/weights/best.pt data={dataset.location}/data.yaml
   ```

5. **Predict on test images**:
   ```bash
   yolo task=detect mode=predict \
     model=runs/detect/train/weights/best.pt \
     source=/path/to/test/images \
     data={dataset.location}/data.yaml \
     imgsz=640 conf=0.1 save=True
   ```

---

## 📈 Results

| Metric         | Value   |
|----------------|---------|
| Precision      | 1.000 ✅ |
| Recall         | 0.996 ✅ |
| mAP@0.5        | 0.995 🟢 |
| mAP@0.5:0.95   | 0.608 🟡 |

Despite the small dataset size (50 images), the model achieved high performance, especially in `Precision` and `mAP@0.5`.

---

## 📊 Training Curve

> The model converged well. Some test images still have missed detections due to limited data diversity.

---

## 🧠 Improvements

- 🔄 Increase dataset size to >200 images
- 🧪 Try data augmentation like mosaic, blur, and crop
- 🎯 Experiment with larger models (`yolov11m`, `yolov11l`)
- 🧩 Use real-world noisy images for better generalization

---
---

## 📁 Folder Structure

```
Register-Number-Detetction-2/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

---

## ✍️ Author

- **Kavinraja D**
- Connect: [GitHub](https://github.com/d-kavinraja) | [LinkedIn](https://linkedin.com/in/d-kavinraja)
