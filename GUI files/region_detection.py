import os
import argparse
import cv2
import torch
from ultralytics import YOLO

# Import your subject code and register number extraction functions
from sub_code_extraction import load_subject_code_model, extract_subject_code_from_image
from reg_no_extraction import load_reg_no_model, extract_reg_no_from_image

# --- Updated Region Detection and Extraction ---
def detect_regions_and_extract(image_data, yolo_weights_path, subcode_model_path, regno_model_path):
    # Load YOLOv8 model
    try:
        model = YOLO(yolo_weights_path)
        print("[INFO] YOLOv8 model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load YOLOv8 model: {e}")
        return None

    if image_data is None:
        print("[ERROR] No image data provided.")
        return None

    # Create output folders
    subject_output_dir = "subject_code_crops"
    register_output_dir = "register_number_crops"
    os.makedirs(subject_output_dir, exist_ok=True)
    os.makedirs(register_output_dir, exist_ok=True)

    # Run YOLOv8 prediction
    results = model.predict(image_data, imgsz=640)
    detections = results[0]

    if detections.boxes is None:
        print("[WARNING] No detections found!")
        return None

    # Load extraction models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    subcode_model = load_subject_code_model(subcode_model_path, device)
    regno_model = load_reg_no_model(regno_model_path, device)

    extracted_data = {
        "reg_number": None,
        "subject_code": None
    }

    for idx, (box, cls) in enumerate(zip(detections.boxes.xyxy, detections.boxes.cls)):
        class_id = int(cls.item())

        x1, y1, x2, y2 = map(int, box.tolist())
        cropped_region = image_data[y1:y2, x1:x2]

        if class_id == 0:  # Register Number class
            crop_filename = os.path.join(register_output_dir, f"register_crop_{idx}.png")
            cv2.imwrite(crop_filename, cropped_region)
            print(f"[INFO] Cropped register number saved at {crop_filename}")

            reg_number_text = extract_reg_no_from_image(cropped_region, regno_model, device)
            extracted_data["reg_number"] = reg_number_text

        elif class_id == 1:  # Subject Code class
            crop_filename = os.path.join(subject_output_dir, f"subject_code_crop_{idx}.png")
            cv2.imwrite(crop_filename, cropped_region)
            print(f"[INFO] Cropped subject code saved at {crop_filename}")

            subject_code_text = extract_subject_code_from_image(cropped_region, subcode_model, device)
            extracted_data["subject_code"] = subject_code_text

    if extracted_data["reg_number"] is None:
        print("[WARNING] No register number detected.")

    if extracted_data["subject_code"] is None:
        print("[WARNING] No subject code detected.")

    return extracted_data


# --- For standalone testing ---
def main():
    parser = argparse.ArgumentParser(description="Region Detection and Extraction (Register Number & Subject Code)")
    parser.add_argument("--image", required=True, help="Path to input answer sheet image")
    parser.add_argument("--yolo-weights", required=True, help="Path to YOLOv8 weights")
    parser.add_argument("--subcode-model", required=True, help="Path to trained subject code model (.pth)")
    parser.add_argument("--regno-model", required=True, help="Path to trained register number model (.pth)")
    args = parser.parse_args()

    image_data = cv2.imread(args.image)

    extracted = detect_regions_and_extract(
        image_data,
        args.yolo_weights,
        args.subcode_model,
        args.regno_model
    )

    if extracted:
        print("\n--- Extraction Result ---")
        print(f"Register Number: {extracted['reg_number']}")
        print(f"Subject Code   : {extracted['subject_code']}")

if __name__ == "__main__":
    main()
