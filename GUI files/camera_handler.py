import cv2
import datetime
import os

def capture_from_camera():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot access the camera")
        return None

    print("🎥 Camera started. Press 'c' to capture, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Camera Feed - Press 'c' to Capture", frame)
        key = cv2.waitKey(1)

        if key == ord('c'):
            # Capture current frame
            captured_frame = frame.copy()
            # Hide the camera feed window temporarily
            cv2.destroyWindow("Camera Feed - Press 'c' to Capture")
            while True:
                cv2.imshow("Captured Image - Press 's' to Save, 'r' to Retake", captured_frame)
                sub_key = cv2.waitKey(0)
                if sub_key == ord('s'):
                    # Save the image
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"captured_image_{timestamp}.png"
                    
                    # Optionally save to an 'inputs' directory like before
                    output_dir = 'inputs'
                    os.makedirs(output_dir, exist_ok=True)
                    filename = os.path.join(output_dir, f"captured_image_{timestamp}.png")
                    
                    cv2.imwrite(filename, captured_frame)
                    print(f"✅ Image saved as {filename}")
                    cap.release()
                    cv2.destroyAllWindows()
                    return captured_frame
                elif sub_key == ord('r'):
                    print("🔄 Retaking photo...")
                    break
                elif sub_key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return None

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return None
