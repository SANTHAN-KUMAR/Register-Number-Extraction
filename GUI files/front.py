import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap, QImage
import cv2
import numpy as np
import os

# --- Assume these imports exist and the functions return NumPy arrays or None ---
try:
    from pdf_handler import handle_pdf_upload
    from camera_handler import capture_from_camera
    from scanner_handler import handle_scan_input
except ImportError as e:
    print(f"Error importing input handler functions: {e}")
    print("Please ensure pdf_handler.py, camera_handler.py, and scanner_handler.py are in the same directory.")
    sys.exit(1)

# --- Import Region Detection Function ---
try:
    from region_detection import detect_regions_and_extract
    MODEL_PROCESSING_AVAILABLE = True
except ImportError:
    MODEL_PROCESSING_AVAILABLE = False
    print("Warning: Your model processing file ('model.py') or function ('process_image_with_model') was not found.")
    print("The 'Proceed' button will be disabled.")
# --- End model import ---


# --- Import Subject Code Extraction ---
try:
    from sub_code_extraction import extract_subject_code_from_image
except ImportError as e:
    print(f"Error importing subject code extraction: {e}")
    extract_subject_code_from_region = None
    
# --- Import Register Number Extraction ---
try:
    from reg_no_extraction import extract_reg_no_from_image
except ImportError as e:
    print(f"Error importing subject code extraction: {e}")
    extract_subject_code_from_region = None

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Answer Sheet Processor")
        self.setGeometry(300, 150, 700, 600)  # Make window larger to fit image

        layout = QVBoxLayout()

        # Welcome Message
        title = QLabel("Answer Sheet Processor")
        title.setFont(QFont('Arial', 20))
        title.setAlignment(Qt.AlignCenter)

        # Buttons for Input Sources
        upload_pdf_btn = QPushButton("Upload PDF")
        upload_pdf_btn.setFixedHeight(50)
        upload_pdf_btn.clicked.connect(self.handle_upload_pdf)

        scan_btn = QPushButton("Scan Image")
        scan_btn.setFixedHeight(50)
        scan_btn.clicked.connect(self.handle_scan)

        use_camera_btn = QPushButton("Use Camera")
        use_camera_btn.setFixedHeight(50)
        use_camera_btn.clicked.connect(self.handle_use_camera)

        # --- Widget to display the image ---
        self.image_label = QLabel("Image will appear here")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(640, 480)  # Set a fixed size or adjust dynamically
        self.image_label.setStyleSheet("border: 1px solid black;")  # Optional: add a border

        # --- Proceed Button ---
        self.proceed_btn = QPushButton("Proceed")
        self.proceed_btn.setFixedHeight(50)
        self.proceed_btn.clicked.connect(self.handle_proceed)
        # Disable initially until an image is loaded, and if model processing is not available
        self.proceed_btn.setEnabled(False and MODEL_PROCESSING_AVAILABLE)  # Ensure it's false unless both conditions met

        # Add widgets to layout
        layout.addWidget(title)
        layout.addSpacing(20)
        layout.addWidget(upload_pdf_btn)
        layout.addWidget(scan_btn)
        layout.addWidget(use_camera_btn)
        layout.addSpacing(20)
        layout.addWidget(self.image_label)
        layout.addSpacing(20)
        layout.addWidget(self.proceed_btn)  # Add the proceed button

        self.setLayout(layout)

        # --- Instance variable to store the current image data ---
        self.current_image_data = None
        # --- End Instance Variable ---

    # --- Helper method to display a NumPy image array ---
    def display_image(self, image_array):
        """Converts a NumPy array (BGR or Grayscale) to QPixmap and displays it in image_label."""
        if image_array is None:
            self.image_label.clear()
            self.image_label.setText("No image data received.")
            return

        if not isinstance(image_array, np.ndarray):
            print(f"Error: display_image received non-NumPy object (Type: {type(image_array)}).")
            self.image_label.setText("Invalid image data format.")
            return

        # Ensure image data is uint8 for QImage
        if image_array.dtype != np.uint8:
            print(f"Note: Image data dtype is {image_array.dtype}, converting to uint8 for display.")
            display_array = image_array.astype(np.uint8)
        else:
            display_array = image_array

        # Get image dimensions and channels
        height, width = display_array.shape[:2]
        channels = 1 if len(display_array.shape) == 2 else display_array.shape[2]

        # Determine QImage format based on channels
        if channels == 3:
            bytes_per_line = channels * width
            # Use display_array.data if already uint8, otherwise use the converted one
            qimage = QImage(display_array.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        elif channels == 1:
            bytes_per_line = channels * width
            qimage = QImage(display_array.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
            print(f"Warning: Image has {channels} channels, unsupported for display.")
            self.image_label.setText(f"Unsupported image format ({channels} channels)")
            return

        # Convert QImage to QPixmap and scale it to fit the label
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Set the pixmap on the label
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setText("")

    # --- Helper method to update UI state based on image data ---
    def update_ui_state(self, image_data):
        self.current_image_data = image_data  # Store the image data

        # Display the image
        self.display_image(image_data)

        # Enable/Disable the Proceed button and update label text
        if image_data is not None and isinstance(image_data, np.ndarray):
            self.proceed_btn.setEnabled(MODEL_PROCESSING_AVAILABLE)  # Only enable if image is valid AND model is available
            self.image_label.setText("")  # Clear text if image is displayed
        else:
            self.proceed_btn.setEnabled(False)
            # The display_image function sets "No image data received." if None is passed

    # --- Modified slot methods to get image, update state, and display ---
    def handle_upload_pdf(self):
        print("Upload PDF button clicked")
        # handle_pdf_upload is expected to return NumPy array or None
        image_data = handle_pdf_upload(self)  # Pass 'self' as parent widget

        # --- Update UI state and display ---
        self.update_ui_state(image_data)

        if image_data is None:
            QMessageBox.warning(self, "Upload Failed", "Could not get image from PDF or upload cancelled.")
        else:
            # Show success message AFTER updating the state
            QMessageBox.information(self, "Upload Success", "First page from PDF loaded.")

    def handle_use_camera(self):
        print("Use Camera button clicked")
        # capture_from_camera is expected to return NumPy array or None
        image_data = capture_from_camera()  # This function handles its own display during capture

        # --- Update UI state and display ---
        self.update_ui_state(image_data)

        if image_data is None:
            QMessageBox.warning(self, "Capture Failed", "Could not capture image from camera or capture cancelled.")
        else:
            # Show success message AFTER updating the state
            QMessageBox.information(self, "Capture Success", "Image captured from camera.")

    def handle_scan(self):
        print("Scan Image button clicked")
        # handle_scan_input is expected to return the first page image (NumPy array) or None
        # and save the full PDF as a side effect.
        image_data = handle_scan_input()  # This function might show temporary windows depending on implementation

        # --- Update UI state and display ---
        self.update_ui_state(image_data)

        if image_data is None:
            QMessageBox.warning(self, "Scan Failed", "Could not scan image or scan cancelled.")
        else:
            # Show success message AFTER updating the state
            QMessageBox.information(self, "Scan Success", "First page from scanner loaded.")
            
    def handle_proceed(self):
        print("Proceed button clicked")

        if self.current_image_data is None or not isinstance(self.current_image_data, np.ndarray):
            QMessageBox.warning(self, "Cannot Proceed", "No image is currently loaded to proceed with.")
            print("No image data found to proceed.")
            return

        regn_weight_pth = "D:/reg_no_gui/weights.pt"
        subcode_model_path = "D:/reg_no_gui/best_subject_code_model.pth"
        regno_model_path = "D:/reg_no_gui/best_crnn_model(git).pth"

        print("Proceeding with region detection and extraction...")

        try:
            # Call the new detect_regions_and_extract function
            from region_detection import detect_regions_and_extract  # <- IMPORTANT: updated function name
            extracted_data = detect_regions_and_extract(
                self.current_image_data,
                regn_weight_pth,
                subcode_model_path,
                regno_model_path
            )

            if extracted_data:
                reg_number = extracted_data.get('reg_number', None)
                subject_code = extracted_data.get('subject_code', None)

                message = ""
                if reg_number:
                    message += f"Register Number: {reg_number}\n"
                else:
                    message += "Register Number: Not Found\n"

                if subject_code:
                    message += f"Subject Code: {subject_code}"
                else:
                    message += "Subject Code: Not Found"

                print(f"Extraction Completed:\n{message}")
                QMessageBox.information(self, "Extraction Result", message)

            else:
                QMessageBox.warning(self, "Extraction Failed", "No register number or subject code extracted.")

        except Exception as e:
            print(f"Error during processing: {e}")
            QMessageBox.critical(self, "Processing Error", f"Error occurred during region detection or extraction.\n{e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
