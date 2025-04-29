from PyQt5.QtWidgets import QFileDialog, QMessageBox
import fitz # PyMuPDF
import numpy as np # Import numpy for image data handling
# You'll need to install numpy: pip install numpy
# Keep show_result as it might be called by the code that receives the image data
import cv2 # OpenCV for image processing

def handle_pdf_upload(parent_widget):
        """
        Opens a file dialog to select a PDF, extracts the first page as image data,
        and returns the image data as a NumPy arr       
        Args:
                parent_widget: The parent widget for the file dialog and message boxes.

        Returns:
                numpy.ndarray: The image data of the first page as a NumPy array (BGR format compatible with OpenCV),
                if a file is selected and processed successfully.
         None: If the file dialog is cancelled or an error occurs during processing.
        """
        file_path, _ = QFileDialog.getOpenFileName(
        parent_widget, "Select Answer Sheet PDF", "", "PDF Files (*.pdf)"
        )

    # If the user cancels the file dialog, file_path will be an empty string
        if not file_path:
            print("PDF upload cancelled.")
            return None # Return None if dialog is cancelled

        try:
                doc = fitz.open(file_path)
                if doc.page_count == 0:
                        QMessageBox.warning(parent_widget, "Warning", "Selected PDF has no pages.")
                        doc.close() # Close the document
                        return None        
                first_page = doc.load_page(0) # Load the first page (index 0)
                # Render the page to a pixmap (image representation)
                # Using a reasonable DPI like 200 or 300 is good for extracting text/details
                pix = first_page.get_pixmap(dpi=200)
                # --- Conversion to NumPy Array ---
                # fitz pixmaps store data byte-by-byte (RGB or RGBA)
                # NumPy arrays for images are typically Height x Width x Channels
                # Channels order in pix.samples is usually RGB or RGBA
                # OpenCV expects BGR, so we might need to convert if pix.n is 3 or 4

                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape([pix.height, pix.width, pix.n])

                # Convert RGB or RGBA to BGR if necessary (OpenCV default)
                # fitz pix.n == 3 is RGB, pix.n == 4 is RGBA
                if pix.n == 3:
                     # Convert RGB to BGR
                     img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR) # Requires cv2 if not already imported
                elif pix.n == 4:
                     # Convert RGBA to BGR (discarding alpha)
                     img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR) # Requires cv2 if not already imported
                # If pix.n == 1 (grayscale), it's already compatible or needs conversion to BGR if cv2 ops require 3 channels
                # For grayscale (pix.n == 1), reshape is [pix.height, pix.width]
                # img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape([pix.height, pix.width]) # for grayscale
                # If grayscale needs to be 3-channel BGR for cv2:
                # img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)


                # --- Removed Saving and Message Boxes ---
                # output_image_path = "first_page.png"
                # pix.save(output_image_path) # <-- Removed this line
                # QMessageBox.information(parent_widget, "Success", f"First page saved as {output_image_path}") # <-- Removed this line
                # show_result(parent_widget, "Pending...", "Pending...") # <-- Removed this line

                # --- Return the image data ---
                doc.close() # Close the document before returning
                return img_array # Return the NumPy array containing the image data

        except FileNotFoundError:
                QMessageBox.critical(parent_widget, "Error", "File not found.")
                return None
        except Exception as e:
                # Catch other potential errors during PDF processing (corrupt file, etc.)
                QMessageBox.critical(parent_widget, "Error", f"Failed to process PDF: {e}")
                # Ensure document is closed on error if it was opened
                if 'doc' in locals() and doc and not doc.is_closed:
                        doc.close()
                return None


