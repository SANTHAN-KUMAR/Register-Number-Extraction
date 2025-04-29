import platform
import os
from PIL import Image # Import Pillow for image handling and PDF creation
import io # Needed if scanned data is returned as bytes
import time

# --- Move OS-specific imports to the top ---
try:
    import wia_scan
    WIA_AVAILABLE = True
    # WIA Constants for Document Handling Source (Common WIA Property ID: 6146)
    # Values can vary, but these are standard WIA values.
    # wia-scan might provide wrappers or constants. Using raw IDs/values as assumption.
    WIA_DPS_DOCUMENT_HANDLING_SELECT = 6146
    WIA_FEEDER = 1
    WIA_FLATBED = 2
except ImportError:
    wia_scan = None # Set to None if import fails
    WIA_AVAILABLE = False
    # print("Warning: wia-scan library not found. Windows scanning will be disabled.") # Optional warning

try:
    import sane # type: ignore
    SANE_AVAILABLE = True
    # SANE Status flags (python-sane might have these as constants or ints)
    # Using common integer values or assuming sane module constants exist
    SANE_STATUS_GOOD = sane.SANE_STATUS_GOOD if SANE_AVAILABLE else 0 # No error, scan good
    SANE_STATUS_NO_DOCS = sane.SANE_STATUS_NO_DOCS if SANE_AVAILABLE else 1 # Feeder empty
    # Add other status flags if needed (e.g., SANE_STATUS_DEVICE_ERROR)
except ImportError:
    sane = None # Set to None if import fails
    SANE_AVAILABLE = False
    # print("Warning: python-sane library not found. Linux scanning will be disabled.") # Optional warning

# --- End of top-level imports ---


def handle_scan_input():
    """
    Handles initiating a scan based on the current operating system.
    Scans all pages from ADF and saves as a single PDF.
    """
    current_os = platform.system()
    if current_os == 'Windows':
        if not WIA_AVAILABLE:
            raise ImportError("The 'wia-scan' library is required for Windows but is not installed. Install with 'pip install wia-scan'")
        return scan_windows_adf_to_pdf()
    elif current_os == 'Linux':
        if not SANE_AVAILABLE:
            raise ImportError("The 'python-sane' library is required for Linux but is not installed. Install with 'pip install python-sane'")
        return scan_linux_adf_to_pdf()
    elif current_os == 'Darwin':
        raise NotImplementedError("Scanner integration for macOS not implemented yet.")
    else:
        raise Exception(f"Unsupported OS: {current_os}")

def scan_windows_adf_to_pdf():
    """
    Scans all pages from a WIA scanner's ADF and saves them as a single PDF.
    Returns the path to the generated PDF file.
    """
    # wia_scan is imported at the top and checked in handle_scan_input

    try:
        print("Attempting multi-page WIA scan from ADF...")
        manager = wia_scan.get_device_manager()
        devices = wia_scan.list_devices(manager)

        if not devices:
            raise Exception("No WIA scanner device found. Please ensure a scanner is connected, drivers are installed, and it has an ADF.")

        # Select the first device found (you might need logic here to choose)
        device_info = devices[0]
        print(f"Using scanner: {device_info[1]}")

        # --- Configure for ADF scanning ---
        # This is highly dependent on the wia-scan library's API.
        # A common WIA approach is to set a property on the device or item.
        # WIA_DPS_DOCUMENT_HANDLING_SELECT (ID 6146) set to WIA_FEEDER (Value 1)
        # Assuming device_info is sufficient or we need to get a connectable device object
        # Let's assume wia_scan allows setting properties on the device object obtained somehow
        # Or maybe scan_single_side_main takes a 'source' argument? Let's try setting property first.

        # Get a connectable device object if device_info is just info
        # Assuming connect_to_device_by_uid takes manager and device_id (first element of tuple)
        try:
            device = wia_scan.connect_to_device_by_uid(manager, device_info[0])
        except Exception as e:
            raise Exception(f"Failed to connect to WIA device {device_info[1]}: {e}") from e

        print("Setting scan source to ADF...")
        # Attempt to set the source property to ADF (ID 6146, Value 1)
        # Assuming the library provides access to properties by ID and allows setting value
        try:
            # Find the property for Document Handling Source
            doc_handling_prop = None
            # Iterate through properties to find the source one by its ID
            for prop in device.properties:
                 if prop.id == WIA_DPS_DOCUMENT_HANDLING_SELECT: # Use the known WIA ID
                     doc_handling_prop = prop
                     break

            if doc_handling_prop is None:
                 print("Warning: Could not find Document Handling Source property (ID 6146). Cannot force ADF source via property.")
                 # Proceeding might use default source, which might not be ADF

            else:
                 # Check available values for the property if possible
                 # print(f"Available values for {doc_handling_prop.name}: {doc_handling_prop.get_available_values()}") # Hypothetical method

                 # Set the value to WIA_FEEDER (Value 1)
                 doc_handling_prop.value = WIA_FEEDER # Use the known WIA value
                 print("Document Handling Source set to Feeder (ADF).")


        except Exception as e:
            print(f"Warning: Failed to set WIA source property: {e}. Proceeding, but scan might use default source.")


        # --- Initiate Multi-Page Scan ---
        # This is the most speculative part without exact wia-scan docs.
        # Does scan_single_side_main, after setting source, return a LIST? Unlikely given the name.
        # Is there another function? Or does a scan method on the device object itself handle it?
        # Let's try calling device.scan() assuming it exists and can handle ADF when source is set.
        # AND assuming it returns a LIST of PIL-like image objects if ADF is used. This is a BIG assumption.

        print("Starting multi-page scan...")
        # Try calling a scan method on the connected device object
        # Assuming device.scan() or similar method exists and returns a list of images for ADF
        try:
             # This method call is highly dependent on the wia-scan library's Device object API
             # If device.scan() doesn't exist or doesn't return a list for ADF, this will fail.
            scanned_images = device.scan() # Hypothetical call
            print(f"Scan initiated. Received {len(scanned_images)} images.")

            # If the library's scan method returns image data (bytes) instead of PIL objects,
            # convert them. Assume it returns PIL objects for now.
            # If it returns bytes:
            # pil_images = []
            # for img_data in scanned_images:
            #     try:
            #         pil_images.append(Image.open(io.BytesIO(img_data)))
            #     except Exception as img_e:
            #          print(f"Warning: Could not open scanned data as image: {img_e}")
            # scanned_images = pil_images # Use the list of PIL images

        except Exception as e:
            # Catch errors during the scan process itself
            raise Exception(f"Error during WIA scan acquisition from ADF: {e}") from e

        if not scanned_images:
            raise Exception("WIA scan completed but returned no images.")


        # --- Save images as a single PDF ---
        print("Saving images as single PDF...")
        output_dir = 'inputs'
        os.makedirs(output_dir, exist_ok=True)
        # Use a dynamic name or timestamp for the PDF
        pdf_filename = f"scanned_document_{int(time.time())}.pdf" # Using timestamp
        pdf_path = os.path.join(output_dir, pdf_filename)

        # Ensure all items in scanned_images are PIL Image objects
        # (If device.scan() returned something else, the conversion logic above is needed)
        if not all(hasattr(img, 'save') for img in scanned_images):
             raise TypeError("Scanned items are not saveable image objects. PDF creation failed.")

        # Save the first image, appending the rest
        try:
            if len(scanned_images) == 1:
                scanned_images[0].save(pdf_path, format='PDF') # Save as PDF if only one page
            else:
                scanned_images[0].save(pdf_path, save_all=True, append_images=scanned_images[1:], format='PDF') # Save multiple pages
            print(f"PDF saved successfully to {pdf_path}")

        except Exception as e:
            raise Exception(f"Error saving images as PDF: {e}") from e

        # --- Return the FIRST image object ---
        # This is the change: Return the first element from the list of images
        print("Returning the first image object.")
        return scanned_images[0] # Return the first image object

    except Exception as e:
        # Catch any other errors during WIA interaction
        raise Exception(f"An error occurred during WIA multi-page scanning: {e}") from e

def scan_linux_adf_to_pdf():
    """
    Scans all pages from a SANE scanner's ADF and saves them as a single PDF.
    Returns the path to the generated PDF file.
    """
    # sane is imported at the top and checked in handle_scan_input

    try:
        print("Attempting multi-page SANE scan from ADF...")
        sane.init()
        # Ensure SANE is exited even if errors occur later
        try:
            devices = sane.get_devices()

            if not devices:
                raise Exception("No SANE scanner found. Please ensure a scanner is connected, configured for SANE, and has an ADF.")

            # Select the first device
            dev_name = devices[0][0]
            dev = sane.open(dev_name)

            # Ensure device is closed even if errors occur later
            try:
                # --- Configure for ADF scanning ---
                print("Setting scan source to ADF...")
                # SANE options are accessed via dev.options dictionary or set_option/set_option_value
                try:
                    if 'source' in dev.options:
                        dev.options['source'].value = 'ADF' # Set source option to ADF
                        print("SANE source option set to 'ADF'.")
                    else:
                        print("Warning: SANE 'source' option not found for this scanner. Cannot force ADF source via option.")
                        # Proceeding might use default source

                    # You might also want to set duplex scanning if supported and needed:
                    # if 'duplex' in dev.options:
                    #     dev.options['duplex'].value = True
                    #     print("SANE duplex option set to True.")

                except Exception as e:
                    print(f"Warning: Failed to set SANE source option: {e}. Proceeding, but scan might use default source.")


                # --- Initiate Multi-Page Scan ---
                # For ADF, you typically start the scan and then snap pages
                # until the scanner status indicates no more documents in the feeder.
                print("Starting multi-page scan...")
                dev.start() # Start the scan process (pre-feed, warm-up etc.)

                scanned_images = []
                page_count = 0

                # Loop to snap images until the feeder is empty or an error occurs
                while True:
                    try:
                        page_count += 1
                        print(f"Attempting to snap page {page_count}...")
                        im = dev.snap() # Snap one image

                        # Check if snap returned a valid image object
                        if not hasattr(im, 'save'):
                             print(f"Warning: SANE snap() for page {page_count} did not return a saveable image object. Stopping scan loop.")
                             break # Stop if snap fails or doesn't return image

                        scanned_images.append(im)
                        print(f"Snapped page {page_count}.")

                        # Check scanner status to see if more documents are available
                        # This status check method might vary slightly in python-sane versions
                        # Assuming sane.get_status(dev) returns (status_code, status_string)
                        status, _ = sane.get_status(dev)

                        if status == SANE_STATUS_NO_DOCS:
                            print("SANE status: No more documents in feeder.")
                            break # Feeder is empty, stop scanning
                        elif status != SANE_STATUS_GOOD:
                            print(f"SANE status: Received non-GOOD status ({status}). Stopping scan loop.")
                            # Handle other potential error statuses here if needed
                            break # Stop on other statuses

                        # If status is SANE_STATUS_GOOD, the loop continues to snap the next page

                    except Exception as e:
                        print(f"Error snapping page {page_count}: {e}. Stopping scan loop.")
                        break # Break loop on any snapping error

                print(f"Multi-page scan complete. Collected {len(scanned_images)} images.")


                if not scanned_images:
                    raise Exception("SANE scan completed but collected no images.")


                # --- Save images as a single PDF ---
                print("Saving images as single PDF...")
                output_dir = 'inputs'
                os.makedirs(output_dir, exist_ok=True)
                # Use a dynamic name or timestamp for the PDF
                pdf_filename = f"scanned_document_{int(os.time())}.pdf" # Using timestamp
                pdf_path = os.path.join(output_dir, pdf_filename)

                # Ensure all items in scanned_images are PIL Image objects
                if not all(hasattr(img, 'save') for img in scanned_images):
                     raise TypeError("Scanned items are not saveable image objects. PDF creation failed.")


                try:
                    if len(scanned_images) == 1:
                         scanned_images[0].save(pdf_path, format='PDF') # Save as PDF if only one page
                    else:
                        # Save the first image, appending the rest
                        scanned_images[0].save(pdf_path, save_all=True, append_images=scanned_images[1:], format='PDF') # Save multiple pages
                    print(f"PDF saved successfully to {pdf_path}")

                except Exception as e:
                    raise Exception(f"Error saving images as PDF: {e}") from e

                print("Returning the first image object.")
                return scanned_images[0] # Return the first image object

            finally:
                # Ensure device is closed
                if 'dev' in locals() and dev:
                    dev.close()

        finally:
            # Ensure SANE is exited
            sane.exit()

    except Exception as e:
        # Catch any other errors during SANE interaction
        raise Exception(f"An error occurred during SANE multi-page scanning: {e}") from e


def scan_macos():
    raise NotImplementedError("Scanner integration for macOS not implemented yet.")
