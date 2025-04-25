# Use an official NVIDIA PyTorch image as a base
# This includes CUDA, CuDNN, and PyTorch pre-installed
# Choose a tag that suits your needs, e.g., a recent stable one
# Find more tags here: https://hub.docker.com/r/nvcr/pytorch/tags
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
# Use --no-cache-dir to avoid storing cache data, reducing image size
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your script, YOLO weights, and CRNN model into the container
COPY script.py .
COPY weights.pt .
COPY best_crnn_model.pth .

# Create directories for output. This matches the paths in your script.
# You might want to mount volumes for these later if you need persistence.
RUN mkdir -p cropped_register_numbers cropped_subject_codes results

# Define the default command to run your script
# This allows you to pass arguments when you run the container,
# for example, the path to the input image.
# CMD ["python", "script.py"]

# You might want to expose a volume for output later, but keeping it simple for now
# VOLUME /app/results
# VOLUME /app/cropped_register_numbers
# VOLUME /app/cropped_subject_codes
