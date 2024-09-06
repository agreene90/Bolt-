# Use a PyTorch base image with CUDA and cuDNN support for GPU acceleration
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

# Install essential packages
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev build-essential \
    libssl-dev libffi-dev python3-setuptools curl \
    i2c-tools libopencv-dev

# Install Python packages for the project
RUN pip install micropython-upip torch torchvision torchaudio snntorch numpy opencv-python-headless matplotlib

# Create a working directory
WORKDIR /app

# Copy the project files to the container
COPY . /app

# Set environment variables for CUDA
ENV CUDA_VISIBLE_DEVICES=0,1  # Support for multi-GPU setups

# Expose any necessary ports (e.g., for data visualization)
EXPOSE 8888

# Entry point to run the main script (e.g., real-time EEG processing)
ENTRYPOINT ["python", "main.py"]

# Optional: Add volume mounting for persistent storage (EEG data, logs)
VOLUME ["/data"]