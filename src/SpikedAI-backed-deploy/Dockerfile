# Use an official Python runtime as a base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system-level dependencies needed for OCR, PDF conversion, and graphics
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgl1-mesa-glx \
    libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Add the system binary path to ensure installed tools are in PATH
ENV PATH="/usr/local/bin:$PATH"

# Best practice: install PyTorch CPU-only separately for stability and smaller image size
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Ensure multipart support for FastAPI file uploads
RUN pip install --no-cache-dir python-multipart

# Copy application code into the container
COPY . .

# Document the port the container listens on
EXPOSE 8080

# Run the FastAPI app using uvicorn, hardcoding port 8080 (Cloud Run default)
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
