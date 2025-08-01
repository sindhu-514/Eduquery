# Use Python base image
FROM python:3.11-slim

# System dependencies for pdf2image and pytesseract
RUN apt-get update && \
    apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8005

# Run the app
CMD ["python", "trial2.py"]
