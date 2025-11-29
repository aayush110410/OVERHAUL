# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SUMO_HOME=/usr/share/sumo

# Install system dependencies (including SUMO and git)
RUN apt-get update && apt-get install -y \
    sumo \
    sumo-tools \
    sumo-doc \
    git \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# Note: We use the CPU-only version of PyTorch to save space and RAM
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
