# Use Ubuntu 22.04 as base for better PPA support (SUMO is best supported here)
FROM ubuntu:22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV SUMO_HOME=/usr/share/sumo

# Install system dependencies
# 1. Add SUMO and Python PPAs
# 2. Install SUMO, Python 3.11, and build tools
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    git \
    && add-apt-repository ppa:sumo/stable \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    sumo \
    sumo-tools \
    sumo-doc \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    build-essential \
    libgl1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# Use python3.11 explicitly
RUN python3.11 -m pip install --no-cache-dir --upgrade pip && \
    python3.11 -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    python3.11 -m pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port
EXPOSE 8000

# Run with python3.11 module
CMD ["python3.11", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
