# Use a lightweight official Python slim image
FROM python:3.11-slim

# Set environment variables to optimize Python execution in containers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .

# Install build dependencies, download tools, and clean up apt cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements (compiles llama-cpp-python)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Create models directory and download Gemma 4 E4B GGUF model
RUN mkdir -p /app/models && \
    curl -L -o /app/models/local_model.gguf \
    "https://huggingface.co/bartowski/google_gemma-4-E4B-it-GGUF/resolve/main/google_gemma-4-E4B-it-Q4_K_M.gguf?download=true"

# Copy application source code
COPY config.py .
COPY main.py .
COPY agent/ ./agent/
COPY models/router/ ./models/router/

# Ensure directories for input and output exist in the container
RUN mkdir -p /input /output

# Set the entrypoint to run the main execution pipeline
ENTRYPOINT ["python", "main.py"]
