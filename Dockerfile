# Use a lightweight official Python slim image
FROM python:3.11-slim

# Set environment variables to optimize Python execution in containers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY config.py .
COPY main.py .
COPY agent/ ./agent/

# Ensure directories for input and output exist in the container
RUN mkdir -p /input /output

# Set the entrypoint to run the main execution pipeline
ENTRYPOINT ["python", "main.py"]
