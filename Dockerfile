# Base image
FROM public.ecr.aws/docker/library/python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential gcc && rm -rf /var/lib/apt/lists/*

# Copy app files
COPY app/ /app

# Copy model files
COPY model/ /app/model

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8080

# Run the API
CMD ["python", "main.py"]
