# Silver Price Prediction - Dockerfile
FROM python:3.10-slim-buster

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . ./

# Expose port (HF Spaces uses 7860)
EXPOSE 7860

# Environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PORT=7860

# Run the application
CMD ["python", "app.py"]
