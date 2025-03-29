# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Disable Python buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (ffmpeg is needed by moviepy, and others support image/video processing)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the container at /app
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install the Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Download spaCy's small English model (used by your app)
RUN python -m spacy download en_core_web_sm

# Copy the rest of your application code into the container at /app
COPY . /app

# Expose the port that your app runs on
EXPOSE 8500

# Run the application with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8500"]
