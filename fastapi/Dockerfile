# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install rclone and fuse3
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    fuse3 \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://rclone.org/install.sh | bash

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create mount point
ENV RCLONE_CONFIG=/app/rclone.conf
ENV MOUNT_DIR=/mnt/chi_data
RUN mkdir -p ${MOUNT_DIR}

# Copy rclone configuration and entrypoint script
COPY rclone.conf ${RCLONE_CONFIG}
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Copy the application code
COPY . .

# (Fix filename below)
COPY crash_model.joblib /app/

# Expose FastAPI port
EXPOSE 8000

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
# Run the FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
