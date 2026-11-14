# ==============================================================================
# GallNet Triage - Production Backend Dockerfile for Render / Cloud Containers
# ==============================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    PORT=8000 \
    HOME=/home/user \
    KERAS_HOME=/home/user/.cache/keras \
    MPLCONFIGDIR=/home/user/.cache/matplotlib \
    PATH=/home/user/.local/bin:$PATH

# Install minimal OS runtime libraries for OpenCV headless image processing & health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for secure container execution
RUN useradd -m -u 1000 user && \
    mkdir -p /app /home/user/.cache/keras /home/user/.cache/matplotlib && \
    chown -R user:user /app /home/user

WORKDIR /app

# Install Python requirements
COPY --chown=user:user backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy model checkpoint and backend source code
COPY --chown=user:user models/ /app/models/
COPY --chown=user:user backend/ /app/backend/

USER user
WORKDIR /app/backend

# Default port documentation
EXPOSE 8000

# Launch Uvicorn with dynamic $PORT environment variable support for Render
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
