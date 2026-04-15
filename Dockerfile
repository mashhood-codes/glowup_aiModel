# Use lightweight Python 3.10 base image
FROM python:3.10-slim-bullseye

# Set work directory
WORKDIR /app

# Install system dependencies required for CV operations in headless mode
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements_streamlit.txt .

# Install PyTorch CPU-only version for smaller image size and faster deployment
# Then install other requirements
RUN pip install --no-cache-dir torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements_streamlit.txt

# Copy application code
COPY . .

# Expose ports
EXPOSE 8000

# Health check for Render
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run with uvicorn (production-ready, proxy headers enabled for HTTPS)
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
