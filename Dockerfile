FROM python:3.10-slim-bookworm

WORKDIR /app

# Install standard C++ dependencies for ONNX runtime, curl, and wget for Coolify health probes
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake lightweight BiRefNet ONNX weights into the build layer (224MB)
RUN mkdir -p models && \
    curl -L "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx" -o models/birefnet_general_quantized.onnx

COPY app.py .

# Explicitly bind to port 8080 to match Coolify reverse proxy mapping
ENV PORT=8080
EXPOSE 8080

# Health check probe for Coolify and Docker inspect
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Hardcode --port 8080 so build-arg PORT=8000 cannot override uvicorn port
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
