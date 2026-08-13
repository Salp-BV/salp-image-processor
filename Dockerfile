# Stage 1: Builder stage
FROM python:3.10-slim-bookworm AS builder

WORKDIR /app

# Install curl and ca-certificates for downloading model weights with retries
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake lightweight BiRefNet ONNX weights (224MB) with download retries
RUN mkdir -p models && \
    curl -sSL --retry 5 --retry-delay 2 --connect-timeout 30 \
    "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx" \
    -o models/birefnet_general_quantized.onnx

# Stage 2: Final minimal runtime stage
FROM python:3.10-slim-bookworm AS runner

WORKDIR /app

# Install curl, wget, and ca-certificates for Coolify health probes
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment and model from builder stage
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/models /app/models
COPY app.py .

# Environment configuration
ENV PATH="/opt/venv/bin:$PATH" \
    PORT=8080 \
    PYTHONUNBUFFERED=1 \
    OMP_WAIT_POLICY=PASSIVE \
    OMP_NUM_THREADS=4

EXPOSE 8080

# Health check probe for Coolify reverse proxy
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start Uvicorn bound to port 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
