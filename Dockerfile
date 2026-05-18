FROM python:3.10-slim-bookworm

WORKDIR /app

# Install standard gcc compiler headers for ONNX runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake quantized BiRefNet general weights into the build layer
RUN mkdir -p models && \
    apt-get update && apt-get install -y --no-install-recommends wget && \
    wget -q -O models/birefnet_general_quantized.onnx https://huggingface.co/briaai/BiRefNet/resolve/main/birefnet-general-epoch_244_quantized.onnx && \
    apt-get purge -y --auto-remove wget && rm -rf /var/lib/apt/lists/*

COPY app.py .

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
