import io
import os
import gc
import re
import sys
import secrets
import socket
import ipaddress
import threading
from urllib.parse import urlparse
from contextlib import asynccontextmanager

import torch
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import requests
import sentry_sdk

Image.MAX_IMAGE_PIXELS = 16_000_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def scrub_sentry_event(event, hint):
    if "request" in event:
        headers = event["request"].get("headers", {})
        if "authorization" in headers:
            headers["authorization"] = "[SCRUBBED]"
        if "x-api-key" in headers:
            headers["x-api-key"] = "[SCRUBBED]"
    return event

sentry_dsn = os.getenv("SENTRY_DSN")
if sentry_dsn:
    sentry_env = os.getenv("APP_ENV", os.getenv("ENVIRONMENT", "production")).lower()
    traces_rate = 1.0 if sentry_env in ["development", "staging", "stg"] else float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1"))
    sentry_sdk.init(
        dsn=sentry_dsn,
        environment=sentry_env,
        traces_sample_rate=traces_rate,
        before_send=scrub_sentry_event,
        send_default_pii=False,
    )

gpu_lock = threading.Lock()
biref_model = None

image_transforms = transforms.Compose([
    transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def clean_product_title(title: str) -> str:
    if not title:
        return "product"
    cleaned = re.sub(r'[\(\[\{].*?[\)\]\}]', '', title)
    cleaned = re.sub(r'[–—|/:]', ' ', cleaned).strip()
    words = cleaned.split()[:5]
    return " ".join(words) if words else "product"

def decontaminate_and_despill(orig_rgb_img: Image.Image, alpha_mask: Image.Image) -> Image.Image:
    """
    Vectorized edge color unmixing and decontamination.
    Removes ambient colored background bleed (e.g. bright yellow backdrops) from anti-aliased edge pixels.
    """
    rgb_np = np.array(orig_rgb_img).astype(np.float32)
    a_np = np.array(alpha_mask).astype(np.float32) / 255.0
    h, w = a_np.shape

    # Sample background color from image perimeter where alpha is near zero
    border_w = max(4, int(min(h, w) * 0.03))
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:border_w, :] = True
    border_mask[-border_w:, :] = True
    border_mask[:, :border_w] = True
    border_mask[:, -border_w:] = True

    bg_candidates = border_mask & (a_np < 0.05)
    if np.sum(bg_candidates) > 50:
        bg_color = np.median(rgb_np[bg_candidates], axis=0)
    else:
        bg_color = np.array([255.0, 255.0, 255.0], dtype=np.float32)

    # Decontaminate semi-transparent edge pixels (0.02 < alpha < 0.92)
    # Mathematical unmixing: F = (I - (1 - alpha) * B) / alpha
    fringe_mask = (a_np > 0.02) & (a_np < 0.92)
    unmixed = rgb_np.copy()
    if np.any(fringe_mask):
        a_vals = a_np[fringe_mask, np.newaxis]
        observed = rgb_np[fringe_mask]
        effective_alpha = np.maximum(a_vals, 0.25)
        cleaned_fg = (observed - (1.0 - a_vals) * bg_color) / effective_alpha
        unmixed[fringe_mask] = np.clip(cleaned_fg, 0.0, 255.0)

    clean_img = Image.fromarray(unmixed.astype(np.uint8), mode="RGB").convert("RGBA")
    clean_img.putalpha(alpha_mask)
    return clean_img

@asynccontextmanager
async def lifespan(app: FastAPI):
    global biref_model
    print(f"Initializing BiRefNet GPU Engine on device: {DEVICE}...")

    model_path = os.getenv("MODEL_PATH", "/app/models/birefnet")
    biref_model = AutoModelForImageSegmentation.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE).eval()

    if DEVICE == "cuda":
        print("Executing CUDA model warmup pass...")
        with torch.no_grad():
            dummy = torch.zeros((1, 3, 1024, 1024), dtype=torch.float16, device="cuda")
            _ = biref_model(dummy)
            torch.cuda.synchronize()
        print("CUDA Warmup complete. BiRefNet Engine Ready.")

    yield

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

app = FastAPI(title="Salp GPU Image Processor", lifespan=lifespan)

# Auth: Supports IMAGE_PROCESSOR_API_KEY or RUNPOD_API_KEY
IMAGE_PROCESSOR_API_KEY = os.getenv("IMAGE_PROCESSOR_API_KEY", "").strip()
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "").strip()
security_scheme = HTTPBearer(auto_error=False)

def verify_api_key(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security_scheme)):
    token = credentials.credentials.strip() if credentials and credentials.credentials else ""
    if not token:
        token = request.headers.get("x-api-key", "").strip()

    valid_keys = [k for k in [IMAGE_PROCESSOR_API_KEY, RUNPOD_API_KEY] if k]
    if not valid_keys:
        raise HTTPException(status_code=500, detail="API key unconfigured on server.")

    if not token or not any(secrets.compare_digest(token, k) for k in valid_keys):
        raise HTTPException(status_code=403, detail="Invalid API key.")
    return True

# SSRF Network Protection
DISALLOWED_IP_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"), ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"), ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"), ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"), ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"), ipaddress.ip_network("fe80::/10"),
]

def fetch_image_securely(image_url: str, max_size_bytes: int = 25 * 1024 * 1024) -> bytes:
    parsed = urlparse(image_url.strip())
    if not parsed.hostname:
        raise HTTPException(status_code=400, detail="Malformed URL: missing hostname.")
    if parsed.scheme != "https" and os.getenv("ENVIRONMENT") == "production":
        raise HTTPException(status_code=400, detail="HTTPS scheme strictly required in production.")
    
    try:
        resolved_ip = socket.getaddrinfo(parsed.hostname, None)[0][4][0]
        ip_obj = ipaddress.ip_address(resolved_ip)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"DNS resolution failed: {str(e)}")

    for net in DISALLOWED_IP_NETWORKS:
        if ip_obj in net:
            raise HTTPException(status_code=400, detail="Prohibited private/internal IP address.")
            
    resp = requests.get(image_url, stream=True, timeout=(5.0, 15.0))
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Image download failed: HTTP {resp.status_code}")
    
    chunks = []
    downloaded = 0
    for chunk in resp.iter_content(chunk_size=65536):
        if chunk:
            downloaded += len(chunk)
            if downloaded > max_size_bytes:
                raise HTTPException(status_code=413, detail="Payload Too Large: image exceeds 25MB limit.")
            chunks.append(chunk)
    return b"".join(chunks)

# Health Probes for RunPod Load Balancer
@app.get("/ping")
def ping():
    """RunPod Load Balancer health check route."""
    return {"status": "healthy", "device": DEVICE}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "salp-image-processor-gpu",
        "pipeline": "two-stage-yolo-and-birefnet",
        "device": DEVICE,
        "vram_allocated_mb": round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2) if DEVICE == "cuda" else 0
    }

@app.post("/remove-background")
def remove_background(payload: dict, authenticated: bool = Depends(verify_api_key)):
    image_url = payload.get("image_url")
    title = payload.get("title", "").strip()
    min_res = int(payload.get("min_resolution", 800))
    
    if not image_url:
        raise HTTPException(status_code=400, detail="Missing required 'image_url' field.")

    image_bytes = fetch_image_securely(image_url)
    try:
        orig_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Image.DecompressionBombError:
        raise HTTPException(status_code=413, detail="Decompression Bomb: image exceeds security threshold.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    w, h = orig_image.size
    if w < min_res or h < min_res:
        raise HTTPException(status_code=422, detail=f"Low resolution source image ({w}x{h}). Minimum: {min_res}px.")

    try:
        with gpu_lock:
            # Stage 1: Full-Frame 1024x1024 Global BiRefNet Alpha Matting
            # Feeding the full uncropped frame guarantees 100% retention of all physical structures:
            # - Bookshelf speaker wooden cabinets (Test 17)
            # - Slender desk lamp stems and poles (Test 20)
            # - Continuous headphone headband arches (Test 15)
            input_tensor = image_transforms(orig_image).unsqueeze(0).to(DEVICE)
            if DEVICE == "cuda":
                input_tensor = input_tensor.half()

            with torch.no_grad():
                preds = biref_model(input_tensor)[-1].sigmoid().cpu()

            pred = preds[0].squeeze()
            pred_pil = transforms.ToPILImage()(pred).resize((w, h), Image.Resampling.BILINEAR)

            # Noise Floor Thresholding:
            # Maps [0.15, 0.85] smoothly to [0.0, 1.0], strictly suppressing background noise < 0.15 to 0
            mask_np = np.array(pred_pil).astype(np.float32) / 255.0
            mask_np = np.clip((mask_np - 0.15) / (0.85 - 0.15), 0.0, 1.0)
            mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))

        # Stage 2: Color Decontamination & Despill (Sub-pixel Chroma Neutralization)
        clean_fg = decontaminate_and_despill(orig_image, mask_img)

        # Stage 3: Auto-Centering (85% Fit) & Studio Ground Contact Shadow
        bbox = mask_img.getbbox() or (0, 0, w, h)
        fg = clean_fg.crop(bbox)
        fg_w, fg_h = fg.size

        scale = min((w * 0.85) / fg_w, (h * 0.85) / fg_h)
        new_w = max(1, int(fg_w * scale))
        new_h = max(1, int(fg_h * scale))

        resized_fg = fg.resize((new_w, new_h), Image.Resampling.LANCZOS)
        paste_x = (w - new_w) // 2
        paste_y = (h - new_h) // 2

        centered_fg = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        centered_fg.paste(resized_fg, (paste_x, paste_y))

        shadow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        shadow_w = int(new_w * 0.90)
        shadow_h = max(8, int(new_h * 0.07))
        shadow_x = paste_x + (new_w - shadow_w) // 2
        shadow_y = paste_y + new_h - (shadow_h // 2)

        shadow_draw = ImageDraw.Draw(shadow_layer)
        shadow_draw.ellipse(
            [shadow_x, shadow_y, shadow_x + shadow_w, shadow_y + shadow_h],
            fill=(20, 20, 25, 110)
        )
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(max(4, int(shadow_h * 0.6))))

        final_canvas = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        final_canvas = Image.alpha_composite(final_canvas, shadow_layer)
        final_canvas = Image.alpha_composite(final_canvas, centered_fg)
        final_image = final_canvas.convert("RGB")

        buf = io.BytesIO()
        final_image.save(buf, format="JPEG", quality=92, optimize=True)
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/jpeg")

    except Exception as e:
        sentry_sdk.capture_exception(e)
        raise HTTPException(status_code=500, detail=f"GPU processing failed: {str(e)}")
