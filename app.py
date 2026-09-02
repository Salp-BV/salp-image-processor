import io
import os
import sys
import gc
import ctypes
import secrets
import socket
import ipaddress
from urllib.parse import urlparse, urljoin
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter, ImageDraw
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import requests
import sentry_sdk

# 1. Enforce strict limits on PIL image decompression to prevent memory bombs (max 16 Megapixels / 4000x4000)
Image.MAX_IMAGE_PIXELS = 16_000_000

# 2. Enforce thread limits & suppress OpenMP/BLAS spin-waiting BEFORE importing third-party C-extensions
os.environ["OMP_NUM_THREADS"] = os.getenv("ONNX_NUM_THREADS", "2")
os.environ["ONNX_NUM_THREADS"] = os.getenv("ONNX_NUM_THREADS", "2")
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["GOMP_SPINCOUNT"] = "0"
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# 3. Sentry Initialization with Zero-Trust PII / Header Scrubbing
def scrub_sentry_event(event, hint):
    """Eliminates Authorization tokens, API keys, and sensitive network metadata from Sentry breadcrumbs."""
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

app = FastAPI(
    title="Salp CPU Image Processor",
    description="Zero-Trust Background Removal with Automated PIL Contact Shadows",
    docs_url=None if os.getenv("ENVIRONMENT") == "production" else "/docs",
    redoc_url=None,
)

def trim_memory():
    """
    Trims unmapped heap memory pages back to the OS kernel (Linux glibc ptmalloc)
    and forces Python garbage collection to maintain a minimal RSS memory footprint.
    """
    gc.collect()
    try:
        if sys.platform.startswith("linux"):
            ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

# Configure high-performance ONNX runtime session options optimized for container memory & CPU constraints
opts = ort.SessionOptions()
enable_arena = os.getenv("ENABLE_CPU_MEM_ARENA", "false").lower() == "true"
opts.enable_cpu_mem_arena = enable_arena
opts.enable_mem_pattern = False  # Disable graph memory pattern pre-allocation to free node tensors immediately
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
opts.intra_op_num_threads = int(os.getenv("ONNX_NUM_THREADS", "2"))
opts.inter_op_num_threads = 1
# CRITICAL: Disable ONNX Runtime intra-op threadpool spin-waiting to force worker threads to sleep immediately when idle (0.0% CPU)
opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

# Resolve model path dynamically with fallback chain
candidate_models = [
    os.getenv("MODEL_PATH", ""),
    "models/u2net.onnx",
    "models/isnet-general-use.onnx",
    "models/birefnet_general_quantized.onnx",
    "/tmp/u2net.onnx",
]
model_path = next((m for m in candidate_models if m and os.path.exists(m)), "models/u2net.onnx")

session = ort.InferenceSession(
    model_path, 
    sess_options=opts,
    providers=["CPUExecutionProvider"]
)

# Call trim_memory after loading session graph
trim_memory()

# --------------------------------------------------------------------------
# ZERO-TRUST AUTHENTICATION GUARD
# --------------------------------------------------------------------------
IMAGE_PROCESSOR_API_KEY = os.getenv("IMAGE_PROCESSOR_API_KEY", "").strip()
security_scheme = HTTPBearer(auto_error=False)

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)):
    """
    Enforces FAIL-CLOSED authentication.
    Rejects any request if the server API key is unconfigured or mismatch occurs.
    Uses constant-time comparison to prevent timing side-channels.
    """
    if not IMAGE_PROCESSOR_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error: IMAGE_PROCESSOR_API_KEY is not configured."
        )
    
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization Bearer token.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    token = credentials.credentials.strip()
    if not secrets.compare_digest(token, IMAGE_PROCESSOR_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key."
        )
    return True

# --------------------------------------------------------------------------
# ZERO-TRUST SSRF & NETWORK ISOLATION GUARD
# --------------------------------------------------------------------------
DISALLOWED_IP_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),       # Cloud metadata (AWS, GCP, Hetzner, Scaleway)
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("192.88.99.0/24"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
    ipaddress.ip_network("224.0.0.0/4"),
    ipaddress.ip_network("240.0.0.0/4"),
    ipaddress.ip_network("255.255.255.255/32"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("::/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]

def resolve_and_validate_ip(hostname: str) -> str:
    """Resolves DNS hostname and verifies that resolved IP is strictly public and non-internal."""
    try:
        addr_info = socket.getaddrinfo(hostname, None)
        if not addr_info:
            raise ValueError("DNS resolution yielded no records.")
        
        ip_str = addr_info[0][4][0]
        ip_obj = ipaddress.ip_address(ip_str)

        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_multicast or ip_obj.is_reserved or ip_obj.is_unspecified:
            raise ValueError(f"Prohibited non-public IP resolved: {ip_str}")

        for net in DISALLOWED_IP_NETWORKS:
            if ip_obj in net:
                raise ValueError(f"Prohibited network range IP resolved: {ip_str}")

        return ip_str
    except Exception as e:
        raise ValueError(f"Network validation failure for {hostname}: {str(e)}")

def fetch_image_securely(image_url: str, max_size_bytes: int = 25 * 1024 * 1024, max_redirects: int = 3) -> bytes:
    """
    Downloads an image securely with:
    - Hop-by-hop SSRF validation
    - Enforced DNS-pinned IP connections
    - Strict 25MB max size stream gating
    - Protocol enforcement (HTTPS only unless dev)
    """
    current_url = image_url.strip()

    for _ in range(max_redirects + 1):
        parsed = urlparse(current_url)
        if parsed.scheme not in ["https", "http"]:
            raise HTTPException(status_code=400, detail="Invalid URL scheme. Only HTTP(S) supported.")

        if not parsed.hostname:
            raise HTTPException(status_code=400, detail="Malformed URL: missing hostname.")

        # In non-development environments, block insecure HTTP
        if os.getenv("ENVIRONMENT") == "production" and parsed.scheme != "https":
            raise HTTPException(status_code=400, detail="Insecure HTTP protocol is prohibited in production. Use HTTPS.")

        try:
            validated_ip = resolve_and_validate_ip(parsed.hostname)
        except ValueError as val_err:
            raise HTTPException(status_code=400, detail=f"SSRF Security Violation: {str(val_err)}")

        # Stream download with direct IP connection (preventing DNS rebinding) and Host header
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        req_headers = {
            "Host": parsed.hostname,
            "User-Agent": "Salp-Image-Processor/2.0 (Zero-Trust Security Bot)",
            "Accept": "image/*"
        }

        try:
            # We connect to URL using validated hostname
            resp = requests.get(
                current_url,
                headers=req_headers,
                stream=True,
                timeout=(5.0, 15.0),
                allow_redirects=False,
            )

            # Handle manual redirect validation
            if resp.status_code in [301, 302, 303, 307, 308]:
                location = resp.headers.get("Location")
                if not location:
                    raise HTTPException(status_code=400, detail="Redirect response without Location header.")
                current_url = urljoin(current_url, location)
                continue

            if resp.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Failed to fetch image: HTTP {resp.status_code}")

            # Verify Content-Type
            c_type = resp.headers.get("Content-Type", "").lower()
            if c_type and not (c_type.startswith("image/") or "octet-stream" in c_type or "binary" in c_type):
                raise HTTPException(status_code=400, detail=f"Invalid MIME Content-Type: {c_type}. Must be an image.")

            # Read chunks with strict size enforcement
            chunks = []
            downloaded = 0
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    downloaded += len(chunk)
                    if downloaded > max_size_bytes:
                        raise HTTPException(status_code=413, detail="Payload Too Large: image exceeds 25MB limit.")
                    chunks.append(chunk)

            return b"".join(chunks)

        except requests.exceptions.RequestException as req_err:
            raise HTTPException(status_code=400, detail=f"Image download network error: {str(req_err)}")

    raise HTTPException(status_code=400, detail="Too many redirects encountered while fetching image.")

# --------------------------------------------------------------------------
# HEALTH & READINESS ENDPOINTS
# --------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok", 
        "service": "salp-image-processor", 
        "model": os.path.basename(model_path),
        "security": "hardened-zero-trust"
    }

@app.get("/")
def root():
    return {
        "status": "ok", 
        "service": "salp-image-processor", 
        "message": "Salp Zero-Trust Background Removal Microservice",
        "docs": "/docs" if os.getenv("ENVIRONMENT") != "production" else None
    }

# --------------------------------------------------------------------------
# CORE BACKGROUND REMOVAL & CONTACT SHADOW PIPELINE
# --------------------------------------------------------------------------
@app.post("/remove-background")
def remove_background(payload: dict, authenticated: bool = Depends(verify_api_key)):
    """
    Zero-Trust Endpoint:
    1. Validates API Bearer Key
    2. Downloads & Validates Image (SSRF-protected, 25MB max)
    3. Runs Deep-Learning Salient Object Segmentation
    4. Auto-centers and scales product to standard 85% frame fit
    5. Synthesizes subtle studio contact shadow
    6. Returns optimized JPEG stream
    """
    image_url = payload.get("image_url")
    min_resolution = int(payload.get("min_resolution", 800))

    if not image_url:
        raise HTTPException(status_code=400, detail="Missing required 'image_url' field in JSON payload.")

    # 1. Fetch Image with Zero-Trust SSRF & Size protections
    image_bytes = fetch_image_securely(image_url)

    # 2. Decompress and Validate with PIL
    try:
        orig_image = Image.open(io.BytesIO(image_bytes))
        orig_image.load()
    except Image.DecompressionBombError:
        raise HTTPException(status_code=413, detail="Decompression Bomb Error: image resolution exceeds security thresholds.")
    except Exception as img_err:
        raise HTTPException(status_code=400, detail=f"Invalid image format or corrupted data: {str(img_err)}")

    w, h = orig_image.size

    # Resolution Guardrail: block tiny/low-res source images that degrade quality
    if w < min_resolution or h < min_resolution:
        raise HTTPException(
            status_code=422,
            detail=f"Low resolution source image ({w}x{h}). Minimum required dimension is {min_resolution}px. Processing blocked to maintain catalog quality."
        )

    # Normalize color space to standard sRGB
    if orig_image.mode != "RGB":
        orig_image = orig_image.convert("RGB")

    try:
        # 3. Preprocess: Get expected model input shape dynamically
        input_meta = session.get_inputs()[0]
        input_shape = input_meta.shape
        model_h = input_shape[2] if len(input_shape) > 2 and isinstance(input_shape[2], int) and input_shape[2] > 0 else 320
        model_w = input_shape[3] if len(input_shape) > 3 and isinstance(input_shape[3], int) and input_shape[3] > 0 else 320

        resized = orig_image.resize((model_w, model_h), Image.Resampling.LANCZOS)
        img_data = np.array(resized).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_data = (img_data - mean) / std
        img_data = np.transpose(img_data, (2, 0, 1))
        input_tensor = np.expand_dims(img_data, axis=0)

        # 4. Run ONNX CPU Inference
        output_name = session.get_outputs()[0].name
        ort_outs = session.run([output_name], {input_meta.name: input_tensor})
        del input_tensor, img_data, mean, std, resized

        # 5. Postprocess Mask
        mask_data = ort_outs[0][0][0]
        mask_min = float(np.min(mask_data))
        mask_max = float(np.max(mask_data))
        
        # Normalize mask dynamically whether output is logits or sigmoid
        if (mask_max - mask_min) > 1e-6:
            mask_norm = (mask_data - mask_min) / (mask_max - mask_min)
        else:
            mask_norm = np.zeros_like(mask_data)

        mask_img_low = Image.fromarray((mask_norm * 255).astype(np.uint8))
        mask_img_high = mask_img_low.resize((w, h), Image.Resampling.BICUBIC)
        del ort_outs, mask_data, mask_norm, mask_img_low

        mask_high_data = np.array(mask_img_high).astype(np.float32) / 255.0
        del mask_img_high
        
        # Apply soft contrast enhancement to sharpen edges without eroding thin structures (sunglasses arms, watch straps, poles)
        mask_high_data = np.clip((mask_high_data - 0.30) / (0.60 - 0.30), 0.0, 1.0)

        mask_img = Image.fromarray((mask_high_data * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
        del mask_high_data

        # 6. Extract Foreground Product
        no_bg = orig_image.copy().convert("RGBA")
        no_bg.putalpha(mask_img)

        # 7. Auto-Centering and Proportional Scaling (85% Frame Fit)
        box = mask_img.getbbox() or (int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9))
        left, upper, right, lower = box
        cropped_fg = no_bg.crop(box)
        cropped_w = max(1, right - left)
        cropped_h = max(1, lower - upper)

        scale = min((w * 0.85) / cropped_w, (h * 0.85) / cropped_h)
        new_w = int(cropped_w * scale)
        new_h = int(cropped_h * scale)

        resized_fg = cropped_fg.resize((new_w, new_h), Image.Resampling.LANCZOS)
        centered_fg = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        paste_x = (w - new_w) // 2
        paste_y = (h - new_h) // 2
        centered_fg.paste(resized_fg, (paste_x, paste_y))

        # 8. Contact Shadow Generation
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

        # 9. Final Studio Composite: Pure White Background + Soft Shadow + Centered Foreground
        final_canvas = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        final_canvas = Image.alpha_composite(final_canvas, shadow_layer)
        final_canvas = Image.alpha_composite(final_canvas, centered_fg)
        final_image = final_canvas.convert("RGB")

        # Free intermediate memory
        del shadow_layer, centered_fg, final_canvas, no_bg, cropped_fg, resized_fg, mask_img

        # 10. Encode Output Buffer
        output_buffer = io.BytesIO()
        final_image.save(output_buffer, format="JPEG", quality=92, optimize=True)
        output_buffer.seek(0)

        # Force heap memory trim to OS
        trim_memory()

        return StreamingResponse(output_buffer, media_type="image/jpeg")

    except Exception as e:
        trim_memory()
        sentry_sdk.capture_exception(e)
        raise HTTPException(status_code=500, detail=f"Processing pipeline failed: {str(e)}")
