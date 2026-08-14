import io
import os
import sys
import gc
import ctypes

# 1. Enforce thread limits & suppress OpenMP/BLAS spin-waiting BEFORE importing third-party C-extensions
os.environ["OMP_NUM_THREADS"] = os.getenv("ONNX_NUM_THREADS", "2")
os.environ["ONNX_NUM_THREADS"] = os.getenv("ONNX_NUM_THREADS", "2")
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["GOMP_SPINCOUNT"] = "0"
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter, ImageDraw
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import requests
import socket
from urllib.parse import urlparse
import ipaddress
import sentry_sdk
import asyncio

sentry_dsn = os.getenv("SENTRY_DSN")
if sentry_dsn:
    sentry_env = os.getenv("APP_ENV", os.getenv("ENVIRONMENT", "production")).lower()
    traces_rate = 1.0 if sentry_env in ["development", "staging", "stg"] else float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1"))
    sentry_sdk.init(
        dsn=sentry_dsn,
        environment=sentry_env,
        traces_sample_rate=traces_rate,
    )

app = FastAPI(
    title="Salp CPU Image Processor",
    description="Apache 2.0 Background Removal with Automated PIL Contact Shadows"
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

# Load BiRefNet model session
session = ort.InferenceSession(
    "models/birefnet_general_quantized.onnx", 
    sess_options=opts,
    providers=["CPUExecutionProvider"]
)

# Call trim_memory after loading session graph
trim_memory()

@app.get("/health")
@app.get("/")
async def health_check():
    return {"status": "ok", "service": "salp-image-processor"}

def get_contact_shadow(w: int, h: int, box) -> Image.Image:
    """
    Calculates the contact plane at the bottom of the product bounding box,
    draws a padded, subtle elliptical drop shadow gradient, and blurs it 
    without clipping or flat edge artifacts.
    """
    left, upper, right, lower = box
    
    product_w = right - left
    product_h = lower - upper
    
    # 1. Subtle, premium shadow sizing (thin, elegant grounding strip)
    shadow_w = int(product_w * 0.90)
    shadow_h = max(int(product_h * 0.04), 8) # Thin vertical span (4%) prevents a heavy blob look
    
    # Opacity 65/255 is incredibly soft, natural, and premium
    shadow_opacity = 65 
    
    # Blur radius scaled to shadow height
    blur_radius = max(int(shadow_h * 0.6), 4)
    
    # 2. Add padding to the drawing canvas to allow the blur to feather out naturally without getting cropped/clipped
    pad = blur_radius * 2
    canvas_w = shadow_w + 2 * pad
    canvas_h = shadow_h + 2 * pad
    
    ellipse_canvas = Image.new("L", (canvas_w, canvas_h), 0)
    draw = ImageDraw.Draw(ellipse_canvas)
    
    # Draw the ellipse in the center of the padded canvas
    draw.ellipse([pad, pad, pad + shadow_w, pad + shadow_h], fill=shadow_opacity)
    
    # Apply Gaussian Blur to the padded canvas (uninterrupted feathering)
    blurred_ellipse = ellipse_canvas.filter(ImageFilter.GaussianBlur(blur_radius))
    
    # Create the transparent shadow layer for the full image
    shadow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    
    # Position the shadow centered horizontally under the product
    shadow_x = left + (product_w - canvas_w) // 2
    shadow_y = lower - pad - (shadow_h // 3) # Anchored perfectly below the sole contact plane
    
    # Paste black color using the blurred ellipse as the alpha transparency mask
    shadow_layer.paste((0, 0, 0, 255), (shadow_x, shadow_y), mask=blurred_ellipse)
    return shadow_layer

def remove_background_and_anchor(orig_image: Image.Image) -> Image.Image:
    w, h = orig_image.size
    
    # 0. Normalize color space to standard sRGB to ensure color accuracy across all displays
    if orig_image.mode != "RGB":
        orig_image = orig_image.convert("RGB")
    
    # 1. Preprocess: Get expected model input shape dynamically
    input_shape = session.get_inputs()[0].shape
    model_h = input_shape[2] if len(input_shape) > 2 and isinstance(input_shape[2], int) and input_shape[2] > 0 else 512
    model_w = input_shape[3] if len(input_shape) > 3 and isinstance(input_shape[3], int) and input_shape[3] > 0 else 512

    resized = orig_image.resize((model_w, model_h), Image.Resampling.LANCZOS)
    img_data = np.array(resized).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std
    img_data = np.transpose(img_data, (2, 0, 1))  # HWC to CHW
    input_tensor = np.expand_dims(img_data, axis=0)

    # 2. Run ONNX CPU Inference
    ort_inputs = {session.get_inputs()[0].name: input_tensor}
    ort_outs = session.run(None, ort_inputs)
    del input_tensor, ort_inputs, img_data, mean, std, resized
    
    # 3. Postprocess: Upscale the soft Sigmoid probability map first using smooth Bicubic interpolation
    mask_data = ort_outs[0][0][0]
    mask_data = np.clip(mask_data, -12, 12)  # Avoid numerical instability / underflow in exp
    mask_prob_low = 1.0 / (1.0 + np.exp(-mask_data))  # Smooth probability map [0, 1] at 512x512
    
    # Upscale the soft gradient first using BICUBIC
    mask_img_low = Image.fromarray((mask_prob_low * 255).astype(np.uint8))
    mask_img_high = mask_img_low.resize((w, h), Image.Resampling.BICUBIC)
    del ort_outs, mask_data, mask_prob_low, mask_img_low

    # Convert to high-resolution numpy array for sub-pixel erosion and razor-sharp contrast mapping
    mask_high_data = np.array(mask_img_high).astype(np.float32) / 255.0
    del mask_img_high
    
    # Apply sub-pixel mask contraction (shift threshold from 0.4-0.6 to 0.45-0.65)
    mask_high_data = np.clip((mask_high_data - 0.45) / (0.65 - 0.45), 0.0, 1.0)
    
    # Adaptive Pedestal/Stool Slicing: Detect if product is sitting on a flat pedestal/stool.
    row_sums = np.sum(mask_high_data > 0.5, axis=1)
    active_rows = np.where(row_sums > 5)[0]
    if len(active_rows) > 0:
        bbox_lower = active_rows[-1]
        bbox_upper = active_rows[0]
        product_height = bbox_lower - bbox_upper
        search_limit = max(bbox_upper + int(product_height * 0.3), bbox_lower - int(h * 0.45))
        
        best_cut_y = None
        max_drop = 0
        for y in range(bbox_lower - 5, search_limit, -1):
            width_current = row_sums[y]
            width_above = row_sums[max(0, y - 12)]
            
            if width_current > w * 0.20 and width_above > w * 0.10:
                drop = width_current - width_above
                if drop > max_drop and drop > (w * 0.06):
                    max_drop = drop
                    best_cut_y = y
        
        if best_cut_y is not None:
            mask_high_data[best_cut_y:] = 0.0
            
    # Re-wrap as PIL image and apply a subtle sub-pixel Gaussian blur for a soft studio finish
    mask_img = Image.fromarray((mask_high_data * 255).astype(np.uint8))
    del mask_high_data, row_sums
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(0.8))

    # 4. Extract Foreground Product
    no_bg = orig_image.copy().convert("RGBA")
    no_bg.putalpha(mask_img)

    # 5. Advanced Auto-Centering and Proportional Scaling (85% Frame Fit)
    box = mask_img.getbbox()
    if not box:
        box = (int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9))
        
    left, upper, right, lower = box
    cropped_fg = no_bg.crop(box)
    cropped_w = right - left
    cropped_h = lower - upper

    scale_w = (w * 0.85) / cropped_w
    scale_h = (h * 0.85) / cropped_h
    scale = min(scale_w, scale_h)

    new_w = int(cropped_w * scale)
    new_h = int(cropped_h * scale)

    resized_fg = cropped_fg.resize((new_w, new_h), Image.Resampling.LANCZOS)

    centered_fg = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    paste_x = (w - new_w) // 2
    paste_y = (h - new_h) // 2
    centered_fg.paste(resized_fg, (paste_x, paste_y))

    # 6. Generate grounded shadow (Adaptive Natural Shadow Extraction + Synthetic Fallback)
    orig_gray = orig_image.convert("L")
    corner_w = min(50, w // 10)
    corner_h = min(50, h // 10)
    top_left_avg = np.mean(np.array(orig_gray.crop((0, 0, corner_w, corner_h))))
    top_right_avg = np.mean(np.array(orig_gray.crop((w - corner_w, 0, w, corner_h))))
    bg_brightness = (top_left_avg + top_right_avg) / 2.0
    
    use_natural_shadow = bg_brightness >= 220.0
    
    shadow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    if use_natural_shadow:
        gray_arr = np.array(orig_gray).astype(np.float32)
        del orig_gray
        shadow_intensity = 255.0 - gray_arr
        del gray_arr
        product_mask = np.array(mask_img).astype(np.float32) / 255.0
        shadow_mask = shadow_intensity * (1.0 - product_mask)
        del shadow_intensity, product_mask
        
        y_indices = np.arange(h).reshape(h, 1)
        y_threshold = lower - int(cropped_h * 0.15)
        ramp_length = max(1, lower - y_threshold)
        vertical_ramp = np.clip((y_indices - y_threshold) / ramp_length, 0.0, 1.0)
        del y_indices
        
        shadow_mask = shadow_mask * vertical_ramp
        del vertical_ramp
        
        shadow_alpha = np.clip((shadow_mask - 15.0) / (120.0 - 15.0), 0.0, 1.0) * 0.85
        del shadow_mask
        
        pad_w = int(cropped_w * 0.1)
        pad_h = int(cropped_h * 0.15)
        shadow_box = (
            max(0, left - pad_w),
            max(0, upper - pad_h),
            min(w, right + pad_w),
            min(h, lower + pad_h)
        )
        s_left, s_upper, s_right, s_lower = shadow_box
        s_w = s_right - s_left
        s_h = s_lower - s_upper
        
        new_s_w = int(s_w * scale)
        new_s_h = int(s_h * scale)
        
        cropped_shadow = Image.fromarray((shadow_alpha * 255).astype(np.uint8)).crop(shadow_box)
        del shadow_alpha
        cropped_shadow = cropped_shadow.filter(ImageFilter.GaussianBlur(3))
        resized_shadow = cropped_shadow.resize((new_s_w, new_s_h), Image.Resampling.BILINEAR)
        
        paste_s_x = paste_x + int((s_left - left) * scale)
        paste_s_y = paste_y + int((s_upper - upper) * scale)
        
        shadow_layer.paste((0, 0, 0, 255), (paste_s_x, paste_s_y), mask=resized_shadow)
    else:
        centered_box = (paste_x, paste_y, paste_x + new_w, paste_y + new_h)
        shadow_layer = get_contact_shadow(w, h, centered_box)

    # 7. Composite: Pure White Background (#FFFFFF) + Dynamic Shadow + Centered Foreground
    white_bg = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    final_canvas = Image.alpha_composite(white_bg, shadow_layer)
    final_image = Image.alpha_composite(final_canvas, centered_fg).convert("RGB")
    
    return final_image

def is_safe_url(url: str) -> bool:
    """
    Blocks Server-Side Request Forgery (SSRF) by validating that the URL scheme is HTTP/HTTPS
    and the resolved hostname maps strictly to a public, non-internal IP address.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
            
        hostname = parsed.hostname
        if not hostname:
            return False
            
        ips = socket.getaddrinfo(hostname, None)
        for family, _, _, _, sockaddr in ips:
            ip_str = sockaddr[0]
            ip_str = ip_str.split('%')[0]
            ip = ipaddress.ip_address(ip_str)
            if (
                ip.is_loopback or 
                ip.is_private or 
                ip.is_link_local or 
                ip.is_multicast or 
                ip.is_reserved or
                ip_str == "169.254.169.254"
            ):
                return False
        return True
    except Exception:
        return False

IMAGE_PROCESSOR_API_KEY = os.getenv("IMAGE_PROCESSOR_API_KEY")
security_scheme = HTTPBearer(auto_error=False)

@app.post("/remove-background")
async def process_image(
    payload: dict,
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme)
):
    if IMAGE_PROCESSOR_API_KEY:
        if not credentials or credentials.credentials != IMAGE_PROCESSOR_API_KEY:
            raise HTTPException(
                status_code=401, 
                detail="Unauthorized: Invalid or missing API key."
            )
            
    image_url = payload.get("image_url")
    min_resolution = payload.get("min_resolution", 800) # Default to 800px standard for premium storefronts
    if not image_url:
        raise HTTPException(status_code=400, detail="Missing 'image_url' in request payload.")
        
    # SSRF Protection Gate
    if not is_safe_url(image_url):
        raise HTTPException(
            status_code=400, 
            detail="Forbidden: Hostname resolves to an internal, reserved, or loopback network address."
        )
        
    try:
        # Download product image from source with size-limit streaming protection (max 25MB)
        max_size = 25 * 1024 * 1024  # 25 MB
        content = bytearray()
        
        with requests.get(image_url, timeout=10, stream=True) as response:
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to fetch image from source URL.")
            
            cl = response.headers.get("Content-Length")
            if cl and int(cl) > max_size:
                raise HTTPException(status_code=413, detail="Image file exceeds maximum allowed size (25MB).")
                
            for chunk in response.iter_content(chunk_size=8192):
                content.extend(chunk)
                if len(content) > max_size:
                    raise HTTPException(status_code=413, detail="Image file size exceeded 25MB threshold during download.")
            
        orig_image = Image.open(io.BytesIO(content)).convert("RGB")
        w, h = orig_image.size
        
        # Resolution Guardrail: block tiny/low-res source images that degrade quality
        if w < min_resolution or h < min_resolution:
            raise HTTPException(
                status_code=422,
                detail=f"Low resolution source image ({w}x{h}). Minimum required dimension is {min_resolution}px. Processing blocked to maintain catalog quality."
            )
            
        # Upper-Bound Resolution Guardrail: Downscale oversized images to max 1200px for storefront optimization and RAM protection
        max_dimension = int(os.getenv("MAX_IMAGE_DIMENSION", "1200"))
        if w > max_dimension or h > max_dimension:
            scale = max_dimension / float(max(w, h))
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            orig_image = orig_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
        # Execute Background Removal + Anchoring
        final_image = remove_background_and_anchor(orig_image)
        
        # Output compressed JPG for storefront load performance
        img_buffer = io.BytesIO()
        final_image.save(img_buffer, format="JPEG", quality=90)
        img_buffer.seek(0)
        
        return StreamingResponse(img_buffer, media_type="image/jpeg")
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference processing failed: {str(e)}")
    finally:
        trim_memory()
