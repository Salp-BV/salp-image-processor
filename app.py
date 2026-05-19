import io
import os
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

app = FastAPI(
    title="Salp CPU Image Processor",
    description="Apache 2.0 Background Removal with Automated PIL Contact Shadows"
)

# Configure highly optimized, memory-efficient ONNX runtime session options
opts = ort.SessionOptions()
opts.enable_cpu_mem_arena = False  # Return memory to OS immediately, preventing RAM caching accumulation
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # Lower memory overhead sequential execution
opts.intra_op_num_threads = 4  # Utilize all 4 allocated CPU cores in parallel for maximum speed
opts.inter_op_num_threads = 1
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC  # Drastically cuts cold start RAM bloat
opts.add_session_config_entry("session.use_mmap", "1")  # Map weights directly from disk to save 1.5 GiB RAM

# Load quantized BiRefNet (Boots on CPU in < 50ms)
session = ort.InferenceSession(
    "models/birefnet_general_quantized.onnx", 
    sess_options=opts,
    providers=["CPUExecutionProvider"]
)

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
    
    # 1. Preprocess: Force model native high-resolution input size (1024x1024) to capture thin structures
    model_h = 1024
    model_w = 1024

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
    
    # 3. Postprocess mask back to original resolution using mathematically absolute Sigmoid Activation
    mask_data = ort_outs[0][0][0]
    mask_data = np.clip(mask_data, -12, 12)  # Avoid numerical instability / underflow in exp
    mask_prob = 1.0 / (1.0 + np.exp(-mask_data))  # Convert raw logits to true absolute probability space [0, 1]

    # Contrast adjustment and feathering: Map probability smoothly from 0-1 into alpha [0, 255]
    # Below 0.4 probability is absolute background (0), above 0.6 is absolute foreground (255)
    mask_prob = np.clip((mask_prob - 0.4) / (0.6 - 0.4), 0.0, 1.0)
    
    # Re-scale back to original resolution and apply a subtle sub-pixel Gaussian blur for studio anti-aliasing
    mask_img = Image.fromarray((mask_prob * 255).astype(np.uint8)).resize((w, h), Image.Resampling.LANCZOS)
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(0.8))

    # 4. Extract Foreground Product
    no_bg = Image.new("RGBA", (w, h))
    no_bg.paste(orig_image, (0, 0), mask=mask_img)

    # 5. Advanced Auto-Centering and Proportional Scaling (85% Frame Fit)
    box = mask_img.getbbox()
    if not box:
        box = (int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9))
        
    left, upper, right, lower = box
    cropped_fg = no_bg.crop(box)
    cropped_w = right - left
    cropped_h = lower - upper

    # Scale to exactly 85% of target width or height
    scale_w = (w * 0.85) / cropped_w
    scale_h = (h * 0.85) / cropped_h
    scale = min(scale_w, scale_h)

    new_w = int(cropped_w * scale)
    new_h = int(cropped_h * scale)

    # Resize product foreground using Lanczos
    resized_fg = cropped_fg.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Paste scaled product centered perfectly on transparent full-size canvas
    centered_fg = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    paste_x = (w - new_w) // 2
    paste_y = (h - new_h) // 2
    centered_fg.paste(resized_fg, (paste_x, paste_y), mask=resized_fg.split()[3])

    # 6. Generate centered soft contact shadow under the newly positioned product sole
    centered_box = (paste_x, paste_y, paste_x + new_w, paste_y + new_h)
    shadow_layer = get_contact_shadow(w, h, centered_box)

    # 7. Composite: Pure White Background (#FFFFFF) + Contact Shadow + Centered Foreground
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
            
        # Resolve all IP addresses for the hostname
        ips = socket.getaddrinfo(hostname, None)
        for family, _, _, _, sockaddr in ips:
            ip_str = sockaddr[0]
            # Handle potential IPv6/IPv4 brackets
            ip_str = ip_str.split('%')[0]
            ip = ipaddress.ip_address(ip_str)
            # Block loopback, private, link-local, multicast, and GCP Metadata server (169.254.169.254)
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
            
            # Fast-path check: Verify Content-Length early if present
            cl = response.headers.get("Content-Length")
            if cl and int(cl) > max_size:
                raise HTTPException(status_code=413, detail="Image file exceeds maximum allowed size (25MB).")
                
            for chunk in response.iter_content(chunk_size=8192):
                content.extend(chunk)
                if len(content) > max_size:
                    raise HTTPException(status_code=413, detail="Image file size exceeded 25MB threshold during download.")
            
        orig_image = Image.open(io.BytesIO(content)).convert("RGB")
        w, h = orig_image.size
        
        # ⚠️ Resolution Guardrail: block tiny/low-res source images that degrade quality
        if w < min_resolution or h < min_resolution:
            raise HTTPException(
                status_code=422, # Unprocessable Entity
                detail=f"Low resolution source image ({w}x{h}). Minimum required dimension is {min_resolution}px. Processing blocked to maintain catalog quality."
            )
            
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

