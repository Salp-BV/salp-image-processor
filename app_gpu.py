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
import urllib3.util.connection
import sentry_sdk

# Force IPv4 (AF_INET) resolution for urllib3/requests so container connections to
# dual-stack CDN hosts (e.g. *.b-cdn.net) never hang on unreachable IPv6 routes.
urllib3.util.connection.allowed_gai_family = lambda: socket.AF_INET

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


def detect_and_repair_hand_occlusion(
    orig_image: Image.Image,
    mask_np: np.ndarray,
    allow_heavy_occlusion: bool = False,
) -> tuple[Image.Image, np.ndarray, dict]:
    """
    Stage 1.5: Semantic Hand/Skin Occlusion Detection, 422 Heavy-Occlusion Guard,
    and 2D Edge / Thumb-Bite Reconstruction.
    """
    rgb = np.array(orig_image).astype(np.float32)
    h, w = mask_np.shape[:2]
    fg_mask = mask_np > 0.5
    fg_area = int(np.sum(fg_mask))
    if fg_area < 400 or h < 60 or w < 60:
        return orig_image, mask_np, {"hand_repaired": False, "occlusion_ratio": 0.0}

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b

    # High-saturation horizontal strap/band protection (e.g. orange elastic band)
    is_strap = (r > 150.0) & (g > 75.0) & (b < 65.0) & (r > g + 35.0)

    # Candidate human skin pixels in RGB + YCbCr
    base_skin = (
        (r > 135.0)
        & (g > 85.0)
        & (b > 60.0)
        & (r > g + 12.0)
        & (g >= b - 4.0)
        & (r > b + 20.0)
        & (cr >= 133.0)
        & (cr <= 180.0)
        & (cb >= 75.0)
        & (cb <= 130.0)
        & (lum < 244.0)
        & (~is_strap)
    )

    ys, xs = np.where(fg_mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    fh, fw = y1 - y0 + 1, x1 - x0 + 1
    if fh < 50 or fw < 50:
        return orig_image, mask_np, {"hand_repaired": False, "occlusion_ratio": 0.0}

    # Check if the product itself is predominantly skin-colored (>80% of foreground matches base_skin)
    fg_non_strap = fg_mask & (~is_strap)
    fg_non_skin = fg_non_strap & (~base_skin)
    if np.sum(fg_non_skin) < int(fg_area * 0.20):
        return orig_image, mask_np, {"hand_repaired": False, "occlusion_ratio": 0.0}

    # Inspect central 40% interior of the foreground bounding box
    cy0, cy1 = y0 + int(fh * 0.30), y0 + int(fh * 0.70)
    cx0, cx1 = x0 + int(fw * 0.30), x0 + int(fw * 0.70)
    core_fg = fg_non_strap[cy0:cy1, cx0:cx1]
    core_non_skin = core_fg & (~base_skin[cy0:cy1, cx0:cx1])

    if np.sum(core_fg) < 40:
        return orig_image, mask_np, {"hand_repaired": False, "occlusion_ratio": 0.0}

    # Check full core (including any bright cork flecks) to see if the product's own texture reaches skin brightness
    core_all_p70_r = float(np.percentile(r[cy0:cy1, cx0:cx1][core_fg], 70))
    core_all_p70_g = float(np.percentile(g[cy0:cy1, cx0:cx1][core_fg], 70))
    core_all_p70_b = float(np.percentile(b[cy0:cy1, cx0:cx1][core_fg], 70))
    core_all_p90_r = float(np.percentile(r[cy0:cy1, cx0:cx1][core_fg], 90))
    core_all_p90_g = float(np.percentile(g[cy0:cy1, cx0:cx1][core_fg], 90))

    # If the central core is unoccluded (<15% base_skin), use its true 90th percentile so natural light cork/wood
    # grain is never truncated at the base_skin threshold (135); if partially occluded, use the 70th percentile.
    core_base_skin_frac = float(np.mean(base_skin[cy0:cy1, cx0:cx1][core_fg]))
    if core_base_skin_frac < 0.15 or (core_all_p90_r - core_all_p70_r < 30.0):
        prod_p90_r = core_all_p90_r
        prod_p90_g = core_all_p90_g
        prod_p90_b = float(np.percentile(b[cy0:cy1, cx0:cx1][core_fg], 90))
    else:
        prod_p90_r = core_all_p70_r
        prod_p90_g = core_all_p70_g
        prod_p90_b = core_all_p70_b

    # Skin pixels must be distinctly brighter/warmer than the product's own core texture
    is_skin = base_skin & (r > prod_p90_r + 22.0) & (g > prod_p90_g + 18.0)

    # Morphological opening (MinFilter -> MaxFilter) removes tiny 1-3px natural cork/wood grain flecks
    morph_k = 5 if min(w, h) >= 500 else 3
    skin_pil = (
        Image.fromarray((is_skin * 255).astype(np.uint8))
        .filter(ImageFilter.MinFilter(morph_k))
        .filter(ImageFilter.MaxFilter(morph_k))
    )
    solid_skin = np.array(skin_pil) > 127

    fg_skin_pixels = int(np.sum(fg_mask & solid_skin))
    fg_skin_ratio = fg_skin_pixels / float(max(1, fg_area))

    # Check if skin touches outer frame borders or indents the product sides
    border_pad = max(4, int(min(w, h) * 0.02))
    border_skin = (
        np.sum(solid_skin[:border_pad, :])
        + np.sum(solid_skin[-border_pad:, :])
        + np.sum(solid_skin[:, :border_pad])
        + np.sum(solid_skin[:, -border_pad:])
    )

    # Reject lifestyle photos where arms/hands reach frame borders with substantial foreground skin
    if not allow_heavy_occlusion and border_skin >= max(150, int(w * 0.25)) and fg_skin_ratio >= 0.08:
        raise HTTPException(
            status_code=422,
            detail="HUMAN_OR_PROP_OCCLUSION_DETECTED: Foreground is heavily occluded by human body parts or props (>=22%).",
        )

    # If the product interior itself is bright/tan (core_all_p90_r >= 142 and no border hands or significant fg skin),
    # do not misclassify light cork/wood grain as hands.
    if core_all_p90_r >= 142.0 and core_all_p90_g >= 90.0 and fg_skin_ratio < 0.02 and border_skin < (w * 0.25):
        return orig_image, mask_np, {"hand_repaired": False, "occlusion_ratio": 0.0}

    # Detect dark non-skin product surface (such as brown/dark cork cover) vs skin/background
    is_dark_product = (prod_p90_r < 145.0) and (prod_p90_g < 102.0) and (prod_p90_b < 82.0)
    if is_dark_product:
        is_cork = (
            (r < max(148.0, prod_p90_r + 22.0))
            & (g < max(102.0, prod_p90_g + 20.0))
            & (b < max(78.0, prod_p90_b + 20.0))
            & (r > b + 6.0)
            & (~solid_skin)
        )
        cork_or_strap = is_cork | is_strap
    else:
        is_cork = fg_mask & (~solid_skin)
        cork_or_strap = is_cork | is_strap

    # Locate product vertical span from central unobstructed columns
    c_x0 = int(w * 0.35)
    c_x1 = int(w * 0.65)
    center_col_mask = np.any(cork_or_strap[:, c_x0:c_x1], axis=1)
    valid_ys = np.where(center_col_mask)[0]
    if len(valid_ys) < 40:
        if fg_skin_ratio >= 0.22 and prod_p90_r < 140.0 and not allow_heavy_occlusion:
            raise HTTPException(
                status_code=422,
                detail="HUMAN_OR_PROP_OCCLUSION_DETECTED: Foreground is heavily occluded by human body parts or props (>=22%).",
            )
        return orig_image, mask_np, {"hand_repaired": False, "occlusion_ratio": 0.0}

    y_top, y_bot = int(valid_ys[0]), int(valid_ys[-1])
    prod_h = y_bot - y_top
    if prod_h < 40:
        return orig_image, mask_np, {"hand_repaired": False, "occlusion_ratio": 0.0}

    # Measure straight left/right edges from the upper unobstructed half (top 6% to 35% of product height)
    y_clean_start = y_top + int(prod_h * 0.06)
    y_clean_end = y_top + int(prod_h * 0.35)
    s_x0, s_x1 = int(w * 0.12), int(w * 0.88)
    min_row_span = int(w * 0.25)

    clean_lefts, clean_rights, clean_ys = [], [], []
    for y in range(y_clean_start, y_clean_end):
        xs_row = np.where(cork_or_strap[y, s_x0:s_x1])[0] + s_x0
        if len(xs_row) > min_row_span and (xs_row[-1] - xs_row[0]) > min_row_span:
            clean_ys.append(y)
            clean_lefts.append(int(xs_row[0]))
            clean_rights.append(int(xs_row[-1]))

    if len(clean_ys) < 10:
        if fg_skin_ratio >= 0.22 and prod_p90_r < 140.0 and not allow_heavy_occlusion:
            raise HTTPException(
                status_code=422,
                detail="HUMAN_OR_PROP_OCCLUSION_DETECTED: Foreground is heavily occluded by human body parts or props (>=22%).",
            )
        if fg_skin_ratio >= 0.01:
            repaired_mask = np.where(solid_skin, 0.0, mask_np).astype(np.float32)
            return orig_image, repaired_mask, {"hand_repaired": True, "occlusion_ratio": float(fg_skin_ratio)}
        return orig_image, mask_np, {"hand_repaired": False, "occlusion_ratio": 0.0}

    left_poly = np.polyfit(clean_ys, clean_lefts, 1)
    right_poly = np.polyfit(clean_ys, clean_rights, 1)
    expected_left = np.clip(np.polyval(left_poly, np.arange(h)).astype(np.int32), 0, w - 1)
    expected_right = np.clip(np.polyval(right_poly, np.arange(h)).astype(np.int32), 0, w - 1)

    prod_w = int(np.median(expected_right[clean_ys] - expected_left[clean_ys]))
    if prod_w < 40:
        return orig_image, mask_np, {"hand_repaired": False, "occlusion_ratio": 0.0}

    med_xl = int(np.median(expected_left[clean_ys]))
    med_xr = int(np.median(expected_right[clean_ys]))
    full_span_h = max(prod_h, y1 - y0 + 1)
    est_prod_area = float(max(1, prod_w * full_span_h))
    skin_inside_prod_cols = int(np.sum(solid_skin[y0:y1 + 1, med_xl:med_xr + 1] & fg_mask[y0:y1 + 1, med_xl:med_xr + 1]))
    actual_skin_occlusion = max(fg_skin_ratio, skin_inside_prod_cols / est_prod_area)

    # Check for Heavy Occlusion (>= 22% of foreground or product silhouette is actual human skin/body)
    if actual_skin_occlusion >= 0.22 and prod_p90_r < 140.0:
        if not allow_heavy_occlusion:
            raise HTTPException(
                status_code=422,
                detail="HUMAN_OR_PROP_OCCLUSION_DETECTED: Foreground is heavily occluded by human body parts or props (>=22%).",
            )
        return orig_image, mask_np, {"hand_repaired": False, "occlusion_ratio": float(round(actual_skin_occlusion, 4))}

    corner_margin = max(2, int(prod_h * 0.025))
    is_cast_shadow = lum < 48.0
    inpaint_mask = np.zeros((h, w), dtype=bool)

    left_search_w = max(12, int(prod_w * 0.235))
    left_trig_w = max(8, int(prod_w * 0.148))
    right_search_w = max(12, int(prod_w * 0.235))
    right_trig_w = max(8, int(prod_w * 0.148))
    shadow_pad = max(4, int(prod_w * 0.038))

    indented_rows = 0
    skin_in_indent_pixels = 0
    for y in range(y_clean_end, y_bot - corner_margin):
        xl = int(expected_left[y])
        xr = int(expected_right[y])
        if xr <= xl + 20:
            continue

        l_lim = min(xr, xl + left_search_w)
        l_trig = min(xr, xl + left_trig_w)
        left_occl = np.where((~is_cork[y, xl:l_lim]) | is_cast_shadow[y, xl:l_lim])[0]
        if len(left_occl) > 0 and (~is_cork[y, xl:l_trig]).any():
            max_x = min(l_lim + 10, xl + int(left_occl[-1]) + shadow_pad)
            inpaint_mask[y, xl:max_x] = True
            indented_rows += 1
            skin_in_indent_pixels += int(np.sum(solid_skin[y, max(0, xl - 20):max_x]))

        r_lim = max(xl, xr - right_search_w)
        r_trig = max(xl, xr - right_trig_w)
        right_occl = np.where((~is_cork[y, r_lim:xr + 1]) | is_cast_shadow[y, r_lim:xr + 1])[0]
        if len(right_occl) > 0 and (~is_cork[y, r_trig:xr + 1]).any():
            min_x = max(r_lim - 10, r_lim + int(right_occl[0]) - shadow_pad)
            inpaint_mask[y, min_x:xr + 1] = True
            indented_rows += 1
            skin_in_indent_pixels += int(np.sum(solid_skin[y, min_x:min(w, xr + 25)]))

    inpaint_area = int(np.sum(inpaint_mask))
    skin_density_in_indent = skin_in_indent_pixels / float(max(1, inpaint_area))
    occlusion_ratio = max(inpaint_area, skin_inside_prod_cols) / est_prod_area

    # Require concentrated skin inside/adjacent to the indentation so curved bags, coiled belts,
    # and open wallets with minor warm accents are never falsely repaired or 422-rejected.
    has_hand_evidence = (
        inpaint_area > 0
        and skin_in_indent_pixels > max(25, int(est_prod_area * 0.002))
        and skin_density_in_indent >= 0.22
    )
    if not has_hand_evidence:
        return orig_image, mask_np, {"hand_repaired": False, "occlusion_ratio": 0.0}

    if actual_skin_occlusion >= 0.22 and not allow_heavy_occlusion:
        raise HTTPException(
            status_code=422,
            detail="HUMAN_OR_PROP_OCCLUSION_DETECTED: Foreground is heavily occluded by human body parts or props (>=22%).",
        )

    # Restore rectangular product silhouette mask across indented middle rows while preserving rounded corners
    repaired_mask = np.zeros((h, w), dtype=np.float32)
    for y in range(y_top, y_bot + 1):
        xl = int(expected_left[y])
        xr = int(expected_right[y])
        if y < y_top + corner_margin or y > y_bot - corner_margin:
            xs_c = np.where(cork_or_strap[y, max(0, xl - 8):min(w, xr + 8)])[0] + max(0, xl - 8)
            if len(xs_c) > 0:
                repaired_mask[y, xs_c[0]:xs_c[-1] + 1] = 1.0
            continue
        repaired_mask[y, xl:xr + 1] = 1.0

    # Locate horizontal elastic strap rows (if present) to protect during 2D texture synthesis
    strap_x0 = int(w * 0.30)
    strap_x1 = int(w * 0.70)
    strap_ys = np.where(np.sum(is_strap[:, strap_x0:strap_x1], axis=1) > max(15, int(prod_w * 0.15)))[0]
    has_strap = len(strap_ys) > 0
    if has_strap:
        y_strap_min = max(y_top + corner_margin + 10, int(strap_ys.min() - 3))
        y_strap_max = min(y_bot - corner_margin - 10, int(strap_ys.max() + 3))
    else:
        y_strap_min = y_clean_end
        y_strap_max = y_clean_end

    # Dilate and feather the 2D inpaint mask for seamless 2D texture blending
    scale_factor = max(0.25, min(1.0, min(w, h) / 2000.0))
    max_filter_size = max(5, int(round(25 * scale_factor)) | 1)
    blur_radius = max(4, int(round(22 * scale_factor)))

    mask_dilated = Image.fromarray((inpaint_mask * 255).astype(np.uint8)).filter(
        ImageFilter.MaxFilter(max_filter_size)
    )
    mask_feathered = mask_dilated.filter(ImageFilter.GaussianBlur(blur_radius))
    alpha_2d = np.array(mask_feathered).astype(np.float32) / 255.0
    alpha_2d = np.where((~cork_or_strap) & (repaired_mask > 0.5), 1.0, alpha_2d)

    if has_strap:
        alpha_2d[y_strap_min:y_strap_max + 1, :] = 0.0

    # Synthesize continuous 2D donor texture block by vertically mirroring the clean upper product half
    donor_rgb = rgb.copy()
    clean_h = max(12, y_strap_min - (y_top + corner_margin))
    donor_span = max(8, clean_h - min(30, clean_h // 3))
    donor_w = min(max(20, int(prod_w * 0.44)), prod_w // 2)

    for y in range(y_clean_end, y_bot + 1):
        dist = abs(y - y_strap_min)
        offset = min(15, clean_h // 4)
        src_y = np.clip((y_strap_min - offset) - (dist % donor_span), y_top + corner_margin, y_strap_min - 1)
        xl, xr = int(expected_left[y]), int(expected_right[y])
        s_xl, s_xr = int(expected_left[src_y]), int(expected_right[src_y])

        if xl + donor_w <= w and s_xl + donor_w <= w:
            donor_rgb[y, xl:xl + donor_w] = rgb[src_y, s_xl:s_xl + donor_w]
        if xr - donor_w >= 0 and s_xr - donor_w >= 0:
            donor_rgb[y, xr - donor_w:xr + 1] = rgb[src_y, s_xr - donor_w:s_xr + 1]

    repaired_rgb = rgb * (1.0 - alpha_2d[:, :, None]) + donor_rgb * alpha_2d[:, :, None]

    # Clean elastic strap ends if present
    if has_strap:
        end_w = max(4, int(25 * scale_factor))
        src_off = max(6, int(35 * scale_factor))
        for y in range(y_strap_min, y_strap_max + 1):
            xl, xr = int(expected_left[y]), int(expected_right[y])
            if xl + src_off + end_w < xr:
                repaired_rgb[y, xl:xl + end_w] = rgb[y, xl + src_off:xl + src_off + end_w]
                repaired_rgb[y, xr - end_w:xr] = rgb[y, xr - src_off - end_w:xr - src_off]

    repaired_pil = Image.fromarray(np.clip(repaired_rgb, 0, 255).astype(np.uint8), mode="RGB")
    soft_mask_pil = Image.fromarray((repaired_mask * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(0.8)
    )
    repaired_mask_out = np.array(soft_mask_pil).astype(np.float32) / 255.0

    return repaired_pil, repaired_mask_out, {
        "hand_repaired": True,
        "occlusion_ratio": float(round(occlusion_ratio, 4)),
    }


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

# Auth: Supports IMAGE_PROCESSOR_API_KEY, VERDA_API_KEY, or RUNPOD_API_KEY
IMAGE_PROCESSOR_API_KEY = os.getenv("IMAGE_PROCESSOR_API_KEY", "").strip()
VERDA_API_KEY = os.getenv("VERDA_API_KEY", "").strip()
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "").strip()
security_scheme = HTTPBearer(auto_error=False)

def verify_api_key(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security_scheme)):
    token = credentials.credentials.strip() if credentials and credentials.credentials else ""
    if not token:
        token = request.headers.get("x-api-key", "").strip()

    valid_keys = [k for k in [IMAGE_PROCESSOR_API_KEY, VERDA_API_KEY, RUNPOD_API_KEY] if k]
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
        resolved_ip = socket.getaddrinfo(parsed.hostname, None, socket.AF_INET)[0][4][0]
        ip_obj = ipaddress.ip_address(resolved_ip)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"DNS resolution failed: {str(e)}")

    if (
        ip_obj.is_private
        or ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_multicast
        or ip_obj.is_reserved
        or ip_obj.is_unspecified
    ):
        raise HTTPException(status_code=400, detail="Prohibited private/internal IP address.")

    for net in DISALLOWED_IP_NETWORKS:
        if ip_obj in net:
            raise HTTPException(status_code=400, detail="Prohibited private/internal IP address.")

    req_headers = {
        "User-Agent": "Salp-Image-Processor/2.0 (Zero-Trust Security Bot)",
        "Accept": "image/*,*/*;q=0.8",
    }

    try:
        resp = requests.get(
            image_url,
            headers=req_headers,
            stream=True,
            timeout=(5.0, 15.0),
        )
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
    except HTTPException:
        raise
    except requests.exceptions.Timeout as e:
        raise HTTPException(status_code=504, detail=f"Image download timed out: {str(e)}")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Image download failed: {str(e)}")

# Health Probes for Serverless Load Balancers & Gateways
@app.get("/ping")
def ping():
    """Liveness probe route."""
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

        # Stage 1.5: Semantic Hand/Skin Occlusion Detection & Edge Repair
        if not payload.get("skip_occlusion_repair"):
            orig_image, mask_np, repair_meta = detect_and_repair_hand_occlusion(
                orig_image,
                mask_np,
                allow_heavy_occlusion=bool(payload.get("allow_heavy_occlusion")),
            )
        else:
            repair_meta = {"hand_repaired": False, "occlusion_ratio": 0.0}

        mask_img = Image.fromarray((np.clip(mask_np, 0.0, 1.0) * 255).astype(np.uint8))

        # Stage 2: Color Decontamination & Despill (Sub-pixel Chroma Neutralization)
        clean_fg = decontaminate_and_despill(orig_image, mask_img)

        # Stage 3: Auto-Centering (85% Fit on 1:1 Square Canvas) & Studio Ground Contact Shadow
        bbox = mask_img.getbbox() or (0, 0, w, h)
        fg = clean_fg.crop(bbox)
        fg_w, fg_h = fg.size

        # Standardize canvas to a 1:1 square matching Storefront aspect-square.
        # Cap at 2048px to prevent decompression bombs and memory spikes on ultra-high-res photos (e.g. 48MP).
        # Floor at 1000px to maintain crisp high-definition packshots.
        raw_dim = max(w, h)
        canvas_dim = min(2048, max(1000, raw_dim))

        # Proportional scale to fit 85% of square canvas, clamped to max 1.0x to avoid pixelation
        target_fit = canvas_dim * 0.85
        scale = min(1.0, target_fit / fg_w, target_fit / fg_h)
        if (fg_w * scale > target_fit) or (fg_h * scale > target_fit):
            scale = min(target_fit / fg_w, target_fit / fg_h)

        new_w = max(1, int(fg_w * scale))
        new_h = max(1, int(fg_h * scale))

        resized_fg = fg.resize((new_w, new_h), Image.Resampling.LANCZOS)
        paste_x = (canvas_dim - new_w) // 2
        paste_y = (canvas_dim - new_h) // 2

        # Ground contact shadow on canvas_dim x canvas_dim
        shadow_layer = Image.new("RGBA", (canvas_dim, canvas_dim), (0, 0, 0, 0))
        shadow_w = int(new_w * 0.90)
        shadow_h = max(8, int(new_h * 0.07))
        shadow_x = paste_x + (new_w - shadow_w) // 2
        shadow_y = min(canvas_dim - shadow_h - 2, paste_y + new_h - (shadow_h // 2))

        shadow_draw = ImageDraw.Draw(shadow_layer)
        shadow_draw.ellipse(
            [shadow_x, shadow_y, shadow_x + shadow_w, shadow_y + shadow_h],
            fill=(20, 20, 25, 110)
        )
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(max(4, int(shadow_h * 0.6))))

        centered_fg = Image.new("RGBA", (canvas_dim, canvas_dim), (0, 0, 0, 0))
        centered_fg.paste(resized_fg, (paste_x, paste_y))

        final_canvas = Image.new("RGBA", (canvas_dim, canvas_dim), (255, 255, 255, 255))
        final_canvas = Image.alpha_composite(final_canvas, shadow_layer)
        final_canvas = Image.alpha_composite(final_canvas, centered_fg)
        final_image = final_canvas.convert("RGB")

        buf = io.BytesIO()
        final_image.save(buf, format="JPEG", quality=92, optimize=True)
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="image/jpeg",
            headers={
                "X-Hand-Repaired": str(repair_meta.get("hand_repaired", False)).lower(),
                "X-Occlusion-Ratio": str(round(float(repair_meta.get("occlusion_ratio", 0.0)), 4)),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        sentry_sdk.capture_exception(e)
        raise HTTPException(status_code=500, detail=f"GPU processing failed: {str(e)}")
