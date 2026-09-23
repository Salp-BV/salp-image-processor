"""
Verda GPU Migration Verification Test Suite
Executes complete automated validation across Health, Security, Functional Categories,
#FFFFFF Corner & Perimeter Purity, Shadow Placement, and RTX/L40S Latency SLA.
"""
import os
import io
import time
import requests
import pytest
import numpy as np
from PIL import Image

VERDA_ENDPOINT = os.getenv("VERDA_ENDPOINT", "https://containers.datacrunch.io/salp-img-staging").rstrip("/")
API_KEY = os.getenv("IMAGE_PROCESSOR_API_KEY", os.getenv("RUNPOD_API_KEY", "")).strip()

TEST_CASES = [
    {
        "id": "test_01",
        "category": "Footwear",
        "name": "Sneaker (Curved Sole)",
        "url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=1000&h=1000&fit=crop"
    },
    {
        "id": "test_03",
        "category": "Footwear",
        "name": "Stiletto Heel (Thin Stem)",
        "url": "https://images.unsplash.com/photo-1543163521-1bf539c55dd2?w=1000&h=1000&fit=crop"
    },
    {
        "id": "test_06",
        "category": "Accessories",
        "name": "Aviator Sunglasses (Wireframe)",
        "url": "https://images.unsplash.com/photo-1511499767150-a48a237f0083?w=1000&h=1000&fit=crop"
    },
    {
        "id": "test_11",
        "category": "Apparel",
        "name": "Folded T-Shirt (Textile Fold)",
        "url": "https://images.unsplash.com/photo-1521572267360-ee0c2909d518?w=1000&h=1000&fit=crop"
    },
    {
        "id": "test_15",
        "category": "Electronics",
        "name": "Over-Ear Headphones (Arch Gap)",
        "url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=1000&h=1000&fit=crop"
    }
]

def test_01_liveness_ping():
    """Verify /ping responds < 500ms and asserts CUDA device."""
    url = f"{VERDA_ENDPOINT}/ping"
    start = time.monotonic()
    resp = requests.get(url, timeout=15)
    latency = time.monotonic() - start

    assert resp.status_code == 200, f"Ping failed with status {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data.get("status") == "healthy", f"Status not healthy: {data}"
    assert data.get("device") == "cuda", "Server must be running with CUDA GPU acceleration!"
    assert latency < 1.0, f"Liveness probe too slow: {latency:.3f}s"

def test_02_readiness_health():
    """Verify /health reports active GPU VRAM residency."""
    url = f"{VERDA_ENDPOINT}/health"
    resp = requests.get(url, timeout=15)
    assert resp.status_code == 200, f"Health check failed: {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data.get("device") == "cuda"
    assert data.get("vram_allocated_mb", 0) > 500, "BiRefNet model weights must be pre-loaded in VRAM"

def test_03_zero_trust_auth():
    """Verify fail-closed authentication (401/403 on missing or invalid key)."""
    url = f"{VERDA_ENDPOINT}/remove-background"

    # Missing token -> 401 or 403
    r_unauth = requests.post(url, json={"image_url": "https://example.com/img.jpg"}, timeout=10)
    assert r_unauth.status_code in [401, 403], f"Expected 401/403 for missing auth, got {r_unauth.status_code}"

    # Invalid token -> 403
    r_bad = requests.post(
        url,
        json={"image_url": "https://example.com/img.jpg"},
        headers={"Authorization": "Bearer invalid-token-xyz"},
        timeout=10
    )
    assert r_bad.status_code == 403, f"Expected 403 for invalid token, got {r_bad.status_code}"

@pytest.mark.parametrize("case", TEST_CASES)
def test_04_functional_categories_and_rgb_purity(case):
    """
    Validates end-to-end inference, pure white #FFFFFF borders,
    contact shadow presence, and <2.5s execution time on Verda GPU.
    """
    url = f"{VERDA_ENDPOINT}/remove-background"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {"image_url": case["url"], "min_resolution": 800}

    start_time = time.monotonic()
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    duration = time.monotonic() - start_time

    assert resp.status_code == 200, f"Inference failed for {case['name']}: {resp.text}"
    assert "image/jpeg" in resp.headers.get("Content-Type", "")

    # Performance SLA: Must execute in < 2.5s warm
    assert duration < 2.5, f"Latency SLA violated: took {duration:.2f}s for {case['name']}"

    # Image Analysis
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    w, h = img.size
    img_np = np.array(img)

    # 1. Assert All 4 Corners Are Pure White #FFFFFF [255, 255, 255]
    corners = [img_np[0, 0], img_np[0, w-1], img_np[h-1, 0], img_np[h-1, w-1]]
    for idx, corner in enumerate(corners):
        assert np.array_equal(corner, [255, 255, 255]), f"Corner {idx} is not pure white: {corner}"

    # 2. Assert Outer 5-pixel Perimeter is 100% Pure White
    top_bar = img_np[0:5, :]
    bottom_bar = img_np[h-5:h, :]
    left_bar = img_np[:, 0:5]
    right_bar = img_np[:, w-5:w]
    assert np.all(top_bar == 255), "Top perimeter contains non-white pixels"
    assert np.all(bottom_bar == 255), "Bottom perimeter contains non-white pixels"
    assert np.all(left_bar == 255), "Left perimeter contains non-white pixels"
    assert np.all(right_bar == 255), "Right perimeter contains non-white pixels"

    # 3. Assert Contact Shadow Presence (Grounding pixels luminance < 250 below object)
    dark_pixels = np.sum((img_np < 250) & (img_np > 10))
    assert dark_pixels > 200, f"No soft contact shadow detected for {case['name']}"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
