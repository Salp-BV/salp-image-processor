import sys
import socket
import pytest
import requests
import numpy as np
from PIL import Image
import urllib3.util.connection
from unittest.mock import patch, MagicMock
from fastapi import HTTPException

from scripts.provision_verda import update_deployment_scaling, API_BASE

_APP_GPU_MODULE = None


def _load_app_gpu():
    """Import app_gpu once safely even in CPU-only test environments without torch/transformers."""
    global _APP_GPU_MODULE
    if _APP_GPU_MODULE is not None:
        return _APP_GPU_MODULE

    if "torch" not in sys.modules:
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        sys.modules["torch"] = mock_torch

    if "torchvision" not in sys.modules:
        mock_torchvision = MagicMock()
        mock_transforms = MagicMock()
        mock_torchvision.transforms = mock_transforms
        sys.modules["torchvision"] = mock_torchvision
        sys.modules["torchvision.transforms"] = mock_transforms

    if "transformers" not in sys.modules:
        sys.modules["transformers"] = MagicMock()

    import app_gpu
    _APP_GPU_MODULE = app_gpu
    return _APP_GPU_MODULE


def test_syntax_and_import():
    # Verify provisioning and application scripts compile cleanly
    import py_compile
    py_compile.compile("scripts/provision_verda.py", doraise=True)
    py_compile.compile("app_gpu.py", doraise=True)
    py_compile.compile("app.py", doraise=True)
    assert True


def test_app_gpu_enforces_ipv4_gai_family():
    app_gpu = _load_app_gpu()
    assert app_gpu is not None
    assert urllib3.util.connection.allowed_gai_family() == socket.AF_INET


def test_fetch_image_securely_timeout_raises_504():
    app_gpu = _load_app_gpu()
    with patch("socket.getaddrinfo") as mock_getaddrinfo, patch("requests.get") as mock_get:
        mock_getaddrinfo.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]
        mock_get.side_effect = requests.exceptions.ConnectTimeout("Connection to salp-media-cdn.b-cdn.net timed out")

        with pytest.raises(HTTPException) as exc_info:
            app_gpu.fetch_image_securely("https://salp-media-cdn.b-cdn.net/products/test.jpg")

        assert exc_info.value.status_code == 504
        assert "Image download timed out" in exc_info.value.detail
        mock_getaddrinfo.assert_called_once_with("salp-media-cdn.b-cdn.net", None, socket.AF_INET)


def test_fetch_image_securely_request_exception_raises_502():
    app_gpu = _load_app_gpu()
    with patch("socket.getaddrinfo") as mock_getaddrinfo, patch("requests.get") as mock_get:
        mock_getaddrinfo.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]
        mock_get.side_effect = requests.exceptions.ConnectionError("Upstream reset connection")

        with pytest.raises(HTTPException) as exc_info:
            app_gpu.fetch_image_securely("https://righteouslight.s2-tastewp.com/wp-content/uploads/test.jpg")

        assert exc_info.value.status_code == 502
        assert "Image download failed" in exc_info.value.detail


def test_fetch_image_securely_passes_headers_and_streams_bytes():
    app_gpu = _load_app_gpu()
    with patch("socket.getaddrinfo") as mock_getaddrinfo, patch("requests.get") as mock_get:
        mock_getaddrinfo.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_content.return_value = [b"fake-", b"image-", b"bytes"]
        mock_get.return_value = mock_resp

        result = app_gpu.fetch_image_securely("https://righteouslight.s2-tastewp.com/wp-content/uploads/test.jpg")
        assert result == b"fake-image-bytes"

        _, kwargs = mock_get.call_args
        assert kwargs["headers"]["User-Agent"] == "Salp-Image-Processor/2.0 (Zero-Trust Security Bot)"
        assert kwargs["headers"]["Accept"] == "image/*,*/*;q=0.8"
        assert kwargs["timeout"] == (5.0, 15.0)
        assert kwargs["stream"] is True


def test_update_deployment_scaling():
    headers = {"Authorization": "Bearer mock-token", "Content-Type": "application/json"}
    with patch("requests.patch") as mock_patch:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_patch.return_value = mock_resp

        update_deployment_scaling("salp-img-staging", headers)

        mock_patch.assert_called_once()
        call_url = mock_patch.call_args[0][0]
        call_json = mock_patch.call_args[1]["json"]
        assert call_url == f"{API_BASE}/container-deployments/salp-img-staging/scaling"
        assert call_json["scaling"]["concurrent_requests_per_replica"] == 4
        assert call_json["scaling"]["max_replicas"] == 2


def test_update_deployment_scaling_production():
    headers = {"Authorization": "Bearer mock-token", "Content-Type": "application/json"}
    with patch("requests.patch") as mock_patch:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_patch.return_value = mock_resp

        update_deployment_scaling("salp-img-production", headers)

        mock_patch.assert_called_once()
        call_url = mock_patch.call_args[0][0]
        call_json = mock_patch.call_args[1]["json"]
        assert call_url == f"{API_BASE}/container-deployments/salp-img-production/scaling"
        assert call_json["scaling"]["concurrent_requests_per_replica"] == 4
        assert call_json["scaling"]["max_replicas"] == 4


def test_update_deployment_scaling_fallback():
    headers = {"Authorization": "Bearer mock-token", "Content-Type": "application/json"}
    with patch("requests.patch") as mock_patch:
        resp_fail = MagicMock()
        resp_fail.status_code = 400
        resp_fail.text = "Unknown field 'scaling'"

        resp_ok = MagicMock()
        resp_ok.status_code = 200

        mock_patch.side_effect = [resp_fail, resp_ok]

        update_deployment_scaling("salp-img-staging", headers)

        assert mock_patch.call_count == 2
        call_json = mock_patch.call_args[1]["json"]
        assert "min_replicas" in call_json


def test_detect_and_repair_hand_occlusion_repairs_thumbs_and_removes_wrist():
    app_gpu = _load_app_gpu()
    w, h = 800, 800
    rgb = np.full((h, w, 3), 255, dtype=np.uint8)
    mask_np = np.zeros((h, w), dtype=np.float32)

    # Dark brown cork journal at x=210..575, y=50..760
    rgb[50:760, 210:576] = [82, 45, 32]
    mask_np[50:760, 210:576] = 1.0

    # Horizontal orange elastic strap at y=240..258
    rgb[240:258, 210:576] = [205, 95, 35]

    # Left thumb (skin color) indenting left edge + dark cast shadow
    rgb[450:550, 180:255] = [220, 172, 138]
    rgb[450:550, 255:266] = [30, 18, 12]
    mask_np[450:550, 210:255] = 0.0  # BiRefNet carved out thumb-bite hole on left

    # Right thumb (skin color) indenting right edge + dark cast shadow
    rgb[460:560, 530:600] = [222, 175, 140]
    rgb[460:560, 518:530] = [30, 18, 12]
    mask_np[460:560, 530:576] = 1.0  # BiRefNet kept right thumb in foreground

    # Floating severed right wrist in foreground mask
    rgb[520:670, 595:665] = [226, 180, 146]
    mask_np[520:670, 595:665] = 1.0

    orig_img = Image.fromarray(rgb, mode="RGB")
    repaired_img, repaired_mask, meta = app_gpu.detect_and_repair_hand_occlusion(orig_img, mask_np)

    assert meta["hand_repaired"] is True
    assert 0.01 < meta["occlusion_ratio"] < 0.22

    # Floating right wrist must be removed from the mask
    assert float(np.max(repaired_mask[530:660, 600:660])) < 0.05

    # Left and right journal edges across the thumb-bite rows must be restored in the mask
    assert float(np.mean(repaired_mask[470:540, 215:570])) > 0.95

    # Skin pixels in the thumb-bite zones must be replaced with dark brown cork texture
    repaired_np = np.array(repaired_img, dtype=np.float32)
    assert float(np.mean(repaired_np[470:540, 215:255, 0])) < 115.0
    assert float(np.mean(repaired_np[470:540, 530:570, 0])) < 115.0

    # Orange elastic strap must be preserved intact
    assert float(np.mean(repaired_np[244:254, 220:565, 0])) > 190.0


def test_detect_and_repair_hand_occlusion_clean_product_untouched():
    app_gpu = _load_app_gpu()
    w, h = 800, 800
    rgb = np.full((h, w, 3), 255, dtype=np.uint8)
    mask_np = np.zeros((h, w), dtype=np.float32)

    # Clean unobstructed brown cork journal
    rgb[60:740, 220:580] = [85, 48, 34]
    rgb[230:248, 220:580] = [205, 95, 35]
    mask_np[60:740, 220:580] = 1.0

    orig_img = Image.fromarray(rgb, mode="RGB")
    repaired_img, repaired_mask, meta = app_gpu.detect_and_repair_hand_occlusion(orig_img, mask_np)

    assert meta["hand_repaired"] is False
    assert meta["occlusion_ratio"] == 0.0
    assert repaired_img is orig_img
    assert np.array_equal(repaired_mask, mask_np)


def test_detect_and_repair_hand_occlusion_heavy_occlusion_raises_422():
    app_gpu = _load_app_gpu()
    w, h = 800, 800
    rgb = np.full((h, w, 3), 255, dtype=np.uint8)
    mask_np = np.zeros((h, w), dtype=np.float32)

    # Product with >= 25% human skin/body occlusion
    rgb[100:700, 200:600] = [60, 45, 35]
    mask_np[100:700, 200:600] = 1.0
    rgb[450:700, 200:600] = [220, 172, 138]

    orig_img = Image.fromarray(rgb, mode="RGB")
    with pytest.raises(HTTPException) as exc_info:
        app_gpu.detect_and_repair_hand_occlusion(orig_img, mask_np)

    assert exc_info.value.status_code == 422
    assert "HUMAN_OR_PROP_OCCLUSION_DETECTED" in exc_info.value.detail
