import sys
import socket
import pytest
import requests
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
