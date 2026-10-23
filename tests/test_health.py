import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from scripts.provision_verda import update_deployment_scaling, API_BASE

def test_syntax_and_import():
    # Verify provisioning scripts compile cleanly
    import py_compile
    py_compile.compile("scripts/provision_verda.py", doraise=True)
    assert True

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
