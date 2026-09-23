#!/usr/bin/env python3
"""
Autonomous Verda Serverless GPU Container Provisioner
Uses the official Verda Cloud REST API (v1) to idempotently manage serverless GPU deployments.
"""
import os
import sys
import time
import requests

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

VAULT_PATH = r"C:\Users\jopbr\.gemini\config\infrastructure_keys.env"

def load_vault_env():
    """Fallback to local master infrastructure keys vault if environment variables not set in shell."""
    if os.path.exists(VAULT_PATH):
        try:
            with open(VAULT_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k not in os.environ and v:
                            os.environ[k] = v
        except Exception as e:
            print(f"Notice: unable to load vault {VAULT_PATH}: {e}")

load_vault_env()

VERDA_CLIENT_ID = os.getenv("VERDA_CLIENT_ID", "").strip()
VERDA_CLIENT_SECRET = os.getenv("VERDA_CLIENT_SECRET", "").strip()
IMAGE_PROCESSOR_API_KEY = os.getenv("IMAGE_PROCESSOR_API_KEY", os.getenv("RUNPOD_API_KEY", "")).strip()
GITHUB_PACKAGES_PAT = os.getenv("GITHUB_PACKAGES_PAT", "").strip()

if not VERDA_CLIENT_ID or not VERDA_CLIENT_SECRET:
    print("❌ Error: VERDA_CLIENT_ID or VERDA_CLIENT_SECRET is not configured.")
    sys.exit(1)

API_BASE = "https://api.verda.com/v1"

def get_auth_headers() -> dict:
    token_url = f"{API_BASE}/oauth2/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": VERDA_CLIENT_ID,
        "client_secret": VERDA_CLIENT_SECRET
    }
    resp = requests.post(token_url, json=payload, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to authenticate with Verda OAuth2 endpoint: {resp.status_code} - {resp.text}")
    token = resp.json().get("access_token")
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

def ensure_ghcr_credentials(headers: dict):
    cred_name = "ghcr-salp-registry"
    resp = requests.get(f"{API_BASE}/container-registry-credentials", headers=headers, timeout=45)
    if resp.status_code == 200:
        creds = resp.json()
        if any(c.get("name") == cred_name for c in creds):
            print(f"✅ Container Registry Credential '{cred_name}' is verified.")
            return cred_name

    if not GITHUB_PACKAGES_PAT:
        print("⚠️ Warning: GITHUB_PACKAGES_PAT not found. Assuming registry is already registered.")
        return cred_name

    print(f"📦 Registering GHCR credentials '{cred_name}' in Verda...")
    reg_payload = {
        "name": cred_name,
        "type": "ghcr",
        "username": "jopperdeplop",
        "access_token": GITHUB_PACKAGES_PAT
    }
    reg_resp = requests.post(f"{API_BASE}/container-registry-credentials", headers=headers, json=reg_payload, timeout=20)
    if reg_resp.status_code in [200, 201]:
        print(f"✅ Successfully registered GHCR credentials '{cred_name}'.")
    else:
        print(f"Notice: registration response {reg_resp.status_code} - {reg_resp.text}")
    return cred_name

def provision(env_name: str = "staging"):
    is_prod = env_name.lower() in ["prod", "production"]
    deployment_name = "salp-img-production" if is_prod else "salp-img-staging"
    image_tag = "gpu-main" if is_prod else "gpu-staging"
    image_name = f"ghcr.io/salp-bv/salp-image-processor:{image_tag}"

    print(f"\n🚀 Initiating Verda Serverless Deployment for '{deployment_name}' ({env_name})...")
    headers = get_auth_headers()
    cred_name = ensure_ghcr_credentials(headers)

    # Check existing deployments
    resp = requests.get(f"{API_BASE}/container-deployments", headers=headers, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to list container deployments: {resp.status_code} - {resp.text}")
    existing_deployments = resp.json()
    existing = next((d for d in existing_deployments if d.get("name") == deployment_name), None)

    env_vars = [
        {"name": "PORT", "value_or_reference_to_secret": "8080", "type": "plain"},
        {"name": "IMAGE_PROCESSOR_API_KEY", "value_or_reference_to_secret": IMAGE_PROCESSOR_API_KEY, "type": "plain"},
        {"name": "VERDA_API_KEY", "value_or_reference_to_secret": IMAGE_PROCESSOR_API_KEY, "type": "plain"},
        {"name": "ENVIRONMENT", "value_or_reference_to_secret": env_name, "type": "plain"},
    ]

    container_def = {
        "name": f"{deployment_name}-0",
        "image": image_name,
        "should_use_cached_image": True,
        "exposed_port": 8080,
        "healthcheck": {
            "enabled": True,
            "port": 8080,
            "path": "/ping"
        },
        "env": env_vars
    }

    if existing:
        print(f"🔄 Updating existing deployment '{deployment_name}' with image '{image_name}'...")
        patch_payload = {
            "containers": [container_def],
            "compute": {
                "name": "L40S",
                "size": 1
            },
            "container_registry_settings": {
                "is_private": True,
                "credentials": {
                    "name": cred_name
                }
            }
        }
        patch_resp = requests.patch(f"{API_BASE}/container-deployments/{deployment_name}", headers=headers, json=patch_payload, timeout=30)
        if patch_resp.status_code not in [200, 201, 204]:
            raise RuntimeError(f"Failed to update deployment: {patch_resp.status_code} - {patch_resp.text}")
        print(f"✅ Update submitted successfully.")
    else:
        print(f"✨ Creating new serverless deployment '{deployment_name}' on Verda (L40S GPU, Scale-to-Zero)...")
        create_payload = {
            "name": deployment_name,
            "container_registry_settings": {
                "is_private": True,
                "credentials": {
                    "name": cred_name
                }
            },
            "compute": {
                "name": "L40S",
                "size": 1
            },
            "scaling": {
                "min_replica_count": 0,
                "max_replica_count": 4 if is_prod else 2,
                "scale_down_policy": {
                    "delay_seconds": 300
                },
                "scale_up_policy": {
                    "delay_seconds": 10
                },
                "queue_message_ttl_seconds": 300,
                "concurrent_requests_per_replica": 2,
                "scaling_triggers": {
                    "queue_load": {
                        "threshold": 2
                    },
                    "cpu_utilization": {
                        "enabled": False,
                        "threshold": 80
                    },
                    "gpu_utilization": {
                        "enabled": False,
                        "threshold": 80
                    }
                }
            },
            "containers": [container_def],
            "is_spot": False
        }
        create_resp = requests.post(f"{API_BASE}/container-deployments", headers=headers, json=create_payload, timeout=30)
        if create_resp.status_code not in [200, 201]:
            raise RuntimeError(f"Failed to create deployment: {create_resp.status_code} - {create_resp.text}")
        print(f"✅ Deployment '{deployment_name}' created successfully.")

    # Retrieve endpoint base URL
    print("🌐 Fetching deployment endpoint details...")
    endpoint_url = None
    for _ in range(12):
        get_resp = requests.get(f"{API_BASE}/container-deployments/{deployment_name}", headers=headers, timeout=20)
        if get_resp.status_code == 200:
            dep_info = get_resp.json()
            endpoint_url = dep_info.get("endpoint_base_url")
            status = dep_info.get("status", "unknown")
            print(f"   Status: {status} | Endpoint: {endpoint_url or 'Provisioning...'}")
            if endpoint_url:
                break
        time.sleep(3)

    if not endpoint_url:
        endpoint_url = f"https://containers.verda.com/{deployment_name}"

    clean_base = endpoint_url.rstrip("/")
    remove_bg_url = f"{clean_base}/remove-background"
    ping_url = f"{clean_base}/ping"
    health_url = f"{clean_base}/health"

    print("\n" + "="*70)
    print(f"🎉 VERDA DEPLOYMENT READY: {deployment_name}")
    print(f"🔗 Public Base URL:     {clean_base}")
    print(f"🚀 Remove Background:  {remove_bg_url}")
    print(f"🩺 Liveness Probe:      {ping_url}")
    print(f"🩺 Readiness Probe:     {health_url}")
    print("="*70)
    return remove_bg_url

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "staging"
    provision(target)
