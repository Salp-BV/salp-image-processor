#!/usr/bin/env python3
"""
Autonomous RunPod Serverless Load Balancer Provisioner
Uses the official RunPod Python SDK to idempotently manage templates and endpoints.
"""
import os
import sys
import requests
import json
import time
import runpod

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
if not RUNPOD_API_KEY:
    print("❌ Error: RUNPOD_API_KEY environment variable is not set.")
    sys.exit(1)

runpod.api_key = RUNPOD_API_KEY
REGISTRY_AUTH_ID = os.getenv("RUNPOD_REGISTRY_AUTH_ID", os.getenv("REGISTRY_AUTH_ID", ""))

def save_lb_endpoint(endpoint_name: str, template_id: str, is_prod: bool, existing_id: str = None) -> str:
    url = f"https://api.runpod.io/graphql?api_key={RUNPOD_API_KEY}"
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    mutation = """
    mutation SaveEndpoint($input: EndpointInput!) {
      saveEndpoint(input: $input) {
        id
        name
        type
        templateId
      }
    }
    """
    input_data = {
        "name": endpoint_name,
        "templateId": template_id,
        "type": "LB",
        "gpuIds": "AMPERE_16,ADA_24",
        "locations": "EU-SE-1,EU-RO-1,EU-FR-1",
        "idleTimeout": int(os.getenv("RUNPOD_IDLE_TIMEOUT", "300")),
        "scalerType": "REQUEST_COUNT",
        "scalerValue": 2,
        "workersMin": 0,
        "workersMax": 6 if is_prod else 2
    }
    if existing_id:
        input_data["id"] = existing_id

    resp = requests.post(url, headers=headers, json={"query": mutation, "variables": {"input": input_data}}, timeout=30)
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL Error saving endpoint: {data['errors']}")
    return data["data"]["saveEndpoint"]["id"]

def provision(env_name: str, image_tag: str, is_prod: bool):
    template_name = f"salp-img-{env_name}-{int(time.time())}"
    endpoint_name = f"salp-img-{env_name}"
    image_name = f"ghcr.io/salp-bv/salp-image-processor:{image_tag}"

    print(f"\n📦 Step 1: Configuring Serverless Template '{template_name}'...")
    template = runpod.create_template(
        name=template_name,
        image_name=image_name,
        container_disk_in_gb=20,
        ports="8080/http",
        env={
            "PORT": "8080",
            "PORT_HEALTH": "8080",
            "HEALTH_CHECK_PATH": "/ping",
            "ENVIRONMENT": env_name,
            "IMAGE_PROCESSOR_API_KEY": os.getenv("IMAGE_PROCESSOR_API_KEY", ""),
            "RUNPOD_API_KEY": RUNPOD_API_KEY
        },
        is_serverless=True,
        registry_auth_id=REGISTRY_AUTH_ID
    )
    template_id = template["id"]
    print(f"✅ Template ready: ID {template_id}")

    print(f"\n🌐 Step 2: Configuring Serverless Endpoint '{endpoint_name}'...")
    try:
        existing_endpoints = runpod.get_endpoints()
    except Exception as e:
        print(f"Warning fetching endpoints: {e}")
        existing_endpoints = []

    target_ep = next((e for e in existing_endpoints if e.get("name") == endpoint_name), None)
    existing_id = target_ep["id"] if target_ep else None

    if existing_id:
        print(f"🔄 Updating existing endpoint '{endpoint_name}' (ID: {existing_id}) with new template {template_id}...")
    else:
        print(f"✨ Creating new LOAD_BALANCER endpoint '{endpoint_name}'...")

    ep_id = save_lb_endpoint(endpoint_name, template_id, is_prod, existing_id)

    print(f"🎉 Endpoint ready! ID: {ep_id}")
    print(f"🔗 Public URL: https://{ep_id}.api.runpod.ai/remove-background")
    print(f"🩺 Health Probe: https://{ep_id}.api.runpod.ai/ping")
    return ep_id

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "staging"
    provision(target, "gpu-main" if target in ["prod", "production"] else "gpu-staging", is_prod=(target in ["prod", "production"]))
