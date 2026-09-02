#!/usr/bin/env python3
"""
Autonomous RunPod Serverless Load Balancer Provisioner
Idempotently provisions or updates 'salp-img-staging' and 'salp-img-prod' LB endpoints.
"""
import os
import sys
import json
import requests

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
if not RUNPOD_API_KEY:
    print("❌ Error: RUNPOD_API_KEY environment variable is not set.")
    sys.exit(1)

GRAPHQL_URL = f"https://api.runpod.io/graphql?api_key={RUNPOD_API_KEY}"
HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json"
}

def run_graphql(query: str, variables: dict = None) -> dict:
    resp = requests.post(
        GRAPHQL_URL,
        headers=HEADERS,
        json={"query": query, "variables": variables or {}},
        timeout=30
    )
    if resp.status_code != 200:
        raise RuntimeError(f"GraphQL request failed (HTTP {resp.status_code}): {resp.text}")
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {json.dumps(data['errors'], indent=2)}")
    return data["data"]

def get_existing_resources() -> tuple[dict, dict]:
    """Fetch existing serverless templates and endpoints for idempotent updates."""
    query = """
    query {
      myself {
        templates {
          id
          name
        }
        endpoints {
          id
          name
        }
      }
    }
    """
    try:
        res = run_graphql(query)
        myself = res.get("myself") or {}
        templates = {t["name"]: t["id"] for t in (myself.get("templates") or []) if "name" in t and "id" in t}
        endpoints = {e["name"]: e["id"] for e in (myself.get("endpoints") or []) if "name" in e and "id" in e}
        return templates, endpoints
    except Exception as e:
        print(f"⚠️ Warning: Could not query existing resources ({e}). Proceeding without cache.")
        return {}, {}

def provision(env_name: str, image_tag: str, is_prod: bool):
    template_name = f"salp-img-processor-{env_name}-tmpl"
    endpoint_name = f"salp-img-{env_name}"
    image_name = f"ghcr.io/salp-bv/salp-image-processor:{image_tag}"

    existing_templates, existing_endpoints = get_existing_resources()

    print(f"\n📦 Step 1: Configuring Serverless Template '{template_name}'...")
    template_mutation = """
    mutation SaveTemplate($input: SaveTemplateInput!) {
      saveTemplate(input: $input) {
        id
        name
        imageName
      }
    }
    """
    template_input = {
        "name": template_name,
        "imageName": image_name,
        "containerDiskInGb": 20,
        "volumeInGb": 0,
        "dockerArgs": "",
        "env": [
            {"key": "PORT", "value": "8080"},
            {"key": "HEALTH_CHECK_PATH", "value": "/ping"},
            {"key": "ENVIRONMENT", "value": env_name},
            {"key": "IMAGE_PROCESSOR_API_KEY", "value": os.getenv("IMAGE_PROCESSOR_API_KEY", "")},
        ]
    }
    if template_name in existing_templates:
        template_input["id"] = existing_templates[template_name]
        print(f"🔄 Updating existing template (ID: {template_input['id']})...")
    else:
        print(f"✨ Creating new template '{template_name}'...")

    t_res = run_graphql(template_mutation, {"input": template_input})
    template_id = t_res["saveTemplate"]["id"]
    print(f"✅ Template ready: ID {template_id}")

    print(f"\n🌐 Step 2: Configuring Load Balancer Endpoint '{endpoint_name}'...")
    endpoint_mutation = """
    mutation SaveEndpoint($input: EndpointInput!) {
      saveEndpoint(input: $input) {
        id
        name
        type
        workersMin
        workersMax
        idleTimeout
      }
    }
    """
    endpoint_input = {
        "name": endpoint_name,
        "templateId": template_id,
        "gpuIds": "ADA_24,AMPERE_24,AMPERE_16",
        "locations": "EU-SE-1,EU-RO-1,EU-FR-1",  # Strictly European datacenters (Sweden, Romania, France)
        "type": "LB",
        "workersMin": 0,  # Scale-to-Zero
        "workersMax": 6 if is_prod else 2,
        "idleTimeout": 15 if is_prod else 10,
        "scalerType": "REQUEST_COUNT",
        "scalerValue": 2,
    }
    if endpoint_name in existing_endpoints:
        endpoint_input["id"] = existing_endpoints[endpoint_name]
        print(f"🔄 Updating existing endpoint (ID: {endpoint_input['id']})...")
    else:
        print(f"✨ Creating new endpoint '{endpoint_name}'...")

    e_res = run_graphql(endpoint_mutation, {"input": endpoint_input})
    ep_id = e_res["saveEndpoint"]["id"]
    print(f"🎉 Endpoint ready! ID: {ep_id}")
    print(f"🔗 Public URL: https://{ep_id}.api.runpod.ai/remove-background")
    print(f"🩺 Health Probe: https://{ep_id}.api.runpod.ai/ping")
    return ep_id

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "staging"
    provision(target, "gpu-main" if target in ["prod", "production"] else "gpu-staging", is_prod=(target in ["prod", "production"]))
