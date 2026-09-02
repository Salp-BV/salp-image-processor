#!/usr/bin/env python3
"""
Autonomous RunPod Serverless Load Balancer Provisioner
Uses the official RunPod Python SDK to idempotently manage templates and endpoints.
"""
import os
import sys
import runpod

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
if not RUNPOD_API_KEY:
    print("❌ Error: RUNPOD_API_KEY environment variable is not set.")
    sys.exit(1)

runpod.api_key = RUNPOD_API_KEY

def provision(env_name: str, image_tag: str, is_prod: bool):
    template_name = f"salp-img-processor-{env_name}-tmpl"
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
            "HEALTH_CHECK_PATH": "/ping",
            "ENVIRONMENT": env_name,
            "IMAGE_PROCESSOR_API_KEY": os.getenv("IMAGE_PROCESSOR_API_KEY", "")
        },
        is_serverless=True
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

    if target_ep:
        ep_id = target_ep["id"]
        print(f"🔄 Updating existing endpoint '{endpoint_name}' (ID: {ep_id}) with new template {template_id}...")
        runpod.update_endpoint_template(ep_id, template_id)
    else:
        print(f"✨ Creating new endpoint '{endpoint_name}'...")
        try:
            new_ep = runpod.create_endpoint(
                name=endpoint_name,
                template_id=template_id,
                gpu_ids="AMPERE_16,ADA_24",
                locations="EU-SE-1,EU-RO-1,EU-FR-1",
                idle_timeout=15 if is_prod else 10,
                scaler_type="QUEUE_DELAY",
                scaler_value=2,
                workers_min=0,
                workers_max=6 if is_prod else 2
            )
            ep_id = new_ep["id"]
        except Exception as e:
            if "at least $0.01 in your account balance" in str(e):
                print(f"⚠️ RunPod Account Notice: {e}")
                print("Endpoint will activate as soon as credits are added to https://runpod.io/console/billing.")
                return None
            raise e

    print(f"🎉 Endpoint ready! ID: {ep_id}")
    print(f"🔗 Public URL: https://{ep_id}.api.runpod.ai/remove-background")
    print(f"🩺 Health Probe: https://{ep_id}.api.runpod.ai/ping")
    return ep_id

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "staging"
    provision(target, "gpu-main" if target in ["prod", "production"] else "gpu-staging", is_prod=(target in ["prod", "production"]))
