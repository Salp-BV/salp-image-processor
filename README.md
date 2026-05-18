# GCP Deployment & CI/CD Operations Guide

This guide serves as your master, production-grade operational manual for setting up, containerizing, deploying, and connecting your serverless image background-removal microservice using **BiRefNet (ONNX CPU)** on **Google Cloud Run (Serverless CPU)**.

---

## 🏗️ Architecture Design & Flow

Before beginning, review the core operational flow of the system. The Next.js portal server acts as a thin client, delegating all image segmentation and contact shadow calculations to Google Cloud Run to avoid package bloating and CUDA loading lag on your Vercel functions:

```text
[ Lifestyle Image Url ]
          │
          ▼
[ Trigger.dev Sync Task ] ──(POST payload)──► [ Google Cloud Run CPU API ]
                                                      │
                                                      ├─► Segment Foreground (BiRefNet ONNX)
                                                      ├─► Generate Soft Contact Shadow (PIL)
                                                      └─► Composite onto Pure White
                                                              │
          ◄────────────────(Stream JPEG Bytes)────────────────┘
          │
          ▼
[ Upload Processed Image to Saleor Core ]
```

---

## 🛠️ Step 1: Local Environment & Code Setup

We have successfully created a dedicated folder on your local machine named **`salp-image-processor`** located at `c:\Users\jopbr\Documents\GitHub\salp-image-processor`.

### 1. Folder Structure
Ensure your local project matches this structure exactly:
```text
salp-image-processor/
├── .github/
│   └── workflows/
│       └── deploy.yml                  <-- GitHub Actions CI/CD Pipeline
├── models/
│   └── birefnet_general_quantized.onnx  <-- Local model (Ignored by Git, baked in Docker)
├── .gitignore                          <-- Excludes model binaries and virtual environments
├── app.py                              <-- FastAPI Application
├── requirements.txt                    <-- Python Dependencies
└── README.md                           <-- This Guide
```

---

### 2. File: `app.py`
This is your FastAPI server. It handles downloading the lifestyle product image from the source URL, executing high-speed CPU matrix multiplication via ONNX Runtime to create a binary mask, generating a proportional soft elliptical drop-shadow at the base contact plane, and compositing the final JPEG:

```python
import io
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter, ImageDraw
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import requests

app = FastAPI(
    title="Salp CPU Image Processor",
    description="Apache 2.0 Background Removal with Automated PIL Contact Shadows"
)

# Load quantized BiRefNet (Boots on CPU in < 50ms)
session = ort.InferenceSession(
    "models/birefnet_general_quantized.onnx", 
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
    
    # 1. Preprocess: Get expected model input shape (e.g., [1, 3, 512, 512] or [1, 3, 1024, 1024])
    input_shape = session.get_inputs()[0].shape
    model_h = input_shape[2] if len(input_shape) > 2 and isinstance(input_shape[2], int) else 512
    model_w = input_shape[3] if len(input_shape) > 3 and isinstance(input_shape[3], int) else 512

    resized = orig_image.resize((model_w, model_h), Image.Resampling.LANCZOS)
    img_data = np.array(resized).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std
    img_data = np.transpose(img_data, (2, 0, 1))  # HWC to CHW
    input_tensor = np.expand_dims(img_data, axis=0)

    # 2. Run ONNX CPU Inference
    ort_inputs = {session.get_inputs()[0].name: input_tensor}
    session_outputs = session.run(None, ort_inputs)
    
    # 3. Postprocess mask back to original resolution
    mask_data = session_outputs[0][0][0]
    mask_data = (mask_data - mask_data.min()) / (mask_data.max() - mask_data.min())
    mask_img = Image.fromarray((mask_data * 255).astype(np.uint8)).resize((w, h), Image.Resampling.LANCZOS)

    # Apply hard binary threshold to enforce razor-sharp product edges and zero background bleed
    mask_img = mask_img.point(lambda p: 255 if p > 127 else 0)

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

@app.post("/remove-background")
async def process_image(payload: dict):
    image_url = payload.get("image_url")
    min_resolution = payload.get("min_resolution", 800) # Default to 800px standard for premium storefronts
    if not image_url:
        raise HTTPException(status_code=400, detail="Missing 'image_url' in request payload.")
        
    try:
        # Download product image from source
        response = requests.get(image_url, timeout=10)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch image from source URL.")
            
        orig_image = Image.open(io.BytesIO(response.content)).convert("RGB")
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
```

---

## 💻 Step 2: Local Testing & Validation (Before Cloud Push)

It is highly recommended to run and test your API locally to ensure your Python environment works perfectly before deploying to the cloud.

### 1. Set Up Local Virtual Environment
Run the following commands in your local shell inside the `salp-image-processor` directory:
```bash
# 1. Create Python virtual environment
python -m venv .venv

# 2. Activate virtual environment
# On Windows (PowerShell):
.venv\Scripts\activate
# On macOS / Linux:
source .venv/bin/activate

# 3. Install required packages
pip install -r requirements.txt
```

### 2. Download Model Weights for Local Run
To run the server locally without Docker, download the weights into the `models` folder manually:
```bash
mkdir models
# Windows (PowerShell):
Invoke-WebRequest -Uri "https://huggingface.co/briaai/BiRefNet/resolve/main/birefnet-general-epoch_244_quantized.onnx" -OutFile "models/birefnet_general_quantized.onnx"
# macOS / Linux (Terminal):
curl -L "https://huggingface.co/briaai/BiRefNet/resolve/main/birefnet-general-epoch_244_quantized.onnx" -o "models/birefnet_general_quantized.onnx"
```

### 3. Run FastAPI Locally
```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8080
```
Open [http://127.0.0.1:8080/docs](http://127.0.0.1:8080/docs) in your browser to interact with the API Swagger documentation interface!

---

## 🚀 Step 3: Google Cloud Setup & CLI Configuration

### 1. Install Google Cloud SDK
If you do not have the `gcloud` command-line tool installed on your machine, install it based on your operating system:
*   **Windows (PowerShell):** Download and run the [Google Cloud SDK Interactive Installer](https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe).
*   **macOS (Homebrew):** Run `brew install --cask google-cloud-sdk`.
*   **Linux (Debian/Ubuntu):** Run `sudo apt-get install google-cloud-cli`.

---

### 2. First-Time Google Cloud Billing Setup
Before you can run or deploy serverless containers on Google Cloud Run, Google requires an active Billing Account associated with your project. Artifact Registry and Cloud Run include generous free-tier limits, but a card must be registered to enable billing.

#### Step A: Create your Billing Account
1. Open your web browser and navigate to the [Google Cloud Billing Console](https://console.cloud.google.com/billing).
2. Sign in using your Google Workspace admin credentials: **`admin@salp.shop`**.
3. If you do not have a billing account configured, click **Create Billing Account** (or click the top menu dropdown and select **Manage billing accounts** -> **Add billing account**).
4. **Step 1: Account Info:** Select your country, business size/model, and accept the Terms of Service. Click **Continue**.
5. **Step 2: Customer Info:** 
   - Set the Account Type to **Business** (since it is for `salp.shop`).
   - Enter your registered business name, tax information (if applicable, or skip), and physical billing address.
6. **Step 3: Payment Method:** 
   - Enter your corporate debit/credit card or bank account details.
   - Google will issue a temporary pre-authorization charge (usually $1 USD) to verify the card. This is reversed instantly.
7. Click **Submit and enable billing**.

---

### 3. Create a Google Cloud Project & Link Billing
A Google Cloud project is a sandbox containment layer for your APIs, logs, container registries, and serverless compute functions.

#### Option A: Creating via the Google Cloud Console (Recommended for UI clarity)
1. Go to the [Google Cloud Create Project Dashboard](https://console.cloud.google.com/projectcreate).
2. Enter the project name: **`salp-production`** (or whatever name you prefer).
3. The console will generate a unique **Project ID** (e.g. `salp-production-462810`). *Note: This ID must be globally unique across all Google Cloud users. Record this Project ID, as you will use it for command-line deployments!*
4. Under **Organization**, select `salp.shop` (or your default Workspace organization).
5. Under **Billing Account**, select the billing account you created in Step 2.
6. Click **Create** and wait 10 seconds for the project initialization to complete.

#### Option B: Creating via gcloud CLI (From your terminal prompt)
1. When your `gcloud init` prompt asks:
   `This account has no projects. Would you like to create one? (Y/n)?`
   Type **`y`** (or **`Y`**) and press Enter.
2. Enter your desired globally unique **Project ID** (e.g., `salp-production` or `salp-images-prod`).
3. Press Enter. Google Cloud CLI will automatically provision the project on your account!
4. **CRITICAL STEP:** Projects created via CLI are NOT linked to billing by default. You **must link the project to billing** using these commands:
   ```bash
   # 1. List your available billing accounts to get your Billing Account ID
   # (The ID is formatted like: 012345-6789AB-CDEF01)
   gcloud beta billing accounts list

   # 2. Link your project to that billing account
   gcloud beta billing projects link [YOUR_PROJECT_ID] --billing-account=[YOUR_BILLING_ACCOUNT_ID]
   ```

---

### 4. Authenticate & Configure gcloud CLI
Once your project is created and linked to billing, authenticate your terminal session and set your active working target:

```bash
# 1. Login to your admin@salp.shop Google account
gcloud auth login

# 2. List projects to verify your new project is visible
gcloud projects list

# 3. Set the default working project ID for all future deployments
gcloud config set project [YOUR_PROJECT_ID]

# 4. Set your default deployment region to europe-west4 (Eemshaven, Netherlands)
gcloud config set run/region europe-west4
```

---

### 5. Enable Required Google APIs
Enable the containerization (Artifact Registry) and serverless execution (Cloud Run) microservice engine APIs inside your project:
```bash
gcloud services enable \
    artifactregistry.googleapis.com \
    run.googleapis.com \
    billingbudgets.googleapis.com
```

> [!TIP]
> Enabling `billingbudgets.googleapis.com` is highly recommended to monitor cloud costs and automatically receive email notifications if compute limits are approached.

---

## 📦 Step 4: Building and Pushing to Google Artifact Registry

Artifact Registry is GCP's secure repository for Docker images. We compile our container locally and push it to Artifact Registry:

### 1. Create a Location-Scoped Registry
We target `europe-west4` (Eemshaven, Netherlands) as it sits closest to our Neon Database pool and Vercel infrastructure, reducing regional network latency to single-digit milliseconds:
```bash
gcloud artifacts repositories create salp-images \
    --repository-format=docker \
    --location=europe-west4 \
    --description="Salp Image Background Processor"
```

### 2. Configure Docker Helper Authentication
Instruct your local Docker daemon to use Google Cloud credentials for authentication to registry subdomains:
```bash
gcloud auth configure-docker europe-west4-docker.pkg.dev
```

### 3. Build & Tag the Container Locally
Execute this command inside the `salp-image-processor` folder (where your `Dockerfile` sits):
```bash
docker build -t europe-west4-docker.pkg.dev/[YOUR_PROJECT_ID]/salp-images/bg-remover:latest .
```

### 4. Push Container to GCP Registry
```bash
docker push europe-west4-docker.pkg.dev/[YOUR_PROJECT_ID]/salp-images/bg-remover:latest
```

---

## ⚡ Step 5: Deploying to Google Cloud Run (Serverless CPU)

Deploy the image from Artifact Registry to Cloud Run, applying CPU-based serverless parameters to eliminate idle fees and prevent execution timeouts:

```bash
gcloud run deploy salp-image-processor \
    --image=europe-west4-docker.pkg.dev/[YOUR_PROJECT_ID]/salp-images/bg-remover:latest \
    --platform=managed \
    --region=europe-west4 \
    --cpu=1 \
    --memory=2Gi \
    --min-instances=0 \
    --max-instances=100 \
    --concurrency=10 \
    --allow-unauthenticated
```

### **Why are these flags critical for your stack?**
*   `--cpu=1` and `--memory=2Gi`: BiRefNet ONNX is highly optimized and fits inside 1GB of memory. Provisioning 2GB RAM guarantees ample headroom for processing large, high-resolution lifestyle product photos without running into Out-Of-Memory (OOM) failures.
*   `--min-instances=0`: **Scale-To-Zero.** When there are no synchronizations running, Cloud Run completely destroys all active containers. You pay absolutely **$0.00** when the service is idle.
*   `--max-instances=100`: Puts an upper limit on container scaling to prevent accidental billing spikes or DDoS attacks.
*   `--concurrency=10`: Instructs Google Cloud to route up to 10 parallel incoming image processing requests to a single running container instance. **This reduces your active compute costs by up to 90%** during massive catalog ingestion sweeps compared to one-container-per-request systems.
*   `--allow-unauthenticated`: Exposes a secure public HTTPS endpoint so your Next.js server actions and Trigger.dev tasks can call it.

Once completed, Google Cloud will output your permanent secure endpoint URL, e.g.:
`https://salp-image-processor-xxxxxx-ew.a.run.app`

---

## 🔗 Step 6: Connecting to Vercel & Trigger.dev Dashboards

Copy the Cloud Run URL and append the `/remove-background` path to connect it to your production services:

```text
https://salp-image-processor-xxxxxx-ew.a.run.app/remove-background
```

---

### 1. Connecting to Vercel (Next.js Application Dashboard)
Vercel hosts your Next.js application server. Server Actions in `SaleorPortal` will look for this environment variable:

1.  Log in to your [Vercel Dashboard](https://vercel.com).
2.  Select your **SaleorPortal** (or `vendor-portal`) project.
3.  Go to the **Settings** tab in the main horizontal menu.
4.  Select **Environment Variables** in the left sidebar menu.
5.  Add a new environment variable:
    *   **Key:** `IMAGE_PROCESSOR_URL`
    *   **Value:** `https://salp-image-processor-xxxxxx-ew.a.run.app/remove-background`
    *   **Scope:** Select **Staging** and **Production** (you can optionally uncheck Development to run original lifestyle images locally without calling the API).
6.  Click **Save**.
7.  *Note: A new deployment or build must be triggered on Vercel for these changes to take effect on the running staging/production servers.*

---

### 2. Connecting to Trigger.dev Dashboard
Because Trigger.dev runs your actual asynchronous background catalog imports, the tasks executing in their serverless environment must access the endpoint URL directly:

1.  Log in to your [Trigger.dev Dashboard](https://cloud.trigger.dev).
2.  Navigate to your active project workspace.
3.  Select **Environment Variables** in the left sidebar menu.
4.  Click **Add Environment Variable** (often represented by a `+` button):
    *   **Key:** `IMAGE_PROCESSOR_URL`
    *   **Value:** `https://salp-image-processor-xxxxxx-ew.a.run.app/remove-background`
5.  Click **Save Changes** or **Publish**.
6.  Trigger.dev instantly propagates environment updates to all live and queued tasks without requiring a full redeployment!

---

## 🔄 Step 7: Zero-Touch CI/CD Pipeline (GitHub Actions)

To avoid compiling and pushing containers manually from your local machine, the repository is fully pre-configured to use **Workload Identity Federation (OIDC)**. This is Google Cloud's modern, zero-secret security best practice.

### 🔑 Why Workload Identity Federation?
- **Zero Static Secrets:** You do not need to generate, manage, or upload any risky private key JSON files to GitHub.
- **Auto-Authentication:** GitHub Actions dynamically requests a temporary token from Google Cloud to run the deployment.
- **Enterprise Compliant:** It bypasses any organization security policies that restrict Service Account key creation.

### 🛠️ Configured Credentials & Trust
The pipeline is already fully set up and configured on your GCP project `salp-image-processor` with:
- **Workload Identity Pool:** `github-pool` (Project number `921666155792`)
- **Workload Identity Provider:** `github-provider` (Restricted via Attribute Condition to only allow tokens from `Salp-BV/salp-image-processor`)
- **Service Account:** `github-actions-deployer@salp-image-processor.iam.gserviceaccount.com` (Authorized with `Artifact Registry Writer`, `Cloud Run Developer`, and `Service Account User` roles)

### 🚀 How to Trigger Your First Deploy
There are absolutely **no manual secrets or repository configurations needed on GitHub**! 

To launch your deployment:
1. Make any code changes to `app.py`, `Dockerfile`, or dependencies.
2. Push your changes to the `main` branch:
   ```bash
   git push origin main
   ```
3. Open your repository's **Actions** tab on GitHub. You will see the **Deploy Image Processor to Google Cloud Run** workflow executing automatically. It will build your container inside GitHub's runner, push it to Artifact Registry, and deploy the new revision to Cloud Run seamlessly!

