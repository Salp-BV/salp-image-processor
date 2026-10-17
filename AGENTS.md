# Agentic Onboarding: Salp Image Processor Microservice

> **Goal**: High-performance Python FastAPI image background removal and studio contact shadow processing microservice powered by **Two-Stage YOLOv8x-seg + BiRefNet PyTorch FP16 CUDA on Verda Serverless NVIDIA L40S GPUs** (`app_gpu.py`, Python 3.11) with edge color decontamination (`decontaminate_and_despill`).

## 1. Project Identity

- **Stack**:
  - **Framework**: FastAPI (Python 3.11)
  - **ML Engine**:
    - **GPU (Primary)**: Two-Stage YOLOv8x-seg + BiRefNet PyTorch FP16 accelerated via CUDA on NVIDIA L40S (`app_gpu.py`, `Dockerfile.gpu`) with sub-pixel chroma neutralization and edge color decontamination (`decontaminate_and_despill`).
    - **CPU (Cold Standby Fallback)**: Standalone container using Python 3.10-slim-bookworm + `u2net.onnx` (`app.py`, `Dockerfile`).
  - **Image Processing & Compositing**: PIL (Pillow), NumPy, Torchvision
- **Deployment & Architecture**:
  - **Primary**: Verda Serverless `NVIDIA L40S` GPU container deployments (`Dockerfile.gpu`, PyTorch FP16, Scale-to-Zero). Direct endpoints: `https://containers.datacrunch.io/salp-img-production` & `https://containers.datacrunch.io/salp-img-staging`.
  - **Cold Standby Fallback**: Standalone CPU `Dockerfile` (`u2net.onnx`). Legacy Hetzner Node 1 Coolify PaaS routes (`image.salp.shop` / `staging.image.salp.shop`) are decommissioned in Coolify and DNS in favor of direct Verda serverless GPU endpoints.

## Git Workflow & Branching Standards

> **Goal**: Maintain a clean, linear git history on staging, zero merge conflicts, and clean production promotions.

### 1. Branching Topology
- **`main`**: Production environment. **Locked**. Changes arrive ONLY via Promotion PRs from `staging`.
- **`staging`**: Integration environment. **Locked**. Changes arrive ONLY via Feature PRs from feature branches.
- **Feature Branches**: ALWAYS branch from latest `staging`. Format: `feat/short-name`, `fix/issue-description`, `chore/task-name`.

### 2. Autonomous Agent Git Protocol (Step-by-Step)

#### Step 1: Initialize Feature Branch
```bash
git checkout staging
git pull --ff-only origin staging
git checkout -b feat/your-feature-name
```

#### Step 2: Keep Feature Branch Synchronized with Staging
Before opening a PR or committing final changes, rebase your feature branch ONTO latest staging:
```bash
git fetch origin staging
git rebase origin/staging
```

#### Step 3: Handling Rebase Conflicts (Deterministic Recovery)
If `git rebase` reports a conflict:
1. Run `git status` to identify conflicting files (`both modified`).
2. Open each conflicting file. Remember: `HEAD` is the incoming upstream staging commit; the conflicting block is your feature commit.
3. Resolve the conflict, preserving all valid business logic.
4. Stage resolved files: `git add <file>`
5. Continue rebase: `git rebase --continue`
6. If rebase becomes corrupted, abort safely: `git rebase --abort`
7. **NEVER** run `git rebase --skip`. **NEVER** force-push to `staging` or `main`.

#### Step 4: Pre-Push Quality Gate
Before pushing to remote, ensure all checks pass locally:
```bash
pytest # or .\.venv\Scripts\pytest on Windows
```

#### Step 5: Push and Open PR
```bash
git push -u origin feat/your-feature-name
```

#### Step 6: Post-Merge Workspace Cleanup (Return to Clean Staging)
After your PR is merged into staging on GitHub:
```bash
git checkout staging
git pull --ff-only origin staging
git branch -d feat/your-feature-name
git fetch --prune
```

---

### 3. PR Merge Policy: Two-Tier Strategy

| PR Type | Route | Allowed Merge Strategy | Prohibited Strategy | Reason |
| :--- | :--- | :--- | :--- | :--- |
| **Feature / Fix PR** | `feat/*` $\rightarrow$ `staging` | **Squash and merge** | Merge Commit, Rebase | Compresses iterative WIP commits into 1 clean atomic commit on `staging`. |
| **Promotion PR** | `staging` $\rightarrow$ `main` | **Create a merge commit** (or Fast-Forward) | ❌ **NEVER Squash and merge** | Preserves exact commit IDs and history graph between staging and main. |

> [!CAUTION]
> **NEVER Squash `staging` into `main`**. Squashing between long-lived branches destroys Git ancestry, causing chronic 3-way merge conflicts on all future releases.

---

### 4. Required PR / Commit Message Format

Your PR title and description become the final commit message on `staging`. It MUST follow this structure:

```text
<type>: <imperative summary in present tense>

Main Objective: <1-2 concise sentences explaining the business/technical purpose>

Key Changes:
- <Module/Area>: <Specific change description>
- <Module/Area>: <Specific change description>

Closes #<issue_number>
```

**Valid Types:** `feat:`, `fix:`, `chore:`, `refactor:`, `test:`, `docs:`, `perf:`, `ci:`

## 2. Infrastructure & Environment Context

- **Verda Serverless GPU (Primary)**:
  - **Direct Endpoints**: Scaled-to-zero serverless GPU container deployments accessible directly via:
    - Production: `https://containers.datacrunch.io/salp-img-production`
    - Staging: `https://containers.datacrunch.io/salp-img-staging`
  - **Dual-Header Authentication**: Secure zero-trust API gating accepting either:
    - `Authorization: Bearer <token>`
    - `X-Api-Key: <token>`
    - Validated in constant-time via `secrets.compare_digest()` against `IMAGE_PROCESSOR_API_KEY`, `VERDA_API_KEY`, or `RUNPOD_API_KEY`.
  - **Hardware & Scale-to-Zero Provisioning**: Dedicated NVIDIA L40S GPU per replica (0 to 4 replicas in production, 0 to 2 in staging, scaling down after 300s idle).
  - **Automated Provisioning**: Managed via `scripts/provision_verda.py` and GitHub Actions `.github/workflows/deploy-verda.yml`. Deployment images: `ghcr.io/salp-bv/salp-image-processor:gpu-main` (production) and `ghcr.io/salp-bv/salp-image-processor:gpu-staging` (staging).
  - **Warm Cache & Performance SLA**: `"should_use_cached_image": True` enabled by default for rapid wake-up (~25s) from scale-to-zero state. Warm inference latency SLA < 2.5s.
- **Cold Standby CPU Architecture**:
  - Standalone CPU `Dockerfile` uses Python 3.10-slim-bookworm + `u2net.onnx` (168MB) as a cold standby fallback image.
  - Legacy Hetzner Node 1 Coolify PaaS routes (`image.salp.shop` / `staging.image.salp.shop`) are decommissioned in Coolify and DNS in favor of direct Verda serverless GPU endpoints.
- **Pipeline Architecture & Invariants**:
  - **Stage 1 (Global Alpha Matting)**: Full-frame 1024x1024 BiRefNet PyTorch FP16 segmentation. Preserves complete physical geometries (wireframes, slender stems, headphone headband arches). Smooth noise floor thresholding maps `[0.15, 0.85]` to `[0.0, 1.0]`.
  - **Stage 2 (Color Decontamination & Despill)**: Vectorized edge chroma neutralization (`decontaminate_and_despill`). Samples background color from image perimeter where alpha < 0.05, then applies mathematical unmixing `F = (I - (1 - alpha) * B) / alpha` on semi-transparent edge pixels (`0.02 < alpha < 0.92`) to eliminate ambient colored backdrop bleed.
  - **Stage 3 (Canvas Normalization & Studio Contact Shadow)**: Auto-centers foreground with 85% proportional fit on a 1:1 square canvas (clamped between 1000px and 2048px). Guarantees 100% pure `#FFFFFF` `[255, 255, 255]` perimeter and corner purity. Synthesizes a subtle, soft Gaussian-blurred ground contact shadow directly beneath the base of the object.
- **Media Output**: Processed image assets are served via Bunny.net CDN (`salp-media-cdn.b-cdn.net`).
- **Observability**: GlitchTip / Sentry error tracking with automatic PII header scrubbing (`Authorization` and `X-Api-Key` scrubbed before transmission).
