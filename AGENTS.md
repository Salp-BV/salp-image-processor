# Agentic Onboarding: Salp Image Processor Microservice

> **Goal**: A high-performance Python FastAPI image background removal and contact shadow processing microservice (BiRefNet ONNX CPU).

## 1. Project Identity

- **Stack**:
  - **Framework**: FastAPI (Python 3.11)
  - **ML Engine**: ONNX Runtime CPU (`birefnet_general_quantized.onnx`)
  - **Image Compositing**: PIL (Pillow) & NumPy
- **Deployment**:
  - **Primary**: Coolify PaaS (Hetzner Node 1) / GitHub Container Registry (`ghcr.io`).
  - **Secondary Backup**: Google Cloud Run (Europe `europe-west4`).

## Git Workflow & Commit Standards

> **Goal**: Maintain a clean, linear git history and rich GitHub Pull Request context.

### 1. Branching Strategy
- **`main` / `master`**: Production environment. **Locked**, requires PRs from `staging`.
- **`staging`**: Integration environment. **Locked**, requires PRs from feature branches.
- **Feature Branches**: ALWAYS branch from `staging`. Use descriptive names (e.g., `feat/image-processor-hardening`).
- **Syncing**: Keep feature branches up to date by rebasing `staging` onto them, rather than creating messy merge commits.

### 2. GitHub Pull Requests (PRs)
- **Target**: Always open your PR against `staging`.
- **Merge Strategy**: We enforce **Squash and merge**. This ensures all minor/wip commits are squashed into a single, highly detailed commit in the `staging` history.

## 2. Infrastructure & Environment Context

- **Coolify Deployment**: Containerized builds are managed by Coolify on Hetzner Node 1 via GitHub Container Registry (`ghcr.io/salp-bv/salp-image-processor:staging` and `:main`).
- **Media Output**: Processed image assets are served via Bunny.net CDN (`salp-media-cdn.b-cdn.net`).
