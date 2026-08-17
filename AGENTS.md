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
pytest
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

- **Coolify Deployment**: Containerized builds are managed by Coolify on Hetzner Node 1 via GitHub Container Registry (`ghcr.io/salp-bv/salp-image-processor:staging` and `:main`).
- **Media Output**: Processed image assets are served via Bunny.net CDN (`salp-media-cdn.b-cdn.net`).
