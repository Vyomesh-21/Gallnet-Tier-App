# GallNet Triage — Production Deployment Guide

This guide details the step-by-step procedure to deploy the **GallNet Triage** application to production using:
- **GitHub** for version and source control
- **Render** for the containerized FastAPI + TensorFlow machine learning backend
- **Vercel** for the high-performance React + Vite frontend

---

## 1. System Architecture

```
                  +-----------------------------------+
                  |           End User / UI           |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  |          Vercel Frontend          |
                  |     (React 18 + Vite + Tailwind)  |
                  |  https://gallnet-triage.vercel.app|
                  +-----------------+-----------------+
                                    |
                    HTTPS REST / CORS (Multipart)
                    VITE_API_URL: Render backend
                                    |
                                    v
                  +-----------------------------------+
                  |          Render Backend           |
                  |    (FastAPI + Python 3.11 Docker) |
                  | https://gallnet-backend.onrender  |
                  +-----------------+-----------------+
                                    |
                   In-Memory Preprocessing & Inference
                                    |
                                    v
                  +-----------------------------------+
                  |   Pretrained GallNet CNN Model    |
                  | (Self-Attention Residual CNN)     |
                  |  45.3 MB Keras 3 Model Checkpoint |
                  +-----------------------------------+
```

### Key Technical Characteristics
- **Privacy-Preserving**: Uploaded ultrasound scans are processed purely in-memory using `io.BytesIO`. Images are **never** written to local disk, temp storage, or persistent cloud databases.
- **Model Checkpoint**: The verified model checkpoint `models/model-031-0.979685-0.992509.keras` is 45.3 MB (well below GitHub's 100 MB file limit), allowing it to be tracked directly in Git without complex LFS setup.
- **CORS Security**: Cross-Origin Resource Sharing is locked down to authorized frontend domains via `FRONTEND_ORIGIN` (no unrestricted wildcards `*` in production).
- **Dynamic Port Routing**: The backend container automatically respects Render's dynamic `$PORT` environment variable via `CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]`.

---

## 2. Local Verification (Windows PowerShell)

Before deploying to the cloud, verify that both the backend and frontend function properly on your local workstation.

### A. Run Backend Tests
Run the automated pytest suite (validates model loading, 9-class mapping, 3-tier triage aggregation, and FastAPI endpoints):

```powershell
# Open PowerShell in the project root: c:\Users\Vyomesh\OneDrive\Desktop\gallnet-tier-app
python -m pytest backend/ -v
```

*Expected output: All 27 tests pass.*

### B. Verify Frontend Production Build
Validate that TypeScript compiles without errors and Vite bundles the production assets:

```powershell
npm.cmd --prefix frontend run build
```

*Expected output: `✓ built in ...` with compiled bundles inside `frontend/dist/`.*

### C. Run Locally with Windows Startup Scripts
To run both services locally for manual testing:
- **Backend**: Double-click `start_backend_windows.bat` (runs on `http://127.0.0.1:8000`)
- **Frontend**: Double-click `start_frontend_windows.bat` (runs on `http://localhost:5173`)

---

## 3. GitHub Source Control Setup

Initialize the Git repository and push your project to GitHub.

> [!NOTE]
> `.gitignore` has been pre-configured to safely include the verified model file `models/model-031-0.979685-0.992509.keras` (45.3 MB) while excluding node modules, caches, scratch files, and datasets.

### PowerShell Commands

```powershell
# 1. Navigate to the project root
cd "c:\Users\Vyomesh\OneDrive\Desktop\gallnet-tier-app"

# 2. Initialize Git
git init

# 3. Verify git status (confirm models/...keras is included, node_modules/ is ignored)
git status

# 4. Stage and commit files
git add .
git commit -m "feat: prepare GallNet Triage for production deployment on Render and Vercel"

# 5. Create a new repository on GitHub (e.g., https://github.com/<your-username>/gallnet-tier-app)
# Then link your remote and push:
git branch -M main
git remote add origin https://github.com/<your-username>/gallnet-tier-app.git
git push -u origin main
```

---

## 4. Backend Deployment on Render

Render will host the containerized FastAPI/TensorFlow backend using the root `Dockerfile`.

### Step 1: Create a Render Web Service
1. Log in to [Render](https://dashboard.render.com/).
2. In the top right corner, click **New +** and select **Web Service**.
3. Choose **Build and deploy from a Git repository**.
4. Select your `gallnet-tier-app` repository (or configure GitHub permissions if needed).

### Step 2: Configure Service Settings
Fill in the configuration fields:
- **Name**: `gallnet-backend` (or your preferred name)
- **Region**: Choose the region closest to you (e.g., *Oregon (US West)* or *Frankfurt (EU Central)*)
- **Branch**: `main`
- **Root Directory**: *(Leave blank / default to project root)*
- **Runtime**: **Docker**
- **Dockerfile Path**: `./Dockerfile`
- **Instance Type**: 
  - *Free* (512 MB RAM): Sufficient for demo testing (note: first inference after cold start takes ~30-50s).
  - *Starter* ($7/mo, 512 MB – 1 GB RAM): Recommended for continuous availability and multi-user medical imaging workloads.

### Step 3: Configure Environment Variables
Under the **Environment Variables** section, add:

| Key | Value | Purpose |
|---|---|---|
| `FRONTEND_ORIGIN` | `https://gallnet-triage.vercel.app` | Allows your Vercel frontend to call the API without CORS errors (you can update this once Vercel gives you your URL) |
| `TF_ENABLE_ONEDNN_OPTS` | `0` | Ensures reproducible numeric results on CPU |
| `PYTHONUNBUFFERED` | `1` | Stream Python logs immediately |

### Step 4: Health Check Configuration
Expand **Advanced Settings**:
- **Health Check Path**: `/health`
- **Auto-Deploy**: **Yes** (re-deploys automatically when you push to `main`)

### Step 5: Deploy
Click **Create Web Service**. 
Render will build the Docker container and start Uvicorn. Once the build log shows `Application startup complete`, note your live backend URL:
`https://gallnet-backend.onrender.com` (example).

---

## 5. Frontend Deployment on Vercel

Vercel will compile and host the React 18 + Vite application with global edge CDN distribution.

### Step 1: Import Project into Vercel
1. Log in to [Vercel](https://vercel.com/).
2. Click **Add New...** -> **Project**.
3. Select your GitHub account and click **Import** next to `gallnet-tier-app`.

### Step 2: Configure Build & Directory Settings
In the **Configure Project** screen:
- **Project Name**: `gallnet-triage` (or your preferred name)
- **Framework Preset**: **Vite**
- **Root Directory**: Click **Edit** and select **`frontend`** *(Critical: do not leave as project root)*
- **Build Command**: `npm run build` (default)
- **Output Directory**: `dist` (default)
- **Install Command**: `npm install` (default)

### Step 3: Configure Environment Variables
Expand **Environment Variables** and add:

| Key | Value | Note |
|---|---|---|
| `VITE_API_URL` | `https://gallnet-backend.onrender.com` | Replace with your actual Render backend URL (no trailing slash) |

### Step 4: Deploy
Click **Deploy**.
Vercel will build the frontend and assign a live production URL, such as:
`https://gallnet-triage.vercel.app`

---

## 6. Post-Deployment CORS Alignment

After Vercel finishes deploying:
1. Copy your public Vercel URL (e.g., `https://gallnet-triage.vercel.app`).
2. Go back to your [Render Dashboard](https://dashboard.render.com/) -> `gallnet-backend`.
3. Click **Environment**.
4. Set `FRONTEND_ORIGIN` to your exact Vercel URL:
   ```
   FRONTEND_ORIGIN = https://gallnet-triage.vercel.app
   ```
   *(If you also want local development support or custom domains, you can separate origins with commas: `https://gallnet-triage.vercel.app,http://localhost:5173`)*
5. Click **Save Changes**. Render will automatically redeploy the service with the updated CORS policy.

---

## 7. Production Verification & Smoke Testing

### 1. Test Backend Health Endpoint
Run in PowerShell or curl:
```powershell
curl.exe -i https://<your-render-backend-url>.onrender.com/health
```
Expected response:
```json
{
  "status": "online",
  "model_loaded": true,
  "model_name": "Self-Attention-Guided Residual CNN (Rashid et al., 2025)",
  "model_error": null,
  "mapping_verified": true,
  "mapping_status": "VERIFIED"
}
```

### 2. Inspect Interactive OpenAPI Docs
Open `https://<your-render-backend-url>.onrender.com/docs` in your browser to inspect and interact with the FastAPI documentation.

### 3. Verify Live Frontend Application
1. Open `https://<your-vercel-domain>.vercel.app` in your browser.
2. Check the header status badge in the top right:
   - Should indicate **Neural Engine Active** (green status dot).
3. Upload an ultrasound image (e.g., JPEG or PNG) and click **Analyze Ultrasound Scan**.
4. Verify:
   - Multiclass probability distribution renders accurately.
   - 3-tier surgical classification (Tier 0, Tier 1, Tier 2, or Review Required) is assigned.
   - Grad-CAM visual explainability heatmap overlay toggles between original and attribution view.
   - No errors in browser Developer Tools (F12) Console or Network tabs.

---

## 8. Troubleshooting & FAQ

### Q1: The frontend shows "Engine Offline" or "Inference service unavailable"
- **Cause A (Cold Start)**: On Render's Free tier, services spin down after 15 minutes of inactivity. The first incoming request will trigger a cold start that takes ~30–50 seconds to initialize TensorFlow. Wait 1 minute and refresh the page.
- **Cause B (CORS Mismatch)**: Open browser DevTools (F12) -> Console. If you see a CORS error, check that `FRONTEND_ORIGIN` on Render exactly matches your Vercel URL protocol and domain (including `https://` without a trailing `/`).
- **Cause C (VITE_API_URL Missing)**: Ensure `VITE_API_URL` was defined in Vercel before the build. If added after, trigger a **Redeploy** on Vercel so the environment variable is injected into the compiled bundle.

### Q2: Render deployment fails with Out of Memory (Exit Code 137)
- TensorFlow model deserialization requires ~400–600 MB of RAM during initial cold boot.
- If Render free tier hits memory limits:
  1. Upgrade to Render's **Starter** instance (1 GB RAM).
  2. Alternatively, verify `TF_ENABLE_ONEDNN_OPTS=0` is set to prevent memory expansion from oneDNN optimizations.

### Q3: Git push rejects the model file
- `models/model-031-0.979685-0.992509.keras` is 45.3 MB, which is under GitHub's 100 MB hard limit and pushes normally.
- If you have an existing Git LFS configuration or corporate proxy blocking large objects, you can track it via Git LFS:
  ```powershell
  git lfs install
  git lfs track "*.keras"
  git add .gitattributes
  ```

### Q4: Direct page refresh on Vercel gives 404
- Handled automatically by `frontend/vercel.json`, which rewrites all route paths to `/index.html` for single-page application handling.
