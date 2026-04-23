# Vercel & Hybrid AI Deployment Guide

This project is configured for **Hybrid Deployment**:
1.  **Frontend/Backend**: Vercel (Node.js)
2.  **AI Engine**: Railway/Render (Python)
3.  **Database/Storage**: Supabase

---

## Phase 1: Deploying the AI Service (Python)

Vercel's Serverless Functions cannot host the `InsightFace` models due to size and memory limits. You must host the `python_daemon.py` on a platform that supports persistent processes.

### Deployment on Railway (Recommended)
1.  Connect your GitHub repository to [Railway.app](https://railway.app).
2.  Set the **Start Command** to: `python python_daemon.py`
3.  Ensure the `requirements.txt` includes `insightface`, `onnxruntime`, `opencv-python-headless`, and `flask`.
4.  Expose the service and copy the **Public URL** (e.g., `https://ai-service-production.up.railway.app`).

---

## Phase 2: Deploying to Vercel (Node.js)

1.  Connect your repository to [Vercel](https://vercel.com).
2.  Add the following **Environment Variables**:
    *   `AI_SERVICE_URL`: The URL from Phase 1.
    *   `DATABASE_URL`: Your Supabase PostgreSQL connection string.
    *   `SUPABASE_URL`: Your Supabase Project URL.
    *   `SUPABASE_ANON_KEY`: Your Supabase Anon Key.
    *   `SESSION_SECRET`: A long random string.
    *   `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_WHATSAPP_FROM`: Your Twilio credentials.
3.  Deploy.

---

## Phase 3: Storage Configuration

Since Vercel has no persistent local storage:
- **Temporary Uploads**: The app automatically uses `/tmp` for processing images.
- **Permanent Images**: The app is already configured to upload newly added students to **Supabase Storage**. Ensure you have a bucket named `faces` in your Supabase project with public access enabled.

---

## Critical Environment Variables

| Variable | Description | Source |
| :--- | :--- | :--- |
| `AI_SERVICE_URL` | Route to the external Python AI service | Railway/Render |
| `DATABASE_URL` | PostgreSQL connection string | Supabase |
| `SUPABASE_URL` | API endpoint for storage/auth | Supabase |
| `SUPABASE_ANON_KEY` | Public key for Supabase client | Supabase |
| `VERCEL` | Automatically set to `1` by Vercel | Automatic |

---

> [!NOTE]
> The `SurveillanceEngine` tracking ticker is disabled on Vercel to prevent resource leaks. High-accuracy single-frame identification remains fully functional.
