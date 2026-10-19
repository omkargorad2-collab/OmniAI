# OmniAI — Multimodal AI Studio

A mobile-responsive FastAPI web app with LLM chat, lightweight NLP intent detection, PDF/text/image Q&A (via vision-capable LLM), image generation, basic local photo editing, and optional asynchronous text-to-video through Replicate.

## Important: what works and what requires configuration

- **Without keys:** website UI, NLP intent classification (server-side), image enhance/grayscale/blur/rotate/mirror, chat history storage.
- **With LLM_API_KEY:** chat, step-by-step doubt solving, PDF and text questions; image questions require a vision-capable model.
- **With IMAGE_API_KEY or LLM_API_KEY and compatible image endpoint:** image generation.
- **With REPLICATE_API_TOKEN and REPLICATE_VIDEO_MODEL:** video generation (only for compatible Replicate models that accept an `input.prompt` field). Model-specific parameters may need code changes.
- **Not implemented:** generative Nano Banana-style image edits, video editing, signup/authentication, cloud persistence, RAG/vector search, voice mode, admin dashboard. These require further work.

## Run locally

1. Install Python 3.10+.
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env`, fill in your API keys.
4. `uvicorn main:app --host 0.0.0.0 --port 8000`
5. Open `http://localhost:8000`.

## Deploy from Android via GitHub + Render

1. Unzip and upload the **contents** of `omniai/` to a GitHub repository (ensure `main.py` and `requirements.txt` are in the repository root).
2. Create a Render **Web Service**, connect your repository, choose Python runtime.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables in Render dashboard: `LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL`, optionally image and Replicate variables.
6. Deploy and open the Render URL.

**Security:** Never put API keys into HTML, GitHub, or frontend JS. Do not commit `.env`. This is a prototype; add authentication, per-user authorization, rate limiting, content moderation, and persistent storage before public launch. SQLite and generated files on Render ephemeral storage can disappear on redeploy/restart. Uploaded PDF text is sent to your configured LLM provider. File upload cap defaults to 15 MB. Generative API requests may incur charges.
