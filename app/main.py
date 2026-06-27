from contextlib import asynccontextmanager
import threading

import httpx
from fastapi import FastAPI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Firebase SDK:
import firebase_admin
from firebase_admin import credentials

# API Routers imports
from app.api.routes import categories, progress, lessons, asr
from app.core.config import settings


# Load environment variables
load_dotenv()

# ==========================================
# FIREBASE INITIALIZATION
# ==========================================
# We wrap it in a try-except to prevent crashing if it's already initialized (important for reloading)
try:
    cred = credentials.Certificate("firebase-credentials.json")
    firebase_admin.initialize_app(cred)
    print("Firebase Admin SDK initialized successfully!")
except ValueError:
    # App is already initialized (happens during uvicorn auto-reloads)
    pass

# ==========================================
# FEEDBACK MODEL WARMUP
# ==========================================
def _warm_feedback_model() -> None:
    """Preload the Stage 2 feedback model into Ollama so the first real request
    is warm. A cold load reads ~2.3 GB from disk (~40s on this machine), so we
    run it in a background thread to avoid blocking server startup. No-ops if the
    model service isn't configured or is unreachable."""
    if not settings.FEEDBACK_MODEL_URL:
        return
    url = settings.FEEDBACK_MODEL_URL.rstrip("/") + "/chat/completions"
    try:
        httpx.post(
            url,
            json={
                "model": settings.FEEDBACK_MODEL_NAME,
                "messages": [{"role": "user", "content": "ok"}],
                "max_tokens": 1,
            },
            timeout=120.0,
        )
        print("Feedback model warmup complete.")
    except Exception as exc:
        print(f"Feedback model warmup skipped: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model in the background; uvicorn starts serving immediately.
    threading.Thread(target=_warm_feedback_model, daemon=True).start()
    yield


# Initialize the FastAPI application
app = FastAPI(
    title="AI Learning App API",
    description="Backend for the AI language learning Android application.",
    version="1.0.0",
    lifespan=lifespan,
)

# ==========================================
# CORS CONFIGURATION
# ==========================================
# This allows external clients (like your web browser testing Swagger) to communicate with the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# ROUTERS INCLUSION
# ==========================================
# Here we attach all the endpoints we built to the main app
app.include_router(categories.router)
app.include_router(progress.router)
app.include_router(lessons.router)
app.include_router(asr.router, tags=["Audio & Speech"])

# Optional: A simple health check endpoint at the root
@app.get("/", tags=["Health"])
def read_root():
    return {"status": "ok", "message": "Server is running perfectly!"}
