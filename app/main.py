from fastapi import FastAPI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Firebase SDK:
import firebase_admin
from firebase_admin import credentials

# API Routers imports
from app.api.routes import categories, progress, lessons, asr


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

# Initialize the FastAPI application
app = FastAPI(
    title="AI Learning App API",
    description="Backend for the AI language learning Android application.",
    version="1.0.0"
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
