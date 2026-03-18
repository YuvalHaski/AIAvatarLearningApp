from fastapi import FastAPI
from dotenv import load_dotenv

# Database imports
from app.core.database import engine, Base
from app.models import domain

# API Routers imports
from app.api.routes import asr

# Load environment variables
load_dotenv()

# Initialize the FastAPI application
app = FastAPI(
    title="AI Learning App API",
    description="Backend for the AI language learning Android application.",
    version="1.0.0"
)

# Include our routers (Endpoints)
app.include_router(asr.router, tags=["Audio & Speech"])

# Optional: A simple health check endpoint at the root
@app.get("/", tags=["Health"])
def read_root():
    return {"status": "ok", "message": "Server is running perfectly!"}
