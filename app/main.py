from fastapi import FastAPI
from dotenv import load_dotenv

# Database imports
from app.core.database import engine, Base
from app.models import domain

# API Routers imports
from app.api.routes import asr

# 1. Load environment variables
load_dotenv()

# 2. Create database tables in Neon (if they don't exist yet)
Base.metadata.create_all(bind=engine)

# 3. Initialize the FastAPI application
app = FastAPI(
    title="AI Learning App API",
    description="Backend for the AI language learning Android application.",
    version="1.0.0"
)

# 4. Include our routers (Endpoints)
app.include_router(asr.router, tags=["Audio & Speech"])

# Optional: A simple health check endpoint at the root
@app.get("/", tags=["Health"])
def read_root():
    return {"status": "ok", "message": "Server is running perfectly!"}
