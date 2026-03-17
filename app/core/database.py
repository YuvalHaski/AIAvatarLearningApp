import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Get the connection string we pasted in the .env file
SQLALCHEMY_DATABASE_URL = os.environ.get("DATABASE_URL")

# Create the SQLAlchemy Engine - The engine is responsible for establishing the actual connection to Neon.
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Create a Session class.
# Every time we want to read or write to the DB (e.g., when a user logs in),
# we will create a new "Session" from this class.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# This is the Base class. All of our database models (Tables) will inherit from this.
Base = declarative_base()

# A Dependency function that we will use in our FastAPI routes.
# It creates a new database session for each request and closes it when the request is done.
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()