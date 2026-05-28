"""Central configuration. Reads environment variables once at import time.

Keeping all env access here (instead of scattered os.environ.get calls) makes it
obvious what the app needs to run and gives one place to validate it.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- Database ---
    DATABASE_URL: str | None = os.environ.get("DATABASE_URL")

    # --- Azure Speech (Pronunciation Assessment) ---
    AZURE_SPEECH_KEY: str | None = os.environ.get("AZURE_SPEECH_KEY")
    AZURE_SPEECH_REGION: str | None = os.environ.get("AZURE_SPEECH_REGION")

    # --- Stage 2 feedback model inference service ---
    # vLLM (or compatible) OpenAI-style endpoint, e.g. http://gpu-host:8000/v1
    FEEDBACK_MODEL_URL: str | None = os.environ.get("FEEDBACK_MODEL_URL")
    FEEDBACK_MODEL_NAME: str = os.environ.get("FEEDBACK_MODEL_NAME", "feedback-model")
    FEEDBACK_MODEL_TIMEOUT: float = float(os.environ.get("FEEDBACK_MODEL_TIMEOUT", "8.0"))

    # --- OpenAI (offline dataset generation only; not used at request time) ---
    OPENAI_API_KEY: str | None = os.environ.get("OPENAI_API_KEY")

    def require_azure(self) -> None:
        """Fail loudly if Azure speech credentials are missing."""
        missing = [
            name
            for name in ("AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION")
            if not getattr(self, name)
        ]
        if missing:
            raise RuntimeError(
                f"Missing required environment variables: {', '.join(missing)}"
            )


settings = Settings()
