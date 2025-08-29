import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


# Load .env from project root and backend directory if present
ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
load_dotenv(BACKEND_DIR / ".env", override=False)


@dataclass
class Settings:
    DB_PATH: str = os.getenv(
        "DB_PATH",
        "/Users/takashiui/Documents/Python/stock_analysis/stock_data.db",
    )
    API_PORT: int = int(os.getenv("API_PORT", "8082"))
    WEB_PORT: int = int(os.getenv("WEB_PORT", "4002"))
    CORS_ORIGINS_STR: str = os.getenv("CORS_ORIGINS", "http://localhost:4002")
    CORS_ORIGINS: list[str] = field(default_factory=list)
    CACHE_PATH: Path = BACKEND_DIR / "db" / "schema_cache.json"


settings = Settings()
if not settings.CORS_ORIGINS:
    settings.CORS_ORIGINS = [s.strip() for s in settings.CORS_ORIGINS_STR.split(",") if s.strip()]
