import os
import logging
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CASES_DIR = PROJECT_ROOT / "cases"
CASES_FILE = CASES_DIR / "cases.json"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

load_dotenv(PROJECT_ROOT / ".env")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logging.getLogger("chromadb.telemetry.product.posthog").disabled = True


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    ingestion_batch_size: int = int(os.getenv("INGESTION_BATCH_SIZE", "100"))
    retriever_k: int = int(os.getenv("RETRIEVER_K", "6"))
    cbr_threshold: float = float(os.getenv("CBR_THRESHOLD", "0.45"))


settings = Settings()
