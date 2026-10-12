"""Central configuration for the International Student RAG project."""

from pathlib import Path


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
STRUCTURED_DIR = PROJECT_ROOT / "data" / "structured"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"

EVALUATION_DIR = PROJECT_ROOT / "evaluation"
RESULTS_DIR = EVALUATION_DIR / "results"


# ============================================================
# ChromaDB
# ============================================================

COLLECTION_NAME = "international_student_kb"


# ============================================================
# Ollama models
# ============================================================

# Main generation model
LLM_MODEL = "qwen3:14b"

# Embedding model
EMBEDDING_MODEL = "qwen3-embedding:0.6b"


# ============================================================
# Retrieval settings
# ============================================================

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

DEFAULT_TOP_K = 5

# Empirically selected for the current evaluation set.
MAX_DISTANCE = 0.95


# ============================================================
# Universities
# ============================================================

UNIVERSITIES = [
    "All",
    "RMIT University",
    "University of Melbourne",
    "Monash University",
    "Australia",
]


# ============================================================
# Knowledge-base categories
# ============================================================

CATEGORIES = [
    "All",
    "course",
    "accommodation",
    "international_students",
    "fees",
    "work_rights",
    "general",
]