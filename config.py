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

# Bump this whenever the ingestion format/metadata changes. rag.py refuses
# to run against an index built by an older ingestion script, because an
# outdated index silently produces wrong university labels.
INDEX_VERSION = 3


# ============================================================
# Ollama models
# ============================================================

# Main generation model. qwen3:8b is a faster drop-in alternative if the
# 14B model is too slow on your laptop.
LLM_MODEL = "qwen3:14b"

# Embedding model
EMBEDDING_MODEL = "qwen3-embedding:0.6b"

# Keep both models loaded in memory between questions. Ollama's default is
# to unload a model after 5 idle minutes, which makes the next question pay
# the full model-load cost again (several seconds for a 14B model).
OLLAMA_KEEP_ALIVE = "30m"

# Generation limits for the LLM path. A grounded answer for this domain is
# short; capping output and context keeps response time predictable.
LLM_NUM_PREDICT = 600
LLM_NUM_CTX = 4096


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

# Label used for Australia-wide government sources (Home Affairs, Fair Work,
# Study Australia). It is NOT a university and is never used as a
# university filter.
GENERAL_ENTITY = "Australia"

# Single source of truth for every university in the knowledge base.
#   key      -> folder name under data/raw/
#   name     -> canonical label stored in ChromaDB metadata and the CSV
#   short    -> short display label
#   aliases  -> phrases a student might type (matched on word boundaries)
UNIVERSITY_REGISTRY = {
    "rmit": {
        "name": "RMIT University",
        "short": "RMIT",
        "aliases": [
            "rmit", "rmit university", "rmit uni",
            "royal melbourne institute of technology",
        ],
    },
    "unimelb": {
        "name": "The University of Melbourne",
        "short": "UniMelb",
        "aliases": [
            "university of melbourne", "the university of melbourne",
            "unimelb", "uni melb", "uni of melbourne", "melbourne uni",
            "melbourne university", "uom",
        ],
    },
    "monash": {
        "name": "Monash University",
        "short": "Monash",
        "aliases": ["monash", "monash university", "monash uni"],
    },
    "deakin": {
        "name": "Deakin University",
        "short": "Deakin",
        "aliases": ["deakin", "deakin university", "deakin uni"],
    },
    "latrobe": {
        "name": "La Trobe University",
        "short": "La Trobe",
        "aliases": [
            "la trobe", "latrobe", "la-trobe", "la trobe university",
            "latrobe university", "ltu",
        ],
    },
    "swinburn": {
        "name": "Swinburne University of Technology",
        "short": "Swinburne",
        "aliases": [
            "swinburne", "swinburn", "swinburne university",
            "swinburne university of technology", "swin uni",
        ],
    },
    "acu": {
        "name": "Australian Catholic University",
        "short": "ACU",
        "aliases": [
            "acu", "australian catholic university", "australian catholic",
        ],
    },
}

# Any spelling of a university found in source data -> canonical label.
UNIVERSITY_NAME_FIXES = {
    "University of Melbourne": "The University of Melbourne",
    "Swinburne University": "Swinburne University of Technology",
    "Swinburne": "Swinburne University of Technology",
    "La Trobe": "La Trobe University",
    "ACU": "Australian Catholic University",
}

UNIVERSITY_NAMES = [entry["name"] for entry in UNIVERSITY_REGISTRY.values()]


# ============================================================
# Courses
# ============================================================

# key -> display name + phrases a student might type (matched on word
# boundaries, case-insensitive). "case_sensitive" aliases only match when
# typed in capitals, so "IT"/"AI" match but "it"/"ai" inside a sentence such
# as "is it online?" do not.
#
# The key is stored in university_courses.csv (course_key) and in ChromaDB
# metadata. Course page snapshots are named data/raw/<uni>/course.pdf (Master
# of Data Science) or data/raw/<uni>/course_<key>.pdf (other courses).
COURSE_REGISTRY = {
    "data_science": {
        "name": "Master of Data Science",
        "aliases": ["data science", "master of data science", "mds", "datasc"],
        "case_sensitive": [],
    },
    "it": {
        "name": "Master of Information Technology",
        "aliases": [
            "information technology", "master of information technology",
            "master of it", "masters of it", "masters in it", "master in it",
            "mit", "it course", "it degree", "it masters", "it master",
        ],
        "case_sensitive": ["IT"],
    },
    "business_analytics": {
        "name": "Master of Business Analytics",
        "aliases": [
            "business analytics", "master of business analytics",
            "business analytic", "analytics",
        ],
        "case_sensitive": [],
    },
    "cyber": {
        "name": "Master of Cyber Security",
        "aliases": [
            "cyber security", "cybersecurity", "cyber-security", "cyber",
            "master of cyber security", "master of cybersecurity",
        ],
        "case_sensitive": [],
    },
    "ai": {
        "name": "Master of Artificial Intelligence",
        "aliases": [
            "artificial intelligence", "master of artificial intelligence",
            "applied artificial intelligence", "ai",
        ],
        "case_sensitive": ["AI"],
    },
}

DEFAULT_COURSE_KEY = "data_science"
COURSE_NAMES = {key: entry["name"] for key, entry in COURSE_REGISTRY.items()}

UNIVERSITIES = ["All", *UNIVERSITY_NAMES, GENERAL_ENTITY]


# ============================================================
# Knowledge-base categories
# ============================================================

CATEGORIES = [
    "All",
    "course",
    "accommodation",
    "international_students",
    "work_rights",
    "general",
]
