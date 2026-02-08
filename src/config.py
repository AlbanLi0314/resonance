"""
Configuration for Academic Matcher
All paths, model names, and constants defined here
"""
import os
from pathlib import Path

# Load .env file if exists
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

# ============================================================
# Directory Structure
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"

# Create directories if not exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# File Paths
# ============================================================
RESEARCHERS_JSON = RAW_DATA_DIR / "researchers.json"
RESEARCHERS_WITH_EMB_PKL = PROCESSED_DATA_DIR / "researchers_optimized.pkl"
FAISS_INDEX_FILE = INDEX_DIR / "faiss_optimized.index"

# ============================================================
# Model Configuration
# ============================================================
SPECTER2_MODEL_NAME = "allenai/specter2_base"
SPECTER2_ADAPTER_NAME = "allenai/specter2"
EMBEDDING_DIM = 768
MAX_TEXT_LENGTH = 512  # SPECTER2 token limit

# ============================================================
# Search Configuration
# ============================================================
RECALL_TOP_K = 20      # Stage 1: how many candidates to recall
FINAL_TOP_K = 5        # Stage 2: how many results to return
MIN_RAW_TEXT_LENGTH = 50  # Filter out researchers with too little info

# ============================================================
# LLM Configuration (Gemini)
# ============================================================
# Gemini 3 model for Hackathon (required)
GEMINI_MODEL = "gemini-3-flash-preview"
GEMINI_MODEL_PRO = "gemini-3-pro-preview"  # For complex reasoning tasks
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ============================================================
# Helper to print config
# ============================================================
def print_config():
    print("=" * 50)
    print("Academic Matcher Configuration")
    print("=" * 50)
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"RESEARCHERS_JSON: {RESEARCHERS_JSON}")
    print(f"RESEARCHERS_WITH_EMB_PKL: {RESEARCHERS_WITH_EMB_PKL}")
    print(f"FAISS_INDEX_FILE: {FAISS_INDEX_FILE}")
    print(f"SPECTER2_MODEL: {SPECTER2_MODEL_NAME}")
    print(f"GEMINI_MODEL: {GEMINI_MODEL}")
    print(f"GEMINI_API_KEY: {'SET' if GEMINI_API_KEY else 'NOT SET'}")
    print("=" * 50)

if __name__ == "__main__":
    print_config()
