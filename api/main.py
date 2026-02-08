"""
FastAPI Application for Academic Matcher
Main entry point with lifespan management for ML models.
"""
import os
import sys
import time
import pickle
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.schemas import HealthResponse
from api.routers import search, chat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("academic_matcher")

# Global state for ML models
ml_models: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for loading/unloading ML models.
    Models are loaded once at startup and shared across all requests.
    """
    logger.info("=" * 50)
    logger.info("Starting Academic Matcher API...")
    logger.info("=" * 50)

    start_time = time.time()

    try:
        # Load SPECTER2 encoder
        logger.info("Loading SPECTER2 encoder...")
        from src.embedding import Specter2Encoder
        encoder = Specter2Encoder()
        encoder.load()
        ml_models["encoder"] = encoder
        logger.info("✓ SPECTER2 encoder loaded")

        # Load researcher data
        logger.info("Loading researcher data...")
        pkl_path = PROJECT_ROOT / "data" / "processed" / "researchers_optimized.pkl"
        with open(pkl_path, "rb") as f:
            researchers = pickle.load(f)
        ml_models["researchers"] = researchers
        logger.info(f"✓ Loaded {len(researchers)} researchers")

        # Load FAISS index
        logger.info("Loading FAISS index...")
        import faiss
        index_path = PROJECT_ROOT / "data" / "index" / "faiss_optimized.index"
        index = faiss.read_index(str(index_path))
        ml_models["index"] = index
        logger.info(f"✓ FAISS index loaded ({index.ntotal} vectors)")

        # Initialize hybrid search
        logger.info("Initializing hybrid search...")
        from src.hybrid_search import HybridSearch
        hybrid_search = HybridSearch(researchers, encoder)
        ml_models["hybrid_search"] = hybrid_search
        logger.info("✓ Hybrid search initialized")

        # Initialize Gemini client
        logger.info("Initializing Gemini client...")
        from src.gemini_client import GeminiClient
        gemini = GeminiClient()
        if gemini.test_connection():
            ml_models["gemini"] = gemini
            logger.info("✓ Gemini client connected")
        else:
            logger.warning("⚠ Gemini client connection failed, chat features disabled")

        elapsed = time.time() - start_time
        logger.info("=" * 50)
        logger.info(f"All models loaded in {elapsed:.1f}s")
        logger.info("API ready to serve requests!")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down, cleaning up resources...")
    ml_models.clear()
    logger.info("Cleanup complete.")


# Create FastAPI app
app = FastAPI(
    title="Cornell Academic Matcher",
    description="AI-powered researcher matching for Cornell University",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware (allow all for hackathon)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(chat.router, prefix="/api", tags=["chat"])


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Cornell Academic Matcher",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if ml_models.get("encoder") else "loading",
        model_loaded=ml_models.get("encoder") is not None,
        researchers_count=len(ml_models.get("researchers", []))
    )


# Export ml_models for routers
def get_ml_models() -> Dict[str, Any]:
    """Get the shared ML models dictionary"""
    return ml_models


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
