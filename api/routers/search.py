"""
Search Router - Handles researcher search endpoints
"""
import time
import logging
from typing import List
from fastapi import APIRouter, HTTPException

from api.schemas import SearchRequest, SearchResponse, ResearcherResult

logger = logging.getLogger("academic_matcher.search")

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search_researchers(request: SearchRequest):
    """
    Search for researchers matching a query.

    Uses hybrid search (SPECTER2 dense + BM25 sparse) with RRF fusion.
    Optional: Query expansion and match explanations via Gemini.
    """
    from api.main import get_ml_models
    ml_models = get_ml_models()

    if not ml_models.get("hybrid_search"):
        raise HTTPException(status_code=503, detail="Search engine not initialized")

    start_time = time.time()

    try:
        hybrid_search = ml_models["hybrid_search"]
        search_query = request.query

        # Optional: Expand query with Gemini
        if request.expand_query:
            try:
                from src.query_expansion import expand_query
                search_query = expand_query(request.query)
                logger.info(f"Expanded query: {search_query[:100]}...")
            except Exception as e:
                logger.warning(f"Query expansion failed: {e}, using original query")

        # Perform hybrid search
        results = hybrid_search.search(
            query=search_query,
            top_k=request.top_k
        )

        # Optional: Add Gemini-generated explanations
        if request.include_explanation and results:
            try:
                from src.match_explainer import explain_matches_batch
                results = explain_matches_batch(request.query, results, top_k=request.top_k)
                logger.info(f"Generated explanations for {len(results)} results")
            except Exception as e:
                logger.warning(f"Match explanation failed: {e}")

        # Convert to response format
        researcher_results = []
        for r in results:
            researcher_results.append(ResearcherResult(
                name=r.get("name", "Unknown"),
                department=r.get("department"),
                research_interests=r.get("research_interests"),
                score=r.get("score"),
                papers=r.get("papers", [])[:3],
                raw_text=r.get("raw_text", "")[:500] if r.get("raw_text") else None,
                explanation=r.get("explanation"),
                confidence=r.get("confidence"),
                key_overlap=r.get("key_overlap"),
                match_type=r.get("match_type")
            ))

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Search '{request.query}' returned {len(results)} results in {elapsed_ms:.0f}ms")

        return SearchResponse(
            query=request.query,
            results=researcher_results,
            search_time_ms=elapsed_ms
        )

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/researchers", response_model=List[ResearcherResult])
async def list_researchers(limit: int = 10, offset: int = 0):
    """
    List all researchers (paginated).
    """
    from api.main import get_ml_models
    ml_models = get_ml_models()

    researchers = ml_models.get("researchers", [])

    # Paginate
    paginated = researchers[offset:offset + limit]

    return [
        ResearcherResult(
            name=r.get("name", "Unknown"),
            department=r.get("department"),
            research_interests=r.get("research_interests"),
            papers=r.get("papers", [])[:3]
        )
        for r in paginated
    ]


@router.get("/researcher/{name}", response_model=ResearcherResult)
async def get_researcher(name: str):
    """
    Get a specific researcher by name.
    """
    from api.main import get_ml_models
    ml_models = get_ml_models()

    researchers = ml_models.get("researchers", [])

    # Find by name (case-insensitive)
    for r in researchers:
        if r.get("name", "").lower() == name.lower():
            return ResearcherResult(
                name=r.get("name", "Unknown"),
                department=r.get("department"),
                research_interests=r.get("research_interests"),
                papers=r.get("papers", []),
                raw_text=r.get("raw_text")
            )

    raise HTTPException(status_code=404, detail=f"Researcher '{name}' not found")
