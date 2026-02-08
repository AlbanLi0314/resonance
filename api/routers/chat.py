"""
Chat Router - Handles Gemini-powered conversation endpoints
"""
import logging
from fastapi import APIRouter, HTTPException

from api.schemas import (
    ChatRequest, ChatResponse,
    ExplainRequest, ExplainResponse
)

logger = logging.getLogger("academic_matcher.chat")

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_with_gemini(request: ChatRequest):
    """
    Continue a conversation about a researcher using Gemini.

    Maintains conversation history for context.
    """
    from api.main import get_ml_models
    ml_models = get_ml_models()

    gemini = ml_models.get("gemini")
    if not gemini:
        raise HTTPException(
            status_code=503,
            detail="Gemini client not available. Please check API key configuration."
        )

    try:
        # Convert history to list of dicts
        history = [{"role": msg.role, "content": msg.content} for msg in request.history]

        response = gemini.chat(
            message=request.message,
            researcher=request.researcher,
            history=history,
            original_query=request.original_query
        )

        researcher_name = request.researcher.get("name", "Unknown")
        logger.info(f"Chat about '{researcher_name}': '{request.message[:50]}...'")

        return ChatResponse(
            response=response,
            researcher_name=researcher_name
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@router.post("/explain", response_model=ExplainResponse)
async def explain_match(request: ExplainRequest):
    """
    Explain why a researcher matches a search query using Gemini.

    This is typically used when a user clicks on a search result
    to understand why it was returned.
    """
    from api.main import get_ml_models
    ml_models = get_ml_models()

    gemini = ml_models.get("gemini")
    if not gemini:
        raise HTTPException(
            status_code=503,
            detail="Gemini client not available. Please check API key configuration."
        )

    try:
        explanation = gemini.explain_match(
            query=request.query,
            researcher=request.researcher,
            user_question=request.question
        )

        researcher_name = request.researcher.get("name", "Unknown")
        logger.info(f"Explain match for '{researcher_name}' on query '{request.query}'")

        return ExplainResponse(
            explanation=explanation,
            researcher_name=researcher_name,
            query=request.query
        )

    except Exception as e:
        logger.error(f"Explain error: {e}")
        raise HTTPException(status_code=500, detail=f"Explain failed: {str(e)}")
