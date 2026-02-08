"""
Pydantic schemas for API request/response models
"""
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


# --- Search ---
class SearchRequest(BaseModel):
    query: str = Field(..., description="Research topic to search for")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    include_explanation: bool = Field(default=False, description="Include Gemini-generated match explanations")
    expand_query: bool = Field(default=True, description="Use Gemini to expand query with related terms")


class ResearcherResult(BaseModel):
    name: str
    department: Optional[str] = None
    research_interests: Optional[str] = None
    score: Optional[float] = None
    papers: Optional[List[Dict[str, Any]]] = None
    raw_text: Optional[str] = None
    # New: Gemini-generated explanation fields
    explanation: Optional[str] = None
    confidence: Optional[int] = None
    key_overlap: Optional[List[str]] = None
    match_type: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    results: List[ResearcherResult]
    search_time_ms: float


# --- Chat ---
class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message")
    researcher: Dict[str, Any] = Field(..., description="Current researcher context")
    history: List[ChatMessage] = Field(default=[], description="Previous conversation messages")
    original_query: str = Field(default="", description="Original search query")


class ChatResponse(BaseModel):
    response: str
    researcher_name: str


# --- Explain Match ---
class ExplainRequest(BaseModel):
    query: str = Field(..., description="Original search query")
    researcher: Dict[str, Any] = Field(..., description="Researcher to explain match for")
    question: Optional[str] = Field(default=None, description="Specific question about the match")


class ExplainResponse(BaseModel):
    explanation: str
    researcher_name: str
    query: str


# --- Health ---
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    researchers_count: int
