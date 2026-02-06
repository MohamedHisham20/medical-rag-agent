from pydantic import BaseModel, Field
from typing import List, Optional

# --- Request Models (What the frontend sends us) ---

class QueryRequest(BaseModel):
    question: str = Field(..., description="The medical question to ask", min_length=3)
    session_id: str = Field(..., description="Unique ID for the chat session")

# --- Response Models (What we send back) ---

class Source(BaseModel):
    """Where did the AI find this info? Crucial for medical credibility."""
    file_name: str
    page_number: int
    snippet: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source] = []
    processing_time: float

class HealthCheck(BaseModel):
    status: str = "ok"