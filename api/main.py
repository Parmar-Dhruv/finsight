from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="FinSight Stub API")

class QueryRequest(BaseModel):
    query: str

class RetrieveRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

@app.get("/health")
async def health_check():
    """Stub health check endpoint."""
    return {"status": "ok", "message": "Service is running"}

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """Stub endpoint for generating an answer to a query."""
    return {
        "query": request.query,
        "answer": "This is a stub answer for your query.",
        "sources": []
    }

@app.post("/retrieve")
async def retrieve_endpoint(request: RetrieveRequest):
    """Stub endpoint for retrieving context chunks for a query."""
    return {
        "query": request.query,
        "results": [
            {"chunk_id": "chunk_1", "text": "Stub chunk content 1", "score": 0.95},
            {"chunk_id": "chunk_2", "text": "Stub chunk content 2", "score": 0.82}
        ]
    }
