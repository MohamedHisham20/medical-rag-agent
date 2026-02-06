from fastapi import FastAPI, UploadFile, File, HTTPException
from app.schemas import QueryRequest, QueryResponse, HealthCheck
import time

app = FastAPI(
    title="Medical RAG Agent API",
    version="1.0.0",
    description="A professional backend for medical document analysis."
)

# 1. Health Check (Always have one!)
@app.get("/health", response_model=HealthCheck)
async def health_check():
    return {"status": "healthy"}

# 2. File Upload Endpoint (Stub)
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # TODO: In the next step, we will pass this file to our RAG service
    return {"filename": file.filename, "message": "File received successfully"}

# 3. Chat Query Endpoint (Stub)
@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    start_time = time.time()
    
    # TODO: Connect this to the AI brain later
    fake_answer = f"I received your question: '{request.question}'. Logic coming soon."
    
    return {
        "answer": fake_answer,
        "sources": [],
        "processing_time": time.time() - start_time
    }

if __name__ == "__main__":
    import uvicorn
    # Hot-reload enabled for development
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)