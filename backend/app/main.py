import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from app.schemas import QueryRequest, QueryResponse, Source, HealthCheck
from app.services.rag_service import rag_service  # <--- Import the service
import time
from typing import Dict

app = FastAPI(
    title="Medical RAG Agent API",
    version="1.0.0"
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Track ingestion status
ingestion_status: Dict[str, dict] = {}

@app.get("/health", response_model=HealthCheck)
async def health_check():
    return {"status": "healthy"}

def process_document_background(file_path: str, filename: str):
    """Background task to process PDF ingestion"""
    try:
        ingestion_status[filename] = {"status": "processing", "progress": 0, "total": 0}
        num_chunks = rag_service.ingest_file(file_path, filename, ingestion_status)
        ingestion_status[filename] = {
            "status": "completed",
            "progress": num_chunks,
            "total": num_chunks,
            "message": "File successfully ingested"
        }
    except Exception as e:
        ingestion_status[filename] = {
            "status": "failed",
            "error": str(e)
        }
        # Clean up file if failed
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # 1. Save file locally
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 2. Start background processing
    ingestion_status[file.filename] = {"status": "queued", "progress": 0}
    background_tasks.add_task(process_document_background, file_path, file.filename)
    
    return {
        "filename": file.filename,
        "message": "File uploaded successfully. Processing in background.",
        "status": "queued"
    }

@app.get("/upload/status/{filename}")
async def get_upload_status(filename: str):
    """Check the status of a file ingestion"""
    if filename not in ingestion_status:
        raise HTTPException(status_code=404, detail="File not found or not being processed")
    return ingestion_status[filename]

@app.get("/upload/status")
async def get_all_upload_statuses():
    """Get the status of all file ingestions"""
    return ingestion_status

@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    start_time = time.time()
    
    try:
        # 1. Ask the Brain
        answer, source_docs = rag_service.query(request.question)
        
        # 2. Format Sources for the frontend
        formatted_sources = []
        for doc in source_docs:
            formatted_sources.append(Source(
                file_name=os.path.basename(doc.metadata.get("source", "unknown")),
                page_number=doc.metadata.get("page", 0),
                snippet=doc.page_content[:200] + "..." # Preview only
            ))
            
        return {
            "answer": answer,
            "sources": formatted_sources,
            "processing_time": time.time() - start_time
        }
        
    except ValueError as e:
        # Handles case where DB is empty
        raise HTTPException(status_code=400, detail=str(e))