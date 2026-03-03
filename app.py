import os
import asyncio
import shutil
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv()

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.rag_pipeline import RAGPipeline

# Dedicated thread pool — keeps indexing OFF the main event loop thread
executor = ThreadPoolExecutor(max_workers=4)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

rag = RAGPipeline()
app.mount("/static", StaticFiles(directory="static"), name="static")

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported.")

    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    print(f"📄 Saved: {file.filename} — indexing...")

    # Use dedicated executor — completely separate from default event loop executor
    loop = asyncio.get_event_loop()
    num_chunks = await loop.run_in_executor(
        executor, rag.index_pdf, str(file_path)
    )

    print(f"✅ Done: {num_chunks} chunks")
    return {
        "message": f"'{file.filename}' indexed successfully!",
        "filename": file.filename,
        "chunks_indexed": num_chunks
    }

@app.post("/ask")
async def ask_question(request: QueryRequest):
    print(f"❓ /ask received: {request.question}")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, rag.answer_sync, request.question)

    print(f"✅ Answer ready")
    return result

@app.get("/documents")
async def list_documents():
    return {"documents": rag.list_indexed_documents()}

@app.delete("/documents")
async def clear_documents():
    rag.clear_index()
    for f in UPLOAD_DIR.glob("*.pdf"):
        f.unlink()
    return {"message": "Cleared."}

@app.get("/health")
async def health():
    return {"status": "ok"}