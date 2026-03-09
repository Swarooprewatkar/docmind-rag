import os
import asyncio
import shutil
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv()

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from src.rag_pipeline import RAGPipeline

# Dedicated thread pool — keeps indexing OFF the main event loop thread
executor = ThreadPoolExecutor(max_workers=4)

app = FastAPI(title="DocMind RAG PDF Q&A", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

rag = RAGPipeline()
app.mount("/static", StaticFiles(directory="static"), name="static")

class QueryRequest(BaseModel):
    question: str

class ExportRequest(BaseModel):
    format: str = "markdown"   # "markdown" or "text"

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
    try:
        doc_info = await loop.run_in_executor(None, rag.index_pdf, str(file_path))
    except Exception as e:
        print(f"❌ Indexing error: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

    print(f"✅ Indexed {doc_info['chunks']} chunks from '{file.filename}'")
    return {
        "message": f"'{file.filename}' uploaded and indexed successfully!",
        "filename": file.filename,
        "chunks_indexed": doc_info["chunks"],
        # Improvement 4: return summary info
        "summary": doc_info.get("summary", ""),
        "topics": doc_info.get("topics", []),
        "suggested_questions": doc_info.get("suggested_questions", []),
    }

@app.post("/ask")
async def ask_question(request: QueryRequest):
    print(f"❓ /ask received: {request.question}")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    print(f"\n❓ Question: {request.question}")
    result = await rag.answer(request.question)
    print(f"✅ Answer generated ({result['confidence']}% confidence) | Turn {result['conversation_turn']}")
    return result

# ── Improvement 1: Clear conversation (keep docs) ─────────────────────────────
@app.post("/conversation/clear")
async def clear_conversation():
    rag.clear_conversation()
    return {"message": "Conversation history cleared."}

@app.get("/documents")
async def list_documents():
    return {"documents": rag.list_indexed_documents()}

@app.delete("/documents")
async def clear_documents():
    rag.clear_index()
    for f in UPLOAD_DIR.glob("*.pdf"):
        f.unlink()
    return {"message": "Cleared."}

# ── Improvement 6: Export Q&A session ─────────────────────────────────────────
@app.get("/export/{fmt}")
async def export_session(fmt: str):
    """Export full conversation as markdown or plain text."""
    history = rag.conversation_history
    docs = [d["name"] for d in rag.list_indexed_documents()]

    if fmt == "markdown":
        lines = ["# DocMind — Q&A Session Export\n"]
        lines.append(f"**Documents:** {', '.join(docs) if docs else 'None'}\n")
        lines.append(f"**Total questions:** {len(history)}\n\n---\n")
        for i, turn in enumerate(history, 1):
            lines.append(f"## Q{i}: {turn['question']}\n")
            lines.append(f"{turn['answer']}\n\n---\n")
        content = "\n".join(lines)
        return Response(
            content=content,
            media_type="text/markdown",
            headers={"Content-Disposition": "attachment; filename=docmind-session.md"}
        )
    else:
        lines = ["DocMind Q&A Session Export", "=" * 40, ""]
        lines.append(f"Documents: {', '.join(docs) if docs else 'None'}")
        lines.append(f"Total questions: {len(history)}\n")
        for i, turn in enumerate(history, 1):
            lines.append(f"Q{i}: {turn['question']}")
            lines.append(f"A: {turn['answer']}\n")
        content = "\n".join(lines)
        return Response(
            content=content,
            media_type="text/plain",
            headers={"Content-Disposition": "attachment; filename=docmind-session.txt"}
        )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "llama-3.3-70b-versatile",
        "provider": "Groq",
        "documents_indexed": len(rag.list_indexed_documents()),
        "conversation_turns": len(rag.conversation_history),
    }