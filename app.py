import os
import asyncio
import shutil
import json
import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime
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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from src.rag_pipeline import RAGPipeline

# Dedicated thread pool — keeps indexing OFF the main event loop thread
executor = ThreadPoolExecutor(max_workers=4)

app = FastAPI(title="DocMind RAG PDF Q&A", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(exist_ok=True)

rag = RAGPipeline()
app.mount("/static", StaticFiles(directory="static"), name="static")

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".txt",
    ".xlsx", ".xls", ".csv",
    ".png", ".jpg", ".jpeg", ".tiff", ".bmp"
}

# ── Request models ────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str

class ExportRequest(BaseModel):
    format: str = "markdown"

class FormulaRequest(BaseModel):
    description: str
    columns: Optional[str] = ""

class AnomalyRequest(BaseModel):
    filename: str
    threshold: Optional[str] = ""

class ReportRequest(BaseModel):
    filename: str
    report_type: Optional[str] = "summary"

class CompareRequest(BaseModel):
    file1: str
    file2: str

class TicketRequest(BaseModel):
    description: str
    priority: Optional[str] = "Medium"

class EmailRequest(BaseModel):
    issue: str
    vendor: Optional[str] = ""
    contract_file: Optional[str] = ""

class AlarmRequest(BaseModel):
    log_text: str

@app.get("/")
def root():
    return FileResponse("static/index.html")


# ── UPLOAD (all formats) ──────────────────────────────────────────────────────
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload any supported file — PDF, Excel, CSV, DOCX, PPTX, TXT, Image."""

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    raw_name  = file.filename
    safe_name = "".join(c if (c.isalnum() or c in "._-") else "_" for c in raw_name)
    safe_name = safe_name.strip("_") or f"document{ext}"
    if not safe_name.endswith(ext):
        safe_name += ext

    file_path = UPLOAD_DIR / safe_name
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    print(f"\nFile saved: {safe_name} (original: {raw_name})")
    print("Indexing started...")

    loop = asyncio.get_event_loop()
    try:
        doc_info = await loop.run_in_executor(None, rag.index_file, str(file_path))
    except Exception as e:
        print(f"Indexing error: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

    print(f"Indexed {doc_info['chunks']} chunks from '{safe_name}'")

    return {
        "message":             f"'{safe_name}' uploaded and indexed successfully!",
        "filename":            safe_name,
        "file_type":           ext,
        "chunks_indexed":      doc_info["chunks"],
        "summary":             doc_info.get("summary", ""),
        "topics":              doc_info.get("topics", []),
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
    for f in UPLOAD_DIR.iterdir():
        if f.is_file():
            f.unlink()
    return {"message": "Cleared."}

# ── EXPORT Q&A SESSION ────────────────────────────────────────────────────────
@app.get("/export/{fmt}")
async def export_session(fmt: str):
    """Export full conversation as markdown or plain text."""
    history = rag.conversation_history
    docs    = [d["name"] for d in rag.list_indexed_documents()]

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


# ── EXCEL FORMULA GENERATOR ───────────────────────────────────────────────────
@app.post("/excel/formula")
async def generate_formula(request: FormulaRequest):
    """Generate Excel formula from plain English description."""
    if not request.description.strip():
        raise HTTPException(status_code=400, detail="Description cannot be empty.")

    prompt_text = f"""You are an expert Excel formula generator.

User request: {request.description}
{f'Column info: {request.columns}' if request.columns else ''}

Generate the exact Excel formula for this request.

Respond ONLY in this exact JSON format (no extra text):
{{
  "formula": "=YOUR_FORMULA_HERE",
  "explanation": "How this formula works in simple terms",
  "example": "Practical example of how to use it",
  "alternatives": ["=ALT_FORMULA_1", "=ALT_FORMULA_2"]
}}"""

    loop = asyncio.get_event_loop()
    try:
        raw = await loop.run_in_executor(None, lambda: rag.llm.invoke([HumanMessage(content=prompt_text)]).content)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    except json.JSONDecodeError:
        return {"formula": raw, "explanation": "", "alternatives": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Formula generation failed: {str(e)}")


# ── EXCEL ANOMALY DETECTOR ────────────────────────────────────────────────────
@app.post("/excel/anomaly")
async def detect_anomalies(request: AnomalyRequest):
    """Detect anomalies in uploaded Excel/CSV file."""
    file_path = UPLOAD_DIR / request.filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File '{request.filename}' not found. Upload it first.")

    ext = file_path.suffix.lower()
    try:
        df = pd.read_excel(file_path) if ext in (".xlsx", ".xls") else pd.read_csv(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read file: {str(e)}")

    data_summary = f"""
File: {request.filename}
Rows: {len(df)}
Columns: {', '.join(df.columns.astype(str).tolist())}

Statistics:
{df.describe().to_string()}

First 10 rows:
{df.head(10).to_string(index=False)}
"""

    prompt_text = f"""You are a data analyst. Analyze this dataset and detect anomalies.

{data_summary}
{f'Special focus: {request.threshold}' if request.threshold else ''}

Identify: outliers, missing values, duplicates, threshold breaches, inconsistent patterns.

Respond ONLY in this exact JSON format:
{{
  "total_rows": {len(df)},
  "anomalies_found": 0,
  "summary": "Overall data quality summary",
  "issues": [
    {{
      "type": "outlier/missing/duplicate/threshold",
      "column": "column_name",
      "description": "what was found",
      "severity": "High/Medium/Low",
      "affected_rows": "row numbers or count"
    }}
  ],
  "recommendations": ["recommendation1", "recommendation2"]
}}"""

    loop = asyncio.get_event_loop()
    try:
        raw = await loop.run_in_executor(None, lambda: rag.llm.invoke([HumanMessage(content=prompt_text)]).content)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    except json.JSONDecodeError:
        return {"summary": raw, "issues": [], "recommendations": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")


# ── EXCEL REPORT GENERATOR ────────────────────────────────────────────────────
@app.post("/excel/report")
async def generate_report(request: ReportRequest):
    """Generate analysis report from uploaded Excel/CSV. Returns .xlsx file."""
    file_path = UPLOAD_DIR / request.filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File '{request.filename}' not found.")

    ext = file_path.suffix.lower()
    try:
        df = pd.read_excel(file_path) if ext in (".xlsx", ".xls") else pd.read_csv(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read file: {str(e)}")

    output_filename = f"report_{request.filename.rsplit('.', 1)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    output_path     = EXPORT_DIR / output_filename

    try:
        import xlsxwriter
        workbook = xlsxwriter.Workbook(str(output_path))

        # Formats
        header_fmt  = workbook.add_format({'bold': True, 'bg_color': '#1e3a5f', 'font_color': 'white', 'border': 1, 'align': 'center'})
        data_fmt    = workbook.add_format({'border': 1})
        title_fmt   = workbook.add_format({'bold': True, 'font_size': 14, 'font_color': '#1e3a5f'})
        number_fmt  = workbook.add_format({'num_format': '#,##0.00', 'border': 1})

        # Sheet 1 — Raw Data
        ws_data = workbook.add_worksheet("Raw Data")
        for col_num, col_name in enumerate(df.columns):
            ws_data.write(0, col_num, str(col_name), header_fmt)
            ws_data.set_column(col_num, col_num, 18)
        for row_num, row in enumerate(df.itertuples(index=False), 1):
            for col_num, val in enumerate(row):
                ws_data.write(row_num, col_num, val, data_fmt)

        # Sheet 2 — Summary
        ws_summary = workbook.add_worksheet("Summary")
        ws_summary.write(0, 0, f"Analysis Report — {request.filename}", title_fmt)
        ws_summary.write(1, 0, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        ws_summary.write(3, 0, "Dataset Overview", header_fmt)
        ws_summary.write(4, 0, "Total Rows");    ws_summary.write(4, 1, len(df))
        ws_summary.write(5, 0, "Total Columns"); ws_summary.write(5, 1, len(df.columns))
        ws_summary.write(6, 0, "Columns");       ws_summary.write(6, 1, ", ".join(df.columns.astype(str).tolist()))

        num_cols = df.select_dtypes(include='number').columns.tolist()
        if num_cols:
            ws_summary.write(8, 0, "Numeric Statistics", header_fmt)
            stats_df = df[num_cols].describe()
            for col_num, col in enumerate(stats_df.columns):
                ws_summary.write(9, col_num + 1, str(col), header_fmt)
            for row_num, (idx, row) in enumerate(stats_df.iterrows(), 10):
                ws_summary.write(row_num, 0, str(idx), header_fmt)
                for col_num, val in enumerate(row):
                    ws_summary.write(row_num, col_num + 1, round(float(val), 2), number_fmt)

        # Sheet 3 — Charts
        if num_cols:
            ws_chart = workbook.add_worksheet("Charts")
            ws_chart.write(0, 0, "Data Visualization", title_fmt)
            ws_chart.write(2, 0, "Column", header_fmt)
            ws_chart.write(2, 1, "Mean",   header_fmt)
            ws_chart.write(2, 2, "Max",    header_fmt)
            ws_chart.write(2, 3, "Min",    header_fmt)

            for i, col in enumerate(num_cols[:10]):
                ws_chart.write(3 + i, 0, str(col))
                ws_chart.write(3 + i, 1, round(float(df[col].mean()), 2))
                ws_chart.write(3 + i, 2, round(float(df[col].max()),  2))
                ws_chart.write(3 + i, 3, round(float(df[col].min()),  2))

            chart = workbook.add_chart({'type': 'column'})
            chart.add_series({
                'name':       'Mean Values',
                'categories': ['Charts', 3, 0, 3 + len(num_cols) - 1, 0],
                'values':     ['Charts', 3, 1, 3 + len(num_cols) - 1, 1],
                'fill':       {'color': '#1e3a5f'}
            })
            chart.set_title({'name': 'Column Mean Values'})
            chart.set_style(10)
            ws_chart.insert_chart('F2', chart, {'x_scale': 1.5, 'y_scale': 1.5})

        workbook.close()

        return Response(
            content=open(output_path, "rb").read(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{output_filename}"'}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


# ── DOCUMENT COMPARISON ───────────────────────────────────────────────────────
@app.post("/compare")
async def compare_documents(request: CompareRequest):
    """Compare two uploaded documents and highlight differences."""
    file1_path = UPLOAD_DIR / request.file1
    file2_path = UPLOAD_DIR / request.file2

    if not file1_path.exists():
        raise HTTPException(status_code=404, detail=f"File '{request.file1}' not found.")
    if not file2_path.exists():
        raise HTTPException(status_code=404, detail=f"File '{request.file2}' not found.")

    def read_text(path: Path) -> str:
        ext = path.suffix.lower()
        if ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            return "\n".join([p.extract_text() or "" for p in reader.pages])
        elif ext == ".docx":
            from docx import Document as DocxDoc
            doc = DocxDoc(str(path))
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        elif ext in (".xlsx", ".xls"):
            return pd.read_excel(path).to_string(index=False)
        elif ext == ".csv":
            return pd.read_csv(path).to_string(index=False)
        else:
            return path.read_text(encoding="utf-8", errors="ignore")

    loop = asyncio.get_event_loop()
    try:
        text1 = await loop.run_in_executor(None, read_text, file1_path)
        text2 = await loop.run_in_executor(None, read_text, file2_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read files: {str(e)}")

    prompt_text = f"""You are a document comparison expert.

Compare these two documents and identify all differences.

Document 1 ({request.file1}):
{text1[:3000]}

Document 2 ({request.file2}):
{text2[:3000]}

Respond ONLY in this exact JSON format:
{{
  "summary": "Overall comparison summary",
  "similarity_score": 85,
  "differences": [
    {{
      "type": "added/removed/modified",
      "section": "section or topic name",
      "file1_content": "what file1 says",
      "file2_content": "what file2 says",
      "significance": "High/Medium/Low"
    }}
  ],
  "common_topics": ["topic1", "topic2"],
  "recommendations": ["recommendation1"]
}}"""

    try:
        raw = await loop.run_in_executor(None, lambda: rag.llm.invoke([HumanMessage(content=prompt_text)]).content)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    except json.JSONDecodeError:
        return {"summary": raw, "differences": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")
    
# ── TICKET GENERATOR ──────────────────────────────────────────────────────────
@app.post("/ticket/generate")
async def generate_ticket(request: TicketRequest):
    """Generate formatted incident ticket from plain English description."""

    prompt_text = f"""You are an IT incident management expert.

Generate a professional incident ticket from this description:
"{request.description}"
Priority hint: {request.priority}

Respond ONLY in this exact JSON format:
{{
  "title": "Short incident title (max 10 words)",
  "description": "Detailed description of the issue",
  "priority": "P1/P2/P3/P4",
  "category": "Network/Hardware/Software/Power/Other",
  "affected_component": "What system/equipment is affected",
  "impact": "Business impact description",
  "suggested_team": "NOC/Field/Vendor/IT/Management",
  "sla_breach_risk": "High/Medium/Low",
  "immediate_actions": ["action1", "action2", "action3"],
  "escalation_path": "Who to escalate to and when"
}}"""

    loop = asyncio.get_event_loop()
    try:
        raw = await loop.run_in_executor(None, lambda: rag.llm.invoke([HumanMessage(content=prompt_text)]).content)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    except json.JSONDecodeError:
        return {"title": "Incident Ticket", "description": raw}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ticket generation failed: {str(e)}")


# ── VENDOR EMAIL DRAFTING ─────────────────────────────────────────────────────
@app.post("/email/draft")
async def draft_email(request: EmailRequest):
    """Draft professional vendor escalation email."""

    # If contract file provided, pull SLA info from RAG
    sla_context = ""
    if request.contract_file:
        try:
            result = await rag.answer(f"What are the SLA terms and penalty clauses in {request.contract_file}?")
            sla_context = f"\nSLA context from contract:\n{result['answer']}"
        except Exception:
            pass

    prompt_text = f"""You are a professional business communication expert.

Draft a formal vendor escalation email for this issue:
"{request.issue}"
{f'Vendor: {request.vendor}' if request.vendor else ''}
{sla_context}

Respond ONLY in this exact JSON format:
{{
  "subject": "Email subject line",
  "body": "Complete professional email body",
  "tone": "Formal/Urgent/Diplomatic",
  "key_points": ["point1", "point2", "point3"],
  "follow_up_date": "Suggested follow-up timeframe"
}}"""

    loop = asyncio.get_event_loop()
    try:
        raw = await loop.run_in_executor(None, lambda: rag.llm.invoke([HumanMessage(content=prompt_text)]).content)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    except json.JSONDecodeError:
        return {"subject": "Vendor Escalation", "body": raw}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Email drafting failed: {str(e)}")

# ── ALARM LOG ANALYZER ────────────────────────────────────────────────────────
@app.post("/alarm/analyze")
async def analyze_alarm(request: AlarmRequest):
    """Analyze SNMP/network alarm logs and provide root cause + recommendations."""
    if not request.log_text.strip():
        raise HTTPException(status_code=400, detail="Log text cannot be empty.")

    prompt_text = f"""You are a telecom network operations expert specializing in alarm analysis.

Analyze these network alarm logs:

{request.log_text[:4000]}

Provide:
1. Root cause identification
2. Priority classification
3. Affected components
4. Recommended immediate actions
5. Similar past incident patterns (if detectable)

Respond ONLY in this exact JSON format:
{{
  "summary": "Brief summary of what happened",
  "root_cause": "Most likely root cause",
  "priority": "P1/P2/P3/P4",
  "affected_components": ["component1", "component2"],
  "alarm_types": ["alarm_type1", "alarm_type2"],
  "immediate_actions": ["action1", "action2", "action3"],
  "long_term_fix": "Permanent resolution recommendation",
  "escalate_to": "Who should handle this",
  "estimated_impact": "How many users/sites affected",
  "similar_pattern": "Any recurring pattern detected"
}}"""

    loop = asyncio.get_event_loop()
    try:
        raw = await loop.run_in_executor(None, lambda: rag.llm.invoke([HumanMessage(content=prompt_text)]).content)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    except json.JSONDecodeError:
        return {"summary": raw, "immediate_actions": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alarm analysis failed: {str(e)}")


# ── BULK ZIP UPLOAD ───────────────────────────────────────────────────────────
@app.post("/upload/bulk")
async def upload_bulk(file: UploadFile = File(...)):
    """Upload a ZIP file containing multiple documents — indexes all of them."""
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP files supported for bulk upload.")

    zip_path = UPLOAD_DIR / file.filename
    with open(zip_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    import zipfile
    results  = []
    errors   = []
    loop     = asyncio.get_event_loop()

    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            ext = Path(name).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            try:
                extracted_path = UPLOAD_DIR / Path(name).name
                with zf.open(name) as src, open(extracted_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

                doc_info = await loop.run_in_executor(None, rag.index_file, str(extracted_path))
                results.append({
                    "filename": Path(name).name,
                    "chunks":   doc_info["chunks"],
                    "status":   "success"
                })
            except Exception as e:
                errors.append({"filename": name, "error": str(e)})

    zip_path.unlink()  # cleanup zip

    return {
        "total_files":      len(results) + len(errors),
        "indexed":          len(results),
        "failed":           len(errors),
        "results":          results,
        "errors":           errors
    }
    
# ── HEALTH ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":             "ok",
        "version":            "4.0.0",
        "model":              "llama-3.3-70b-versatile",
        "embeddings":         "gemini-embedding-001",
        "provider":           "Groq + Google",
        "documents_indexed":  len(rag.list_indexed_documents()),
        "conversation_turns": len(rag.conversation_history),
        "supported_formats":  list(SUPPORTED_EXTENSIONS),
        "features": [
            "multi_format_upload",
            "excel_formula_generator",
            "anomaly_detector",
            "excel_report_generator",
            "document_comparison",
            "ticket_generator",
            "vendor_email_drafting",
            "alarm_analyzer",
            "bulk_zip_upload",
            "conversation_memory",
            "session_export",
            "ocr_support"
        ]
    }
