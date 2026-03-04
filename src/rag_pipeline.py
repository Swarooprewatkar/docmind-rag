import os
import asyncio
import shutil
import gc
import time
from typing import List, Dict, Any
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

CHROMA_DIR = "/tmp/chroma_db"

SYSTEM_PROMPT = """You are an expert document analyst.
Answer questions based ONLY on the provided context below.
If the answer is not found in the context, say exactly:
"I couldn't find this information in the uploaded documents."

Be precise, structured, and always reference page numbers when available.

Context:
{context}
"""

class RAGPipeline:
    def __init__(self):
        print("🔧 Initializing RAG Pipeline...")

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.environ.get("GOOGLE_API_KEY")
        )

        groq_key = os.environ.get("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("❌ GROQ_API_KEY not found. Check your .env file.")

        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            api_key=groq_key,
        )

        self.vectorstore = None
        self.indexed_docs: List[str] = []
        self._load_existing_index()
        print("✅ RAG Pipeline ready.")

    def _load_existing_index(self):
        """Load ChromaDB index from disk if it exists."""
        if Path(CHROMA_DIR).exists():
            try:
                self.vectorstore = Chroma(
                    persist_directory=CHROMA_DIR,
                    embedding_function=self.embeddings
                )
                count = self.vectorstore._collection.count()
                print(f"✅ Loaded existing index ({count} vectors).")
            except Exception as e:
                print(f"⚠️ Could not load existing index: {e}")
                self.vectorstore = None

    def index_pdf(self, file_path: str) -> int:
        """
        Synchronous method — loads, chunks, embeds, and stores a PDF.
        Called from async context via run_in_executor.
        """
        print(f"📄 Loading PDF: {file_path}")
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        print(f"   → {len(pages)} pages loaded")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(pages)
        print(f"   → {len(chunks)} chunks created")

        filename = Path(file_path).name
        for chunk in chunks:
            chunk.metadata["source_file"] = filename

        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=CHROMA_DIR
            )
        else:
            self.vectorstore.add_documents(chunks)

        self.vectorstore.persist()

        if filename not in self.indexed_docs:
            self.indexed_docs.append(filename)

        return len(chunks)

    async def answer(self, question: str) -> Dict[str, Any]:
        """
        Async method — retrieves relevant chunks and generates an answer.
        Runs all blocking calls in thread pool via run_in_executor.
        """
        if self.vectorstore is None:
            return {
                "answer": "⚠️ No documents indexed yet. Please upload a PDF first.",
                "sources": [],
                "confidence": 0
            }

        loop = asyncio.get_event_loop()

        # ── Step 1: Retrieve relevant chunks ──
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10}
        )

        relevant_docs = await loop.run_in_executor(
            None, retriever.invoke, question
        )
        print(f"   → {len(relevant_docs)} relevant chunks retrieved")

        if not relevant_docs:
            return {
                "answer": "I couldn't find relevant information in the uploaded documents.",
                "sources": [],
                "confidence": 0
            }

        # ── Step 2: Format context ──
        context = "\n\n---\n\n".join([
            f"[Source: {doc.metadata.get('source_file', 'unknown')} | "
            f"Page {doc.metadata.get('page', 0) + 1}]\n{doc.page_content}"
            for doc in relevant_docs
        ])

        # ── Step 3: Build and run chain ──
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}")
        ])
        chain = prompt | self.llm | StrOutputParser()

        answer_text = await loop.run_in_executor(
            None,
            lambda: chain.invoke({"context": context, "question": question})
        )

        # ── Step 4: Extract sources ──
        sources = list({
            f"{doc.metadata.get('source_file', 'unknown')} — Page {doc.metadata.get('page', 0) + 1}"
            for doc in relevant_docs
        })

        confidence = min(len(relevant_docs) * 20, 95)

        return {
            "answer": answer_text,
            "sources": sources,
            "confidence": confidence
        }
    def answer_sync(self, question: str) -> dict:
        """Fully synchronous version — safe to run in executor."""

        print(f"DEBUG: answer_sync called: {question}")

        if self.vectorstore is None:
            return {
                "answer": "⚠️ No documents indexed yet. Please upload a PDF first.",
                "sources": [],
                "confidence": 0
            }

        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10}
        )

        relevant_docs = retriever.invoke(question)
        print(f"DEBUG: retrieved {len(relevant_docs)} docs")

        if not relevant_docs:
            return {
                "answer": "I couldn't find relevant information in the uploaded documents.",
                "sources": [],
                "confidence": 0
            }

        context = "\n\n---\n\n".join([
            f"[Source: {doc.metadata.get('source_file', 'unknown')} | "
            f"Page {doc.metadata.get('page', 0) + 1}]\n{doc.page_content}"
            for doc in relevant_docs
        ])
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        answer_text = chain.invoke({"context": context, "question": question})

        print(f"DEBUG: got answer: {answer_text[:80]}...")

        sources = list({
            f"{doc.metadata.get('source_file', 'unknown')} — Page {doc.metadata.get('page', 0) + 1}"
            for doc in relevant_docs
        })
        confidence = min(len(relevant_docs) * 20, 95)

        return {
            "answer": answer_text,
            "sources": sources,
            "confidence": confidence
        }
    def list_indexed_documents(self) -> List[str]:
        return self.indexed_docs

    def clear_index(self):
        """Wipe ChromaDB and reset state safely."""

        print("🗑️ Clearing index...")

        # Remove reference first
        self.vectorstore = None
        self.indexed_docs = []

        # Force cleanup
        gc.collect()
        time.sleep(1)  # Important for SQLite on Render

        # Remove directory safely
        chroma_path = Path(CHROMA_DIR)
        if chroma_path.exists():
            shutil.rmtree(chroma_path, ignore_errors=True)

        # Recreate directory
        os.makedirs(CHROMA_DIR, exist_ok=True)

        print("✅ Index cleared.")
