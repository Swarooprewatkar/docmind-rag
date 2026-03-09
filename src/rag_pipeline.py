import os
import asyncio
import gc
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

# ── Improvement 3: Better chunking separators ──────────────────────────────────
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150
SEPARATORS    = ["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]

SYSTEM_PROMPT = """You are an expert document analyst.
Answer questions based ONLY on the provided context below.
If the answer is not found in the context, say exactly:
"I couldn't find this information in the uploaded documents."

Be precise, structured, and always reference page numbers when available.

Context:
{context}
"""

SUMMARY_PROMPT = """You are a document analyst. Given the beginning of a document, produce:
1. A 2-3 sentence summary of what this document is about
2. 3-5 key topics covered (as a comma-separated list)
3. 3 suggested questions a user might ask about this document

Respond in this exact JSON format:
{{
  "summary": "...",
  "topics": ["topic1", "topic2", "topic3"],
  "suggested_questions": ["question1", "question2", "question3"]
}}

Document content:
{content}
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
        
        # ── Improvement 1: Conversation memory per session ────────────────────
        self.conversation_history: List[Dict[str, str]] = []
        
        self._load_existing_index()
        print("✅ RAG Pipeline ready.")

    def _load_existing_index(self):
        """Load ChromaDB index if exists."""

        try:
            self.vectorstore = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=self.embeddings
            )

            count = self.vectorstore._collection.count()

            if count == 0:
                self.vectorstore = None
            else:
                print(f"✅ Loaded existing index ({count} vectors).")

        except Exception as e:
            print("⚠️ Could not load index:", e)
            self.vectorstore = None

    def index_pdf(self, file_path: str) -> int:
        """
        Synchronous — loads, chunks (smart separators), embeds, stores PDF.
        Also generates auto-summary using first 3000 chars.
        Returns dict with chunk count + summary info.
        """
        print(f"📄 Loading PDF: {file_path}")
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        print(f"   → {len(pages)} pages loaded")

        # Improvement 3: Smart recursive chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=SEPARATORS,
            length_function=len,
        )
        chunks = splitter.split_documents(pages)
        print(f"   → {len(chunks)} chunks created")

        filename = Path(file_path).name
        # Improvement 5: Store paragraph position metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["source_file"] = filename
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
            # paragraph position within page
            chunk.metadata["position"] = f"chunk {i+1}/{len(chunks)}"

        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=CHROMA_DIR
            )
        else:
            self.vectorstore.add_documents(chunks)

        # Improvement 4: Auto-generate summary from first 3000 chars
        first_content = " ".join([p.page_content for p in pages[:3]])[:3000]
        doc_info = self._generate_summary(filename, first_content, len(chunks))

        # Update or add doc info
        self.indexed_docs = [d for d in self.indexed_docs if d["name"] != filename]
        self.indexed_docs.append(doc_info)
        
        return doc_info

    def _generate_summary(self, filename: str, content: str, chunks: int) -> Dict:
        """Generate auto summary, topics and suggested questions."""
        import json
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("human", SUMMARY_PROMPT)
            ])
            chain = prompt | self.llm | StrOutputParser()
            raw = chain.invoke({"content": content})

            # Strip markdown fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw.strip())

            return {
                "name": filename,
                "chunks": chunks,
                "summary": parsed.get("summary", ""),
                "topics": parsed.get("topics", []),
                "suggested_questions": parsed.get("suggested_questions", []),
            }
        except Exception as e:
            print(f"⚠️ Summary generation failed: {e}")
            return {
                "name": filename,
                "chunks": chunks,
                "summary": "",
                "topics": [],
                "suggested_questions": [],
            }

    async def answer1(self, question: str) -> Dict[str, Any]:
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
        
        
    # ── Improvement 1: Multi-turn conversation memory ─────────────────────────
    async def answer(self, question: str) -> Dict[str, Any]:
        if self.vectorstore is None:
            return {
                "answer": "⚠️ No documents indexed yet. Please upload a PDF first.",
                "sources": [], "confidence": 0, "conversation_turn": 0
            }

        loop = asyncio.get_event_loop()

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
                "sources": [], "confidence": 0, "conversation_turn": len(self.conversation_history)
            }

        # Improvement 5: Include paragraph/position in context
        context = "\n\n---\n\n".join([
            f"[Source: {doc.metadata.get('source_file','unknown')} | "
            f"Page {doc.metadata.get('page', 0) + 1} | "
            f"Position: {doc.metadata.get('position', 'unknown')}]\n{doc.page_content}"
            for doc in relevant_docs
        ])

        # Improvement 1: Build messages with conversation history
        messages = [("system", SYSTEM_PROMPT)]

        # Add last 4 turns of history (keep context window manageable)
        for turn in self.conversation_history[-4:]:
            messages.append(("human", turn["question"]))
            messages.append(("assistant", turn["answer"]))

        messages.append(("human", "{question}"))

        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | self.llm | StrOutputParser()

        answer_text = await loop.run_in_executor(
            None,
            lambda: chain.invoke({"context": context, "question": question})
        )

        # Save to conversation history
        self.conversation_history.append({
            "question": question,
            "answer": answer_text
        })

        # Improvement 5: Rich source info with page + position
        sources = list({
            f"{doc.metadata.get('source_file','unknown')} — Page {doc.metadata.get('page', 0) + 1}"
            for doc in relevant_docs
        })

        confidence = min(len(relevant_docs) * 20, 95)

        return {
            "answer": answer_text,
            "sources": sources,
            "confidence": confidence,
            "conversation_turn": len(self.conversation_history),
            "source_passages": [
                {
                    "file": doc.metadata.get("source_file", "unknown"),
                    "page": doc.metadata.get("page", 0) + 1,
                    "position": doc.metadata.get("position", ""),
                    "text": doc.page_content[:200] + "..."
                }
                for doc in relevant_docs[:3]
            ]
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
        
    def clear_conversation(self):
        """Clear conversation history only — keep documents indexed."""
        self.conversation_history = []

    def list_indexed_documents(self) -> List[str]:
        return self.indexed_docs

    def clear_index(self):
        """Clear vectorstore safely without deleting filesystem."""

        print("🗑️ Clearing index...")

        try:
            if self.vectorstore:
                self.vectorstore.delete_collection()
        except Exception as e:
            print("⚠️ Collection delete warning:", e)

        self.vectorstore = None
        self.indexed_docs = []
        self.conversation_history = []
        gc.collect()

        print("✅ Index cleared.")
