"""Small, reproducible RAG pipeline for research-paper QA."""

from __future__ import annotations

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DB_DIR = Path(os.getenv("CHROMA_DIR", ".chroma"))
COLLECTION = "research_papers"


def load_documents(data_dir: Path = DATA_DIR) -> list[tuple[str, str]]:
    """Load text and PDF research documents."""
    documents = []
    for path in sorted(data_dir.glob("**/*")) if data_dir.exists() else []:
        if path.suffix.lower() == ".txt":
            documents.append((path.name, path.read_text(encoding="utf-8")))
        elif path.suffix.lower() == ".pdf":
            text = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
            if text.strip():
                documents.append((path.name, text))
    return documents


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 100) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


class ResearchRAG:
    def __init__(self):
        self.embedder = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
        self.client = chromadb.PersistentClient(path=str(DB_DIR))
        self.collection = self.client.get_or_create_collection(COLLECTION)

    def index(self, documents: list[tuple[str, str]]) -> int:
        ids, texts, metas = [], [], []
        for name, text in documents:
            for i, chunk in enumerate(chunk_text(text)):
                ids.append(f"{name}:{i}")
                texts.append(chunk)
                metas.append({"source": name, "chunk": i})
        if not texts:
            return 0
        embeddings = self.embedder.encode(texts, normalize_embeddings=True).tolist()
        self.collection.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=embeddings)
        return len(texts)

    def retrieve(self, query: str, k: int = 4) -> list[str]:
        embedding = self.embedder.encode([query], normalize_embeddings=True).tolist()
        result = self.collection.query(query_embeddings=embedding, n_results=k)
        return result.get("documents", [[]])[0]

    def answer(self, query: str, k: int = 4) -> tuple[str, list[str]]:
        context = self.retrieve(query, k=k)
        if not context:
            return "I could not find supporting information in the indexed papers.", []
        prompt = (
            "Answer the question using ONLY the provided research-paper context. "
            "If the context is insufficient, say so. Do not invent citations or facts.\n\n"
            f"Context:\n{'\n\n'.join(context)}\n\nQuestion: {query}"
        )
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=prompt,
        )
        return response.text.strip(), context


if __name__ == "__main__":
    rag = ResearchRAG()
    count = rag.index(load_documents())
    print(f"Indexed {count} chunks")
    question = input("Question: ").strip()
    answer, context = rag.answer(question)
    print("\nAnswer:\n", answer)
    print(f"\nRetrieved contexts: {len(context)}")
