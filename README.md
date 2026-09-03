# Intelligent Research Paper RAG Evaluation

A production-style RAG evaluation project for research-paper question answering using Gemini, ChromaDB, and DeepEval.

## Architecture

Documents → Chunking → Embeddings → ChromaDB → Retriever → Gemini → Answer → DeepEval

## Evaluation

- Answer Relevancy
- Faithfulness
- Contextual Relevancy
- Contextual Precision
- Contextual Recall

The project separates deterministic unit tests from LLM quality evaluation and runs both in GitHub Actions.

## Setup

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env  # Windows
```

Set `GEMINI_API_KEY` in `.env` or the environment.

## Run RAG

```bash
python -m src.rag_pipeline
```

## Run tests

```bash
pytest
```

## Run DeepEval

```bash
python -m evals.run_deepeval
```

CI runs unit tests on every push/PR. The LLM evaluation job runs only when `GEMINI_API_KEY` is configured as a GitHub Actions secret, avoiding accidental unauthenticated calls.
