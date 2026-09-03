# Intelligent Research Paper RAG Evaluation

A portfolio-ready Retrieval-Augmented Generation (RAG) evaluation project based on the Intelligent Research Paper System. It demonstrates document ingestion, semantic retrieval with ChromaDB, grounded Gemini generation, automated DeepEval testing, and GitHub Actions CI.

## Architecture

```text
Research Papers (TXT/PDF)
        |
        v
   Text Extraction
        |
        v
 Chunking + Overlap
        |
        v
Sentence Transformers
        |
        v
     ChromaDB
        |
        v
     Top-k Retrieval
        |
        v
 Gemini Grounded Answer
        |
        +--------------------+
        |                    |
        v                    v
  User Answer          DeepEval Metrics
                            |
          +-----------------+------------------+
          |                 |                  |
     Generator          Retriever         Regression
     Relevancy          Relevancy           Dataset
     Faithfulness       Precision
                        Recall
``` 

## Why DeepEval?

DeepEval provides native RAG metrics for both sides of the pipeline: **Answer Relevancy** and **Faithfulness** for generation, plus **Contextual Relevancy**, **Contextual Precision**, and **Contextual Recall** for retrieval. This gives component-level diagnosis instead of relying on one overall score.

## Project structure

```text
src/rag_pipeline.py       # ingestion, chunking, embeddings, retrieval, Gemini generation
data/healthcare_ai.txt    # small reproducible evaluation corpus
evals/dataset.json        # golden evaluation questions and expected outputs
evals/run_deepeval.py     # end-to-end DeepEval runner
tests/test_rag_pipeline.py# deterministic unit tests
.github/workflows/ci.yml   # pytest + optional DeepEval CI
```

## Setup

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Create `.env` from `.env.example` and set `GEMINI_API_KEY`.

## Run the RAG application

```bash
python -m src.rag_pipeline
```

The command indexes documents in `data/`, asks a question, retrieves the most relevant chunks, and generates a grounded answer.

## Run deterministic tests

```bash
pytest -q
```

## Run DeepEval

```bash
python -m evals.run_deepeval
```

The default quality gate is **0.70**. Override it with `DEEPEVAL_THRESHOLD` when needed.

## CI/CD

GitHub Actions runs unit tests on every push and pull request. The DeepEval job runs when the repository has a `GEMINI_API_KEY` Actions secret configured. This keeps normal CI deterministic while allowing authenticated LLM quality regression tests.

Add the secret under **Repository Settings → Secrets and variables → Actions**.

## Resume-ready description

> Built an end-to-end RAG evaluation system for research-paper question answering using ChromaDB, Sentence Transformers, Gemini, and DeepEval. Implemented semantic retrieval, grounded generation, golden test datasets, and automated evaluation across answer relevancy, faithfulness, contextual relevancy, precision, and recall, with GitHub Actions CI quality gates.
