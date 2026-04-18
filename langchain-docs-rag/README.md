# FastAPI Docs RAG

A hands-on implementation of four RAG (Retrieval-Augmented Generation) architectures, built progressively from Naive RAG to Agentic RAG. Each stage fixes the failure modes of the previous one. Corpus is the full FastAPI documentation. Evaluated with RAGAS on the same question set.

**Corpus:** FastAPI documentation (154 markdown files from the official GitHub repo)  
**LLM:** GPT-4o-mini  
**Embeddings:** OpenAI text-embedding-3-small  
**Vector store:** ChromaDB (local)

---

## Architecture

```
Stage 1 — Naive RAG
  question → embed → similarity search → top-k chunks → LLM → answer

Stage 2 — Advanced RAG
  question → HyDE rewrite → embed → similarity search → cross-encoder rerank → LLM → answer

Stage 3 — Modular RAG
  question → [dense search + BM25 search] → RRF fusion → LLM → answer

Stage 4 — Eval + UI
  RAGAS scoring across all stages + Streamlit demo
```

---

## Results

Evaluated on 3 FastAPI questions using RAGAS metrics (faithfulness, answer relevancy, context precision).

| Pipeline           | Faithfulness | Answer Relevancy | Context Precision |
|--------------------|:------------:|:----------------:|:-----------------:|
| Stage 1 — Naive    |    0.944     |      0.990       |       0.972       |
| Stage 2 — Advanced |    0.917     |      0.990       |       0.944       |
| Stage 3 — Modular  |    0.944     |      0.983       |       0.694       |

**Key finding:** Context Precision degrades across stages on this corpus. Stage 1 achieves the highest precision — pure vector search returns highly relevant chunks for a technically focused corpus like FastAPI docs. Stage 2's reranker slightly reduces precision but maintains near-identical answer quality. Stage 3 scores significantly lower on Context Precision because BM25 promotes keyword-matched but semantically weak chunks — FastAPI's uniform vocabulary (every page mentions "app", "router", "request") makes keyword search less discriminative, and RRF fusion then surfaces those low-quality chunks into the final result set.

This demonstrates an important RAG principle: more complex pipelines are not always better. Hybrid search excels when the corpus has varied vocabulary and queries contain rare technical terms. On uniform corpora, pure vector search can outperform it.

---

## Project Structure

```
fastapi-docs-rag/
├── src/
│   ├── stage1_naive/
│   │   ├── ingest.py          # scrape → chunk → embed → ChromaDB
│   │   ├── retriever.py       # load vectorstore, return retriever
│   │   └── main.py            # naive RAG chain
│   ├── stage2_advanced/
│   │   ├── query_rewriter.py  # HyDE query rewriting
│   │   ├── reranker.py        # cross-encoder reranking
│   │   └── main.py            # advanced RAG chain
│   ├── stage3_modular/
│   │   ├── hybrid_retriever.py # BM25 + dense search
│   │   ├── fusion.py           # Reciprocal Rank Fusion
│   │   └── main.py             # modular RAG chain
│   └── stage4_eval/
│       ├── eval.py             # RAGAS evaluation across all stages
│       └── app.py              # Streamlit UI
├── vectorstore/                # ChromaDB persisted on disk
├── notebooks/
│   └── exploration.ipynb
├── requirements.txt
└── .env                        # OPENAI_API_KEY (not committed)
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/fastapi-docs-rag
cd fastapi-docs-rag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:
```
OPENAI_API_KEY=sk-...
```

---

## Running Each Stage

**Ingest (run once):**
```bash
python src/stage1_naive/ingest.py
```

**Stage 1 — Naive RAG:**
```bash
python src/stage1_naive/main.py
```

**Stage 2 — Advanced RAG:**
```bash
python src/stage2_advanced/main.py
```

**Stage 3 — Modular RAG:**
```bash
python src/stage3_modular/main.py
```

**Evaluate all pipelines:**
```bash
python src/stage4_eval/eval.py
```

**Streamlit UI:**
```bash
streamlit run src/stage4_eval/app.py
```

---

## Key Concepts

**HyDE (Hypothetical Document Embeddings)** — Instead of embedding the raw question, the LLM first generates a hypothetical answer. That answer embeds closer to real document chunks than a question does, improving retrieval recall.

**Cross-encoder reranking** — After retrieving top-k candidates by cosine similarity, a cross-encoder (BERT fine-tuned on MS MARCO) scores each (question, chunk) pair jointly — far more precise than embedding similarity alone.

**BM25 + RRF** — Dense vector search understands meaning but misses exact keyword matches. BM25 keyword search does the opposite. Reciprocal Rank Fusion merges both ranked lists: a chunk appearing in both at decent ranks beats one that topped only a single list.

---

## Stack

| Component | Library |
|---|---|
| Orchestration | LangChain |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector store | ChromaDB |
| Reranker | sentence-transformers (cross-encoder/ms-marco-MiniLM-L-6-v2) |
| Keyword search | rank-bm25 |
| Evaluation | RAGAS |
| UI | Streamlit |
