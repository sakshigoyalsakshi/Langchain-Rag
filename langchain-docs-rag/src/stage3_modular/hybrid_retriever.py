import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage1_naive"))

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings

from fusion import fuse
from retriever import load_retriever

load_dotenv()

VECTORSTORE_DIR = Path(__file__).resolve().parents[2] / "vectorstore" / "stage1"
COLLECTION_NAME = "langchain_docs_stage1"


def hybrid_retrieve(question: str, k: int = 4) -> list:
    # Dense retrieval
    vector_retriever = load_retriever(k=10)
    vector_results = vector_retriever.invoke(question)

    # BM25 retrieval — needs raw text from the vectorstore
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    all_docs = vectorstore.get()
    bm25 = BM25Retriever.from_texts(all_docs["documents"], k=10)
    bm25_results = bm25.invoke(question)

    # Fuse and return top k
    return fuse(vector_results, bm25_results)[:k]
