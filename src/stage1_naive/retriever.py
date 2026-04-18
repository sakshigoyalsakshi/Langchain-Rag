from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

VECTORSTORE_DIR = Path(__file__).resolve().parents[2] / "vectorstore" / "stage1"


def load_retriever(k=4):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vec_store = Chroma(persist_directory=str(VECTORSTORE_DIR), embedding_function=embeddings,
           collection_name="langchain_docs_stage1")
    return vec_store.as_retriever(search_kwargs={"k": k})