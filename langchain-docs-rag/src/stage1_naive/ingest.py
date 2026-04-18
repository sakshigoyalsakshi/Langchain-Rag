from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

VECTORSTORE_DIR = Path(__file__).resolve().parents[2] / "vectorstore" / "stage1"
DOCS_DIR = Path(__file__).resolve().parents[2] / "data" / "fastapi_repo" / "docs" / "en" / "docs"

def ingest():
    loader = DirectoryLoader(
        str(DOCS_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} files")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    if not chunks:
        print("No text extracted — pages may be empty.")
        return

    print("Embedding and storing in ChromaDB...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
        collection_name="langchain_docs_stage1",
    )
    print(f"Done. {len(chunks)} chunks stored at {VECTORSTORE_DIR}")


if __name__ == "__main__":
    ingest()
