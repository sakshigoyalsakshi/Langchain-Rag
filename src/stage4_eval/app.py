import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "stage1_naive"))
sys.path.insert(0, str(ROOT / "src" / "stage2_advanced"))
sys.path.insert(0, str(ROOT / "src" / "stage3_modular"))
sys.path.insert(0, str(ROOT / "src" / "stage4_agentic"))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.title("RAG Pipeline Demo — FastAPI Documentation")
st.caption("Compare Naive, Advanced, Modular, and Agentic RAG on the same question.")

pipeline = st.selectbox(
    "Choose a pipeline",
    ["Stage 1 — Naive RAG", "Stage 2 — Advanced RAG", "Stage 3 — Modular RAG", "Stage 4b — Agentic RAG"],
)

question = st.text_input("Ask a question", placeholder="How do I add authentication to a FastAPI app?")

if st.button("Ask") and question:
    with st.spinner("Retrieving and generating..."):
        if pipeline == "Stage 1 — Naive RAG":
            from retriever import load_retriever
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            retriever = load_retriever(k=4)
            docs = retriever.invoke(question)
            context = "\n\n".join(d.page_content for d in docs)
            prompt = ChatPromptTemplate.from_template("Answer using ONLY the context.\n\nContext:\n{context}\n\nQuestion: {question}")
            answer = (prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | StrOutputParser()).invoke({"context": context, "question": question})

        elif pipeline == "Stage 2 — Advanced RAG":
            from query_rewriter import query_rewriter
            from reranker import rerank
            from retriever import load_retriever
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            docs = rerank(question, load_retriever(k=10).invoke(query_rewriter(question)))
            context = "\n\n".join(d.page_content for d in docs)
            prompt = ChatPromptTemplate.from_template("Answer using ONLY the context.\n\nContext:\n{context}\n\nQuestion: {question}")
            answer = (prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | StrOutputParser()).invoke({"context": context, "question": question})

        elif pipeline == "Stage 3 — Modular RAG":
            from hybrid_retriever import hybrid_retrieve
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            docs = hybrid_retrieve(question)
            context = "\n\n".join(d.page_content for d in docs)
            prompt = ChatPromptTemplate.from_template("Answer using ONLY the context.\n\nContext:\n{context}\n\nQuestion: {question}")
            answer = (prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | StrOutputParser()).invoke({"context": context, "question": question})

        else:
            from main import ask as agentic_ask
            answer = agentic_ask(question)
            docs = []

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Retrieved context"):
        if docs:
            for i, doc in enumerate(docs):
                st.markdown(f"**Chunk {i+1}**")
                st.write(doc.page_content)
                st.divider()
        else:
            st.write("Agentic RAG retrieves dynamically — no fixed context to show.")
