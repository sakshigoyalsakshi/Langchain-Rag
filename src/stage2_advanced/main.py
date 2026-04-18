from query_rewriter import query_rewriter
from reranker import rerank

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage1_naive"))
from  retriever import load_retriever

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

def ask(question):
    rewritten = query_rewriter(question)
    retriever = load_retriever(k=10)
    docs = retriever.invoke(rewritten)
    docs = rerank(question, docs)

    prompt = ChatPromptTemplate.from_template("""
        You are an expert on LangChain. Answer using ONLY the context below.
        If the answer isn't in the context, say "I don't have enough context to answer this."

        Context:
        {context}

        Question: {question}
        """)
    context = "\n\n".join(doc.page_content for doc in docs)

    chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | StrOutputParser()

    return chain.invoke({"context": context, "question": question})


if __name__ == "__main__":
     question = input("Ask your question")
     print(ask(question))