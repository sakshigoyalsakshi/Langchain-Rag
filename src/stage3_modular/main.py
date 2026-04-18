import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from hybrid_retriever import hybrid_retrieve

load_dotenv()

PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer using ONLY the context below.
If the answer isn't in the context, say "I don't have enough context to answer this."

Context:
{context}

Question: {question}
""")


def ask(question: str) -> str:
    docs = hybrid_retrieve(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    chain = PROMPT | ChatOpenAI(model="gpt-4o-mini", temperature=0) | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


if __name__ == "__main__":
    question = input("Ask your question: ")
    print(ask(question))
