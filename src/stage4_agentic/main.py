from langchain.agents import create_agent as create_react_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage1_naive"))
from retriever import load_retriever

load_dotenv()

@tool
def search_docs(query: str) -> str:
    """Search the FastAPI documentation for relevant information."""
    retriever = load_retriever(k=4)
    docs = retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs)


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [search_docs]
agent = create_react_agent(llm, tools)

def ask(question: str) -> str:
    result = agent.invoke({"messages": [("human", question)]})
    return result["messages"][-1].content


if __name__ == "__main__":
    question = input("Ask your question: ")
    print(ask(question))