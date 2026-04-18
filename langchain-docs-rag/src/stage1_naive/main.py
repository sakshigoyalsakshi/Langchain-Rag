from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from retriever import load_retriever
from dotenv import load_dotenv

load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ask(question: str):
    retriever = load_retriever()
    prompt = ChatPromptTemplate.from_template("""
        You are an expert on LangChain. Answer using ONLY the context below.
        If the answer isn't in the context, say "I don't have enough context to answer this."

        Context:
        {context}

        Question: {question}
        """)
    chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    | StrOutputParser()
    )
    return chain.invoke(question)

if __name__ == "__main__":
    question = input("Ask your question")
    print(ask(question))