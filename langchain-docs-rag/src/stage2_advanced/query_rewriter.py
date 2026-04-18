from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

def query_rewriter(question):
    prompt = ChatPromptTemplate.from_template("""
        Write a short paragraph that directly answers this question, as if you were a LangChain documentation page: {question}
        """)
    
    chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | StrOutputParser()
    return chain.invoke({"question": question})
