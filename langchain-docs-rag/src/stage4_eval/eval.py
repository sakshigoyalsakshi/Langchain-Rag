import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "stage1_naive"))
sys.path.insert(0, str(ROOT / "src" / "stage2_advanced"))
sys.path.insert(0, str(ROOT / "src" / "stage3_modular"))

from dotenv import load_dotenv
load_dotenv()

EVAL_SET = [
    {
        "question": "How do I add authentication to a FastAPI app?",
        "ground_truth": "FastAPI supports OAuth2 with password flow and JWT tokens. You use OAuth2PasswordBearer as a dependency and verify tokens in a get_current_user dependency that is injected into protected routes.",
    },
    {
        "question": "How do I define path parameters in FastAPI?",
        "ground_truth": "Path parameters are declared in the path string with curly braces and as function arguments with type annotations, for example: @app.get('/items/{item_id}') with def read_item(item_id: int).",
    },
    {
        "question": "What is dependency injection in FastAPI?",
        "ground_truth": "Dependency injection in FastAPI uses the Depends function to declare dependencies as function parameters. FastAPI resolves and injects them automatically, enabling reusable logic like authentication and database sessions.",
    },
]

def ask_stage1(question):
    from retriever import load_retriever
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    retriever = load_retriever(k=4)
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    prompt = ChatPromptTemplate.from_template("Answer using ONLY the context.\n\nContext:\n{context}\n\nQuestion: {question}")
    answer = (prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | StrOutputParser()).invoke({"context": context, "question": question})
    return answer, [d.page_content for d in docs]

def ask_stage2(question):
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
    return answer, [d.page_content for d in docs]

def ask_stage3(question):
    from hybrid_retriever import hybrid_retrieve
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    docs = hybrid_retrieve(question)
    context = "\n\n".join(d.page_content for d in docs)
    prompt = ChatPromptTemplate.from_template("Answer using ONLY the context.\n\nContext:\n{context}\n\nQuestion: {question}")
    answer = (prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | StrOutputParser()).invoke({"context": context, "question": question})
    return answer, [d.page_content for d in docs]

def evaluate_pipeline(name, ask_fn):
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from datasets import Dataset

    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    print(f"\nEvaluating {name} ...")
    rows = []
    for item in EVAL_SET:
        answer, contexts = ask_fn(item["question"])
        rows.append({"question": item["question"], "answer": answer, "contexts": contexts, "ground_truth": item["ground_truth"]})

    result = evaluate(
        Dataset.from_list(rows),
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=embeddings,
    )
    df = result.to_pandas()
    return {
        "pipeline": name,
        "faithfulness": df["faithfulness"].mean(),
        "answer_relevancy": df["answer_relevancy"].mean(),
        "context_precision": df["context_precision"].mean(),
    }

def main():
    results = [
        evaluate_pipeline("Stage 1 — Naive RAG",    ask_stage1),
        evaluate_pipeline("Stage 2 — Advanced RAG", ask_stage2),
        evaluate_pipeline("Stage 3 — Modular RAG",  ask_stage3),
    ]
    print("\n" + "=" * 70)
    print(f"{'Pipeline':<25} {'Faithfulness':>13} {'Ans Relevancy':>14} {'Ctx Precision':>14}")
    print("=" * 70)
    for r in results:
        print(f"{r['pipeline']:<25}{r.get('faithfulness',0):>13.3f}{r.get('answer_relevancy',0):>14.3f}{r.get('context_precision',0):>14.3f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
