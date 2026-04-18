from sentence_transformers import CrossEncoder


def rerank(question, docs, top_n=3):
    encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    input_pair = [[question, doc.page_content] for doc in docs]
    scores = encoder.predict(input_pair)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]