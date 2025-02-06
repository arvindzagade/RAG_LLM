# from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI

def query_rag(user_query, vector_store):
    retrieved_docs = vector_store.similarity_search(user_query, k=3)

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    llm = ChatOpenAI(api_key=OPENAI_API_KEY)

    response = llm.predict(f"Based on these papers, answer: {user_query}\n\nContext:\n{context}")
    return response, retrieved_docs
