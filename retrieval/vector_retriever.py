from langchain_community.vectorstores import Chroma
from .base_retriever import BaseRetriever

class VectorRetriever(BaseRetriever):
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore

    def retrieve(self, query: str, top_k: int = 2) -> list[str]:
        results = self.vectorstore.similarity_search(query, k=top_k)
        return [r.page_content for r in results]