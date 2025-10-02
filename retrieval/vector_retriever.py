from langchain.vectorstores import Chroma
from .base_retriever import BaseRetrievr

class VectorRetriever(BaseRetrievr):
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore

    def retrieve(self, query: str, top_match: int = 3) -> list[str]:
        results = self.vectorstore.similarity_search(query, k=top_match)
        return [r.page_content for r in results]