from langchain_openai import OpenAIEmbeddings

from .base_embedder import BaseEmbedder

class OpenAIEmbedder(BaseEmbedder):
   

    def __init__(self, model_name: str = "text-embedding-3-small"):
       
        self.embedder = OpenAIEmbeddings(model=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple text chunks."""
        return self.embedder.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query string."""
        return self.embedder.embed_query(text)
