from langchain_openai import OpenAIEmbeddings
from .base_embedder import BaseEmbedder

class OpenAIEmbedded(BaseEmbedder):
    def __init__(self, model: str = "text-embedding-3-large"):
        self.embedder = OpenAIEmbeddings(model=model)
    def embed(self, texts: list[str])-> list[list[float]]:
        return self.embedder.embed_file(texts)