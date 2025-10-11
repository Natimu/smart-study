from abc import ABC, abstractmethod

class BaseEmbedder(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for a list of documents."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Return embedding for a single query."""
        pass