from abc import ABC, abstractmethod

class BaseEmbedded(ABC):
    @abstractmethod
    def embed(self, texts:list[str])->list[list[float]]:
        pass