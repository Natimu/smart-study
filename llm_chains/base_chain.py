from abc import ABC, abstractmethod

class BaseChain(ABC):
    @abstractmethod
    def run(self, query: str) -> str:
        pass
