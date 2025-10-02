from abc import ABC, abstractmethod

class BaseRetrievr(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_match: int=2)-> list[str]:
        pass