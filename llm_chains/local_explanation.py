from langchain_community.chat_models import ChatOllama
from .base_chain import BaseChain

class LocalExplanation(BaseChain):
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = ChatOllama(model="mistral", temperature=0)

    def run(self, query: str) -> str:
        context_chunks = self.retriever.retrieve(query)
        context_text = "\n\n".join(context_chunks)
        prompt = f"""
        You are a helpful assistant. Use the following context to answer the question.
        Context:
        {context_text}

        Question: {query}
        Answer:
        """
        response = self.llm.invoke(prompt).content
        return response
