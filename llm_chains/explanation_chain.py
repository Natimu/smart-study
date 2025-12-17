from langchain_community.chat_models import ChatOpenAI
from .base_chain import BaseChain

class ExplanationChain(BaseChain):
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def run(self, query: str) -> str:
        context_chunks = self.retriever.retrieve(query)
        context_text = "\n\n".join(context_chunks)
        prompt = f"""
        You are a helpful study assistant. Use the following context to answer the question.
        Context:
        {context_text}

        Question: {query}
        Answer:
        """
        response = self.llm.call_as_llm(prompt)
        return response
