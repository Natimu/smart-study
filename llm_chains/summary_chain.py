from langchain_community.chat_models import ChatOllama
from .base_chain import BaseChain

class SummaryChain(BaseChain):
    "build a structured cheat sheet"

    def __init__(self, retriever, model: str = "mistral", temperature: float = 0.1):
        self.retriever = retriever
        self.llm = ChatOllama(model=model, temperature=temperature)
        
    def run(
            self,
            topic: str,
            style: str ="cheat_sheet",
            top_k: int = 8,
    ) -> str:
        
        context_chunks = self.retriever.retrieve(topic, top_k=top_k)
        context = "\n\n".join(context_chunks)
        prompt = f"""
                    You are a study assistant. Create a high-quality summary using ONLY the context below.
                    Do not invent facts.

                    Topic: {topic}
                    Style: {style}

                    If style is "cheat_sheet", output:
                    - Key definitions
                    - Core concepts
                    - Important lists/steps
                    - Common pitfalls/mistakes
                    - Quick recall section (3-6 bullets)
                    - If formulas exist, include them

                    Context:
                    {context}

                    Cheat Sheet:
                    """
        return self.llm.invoke(prompt).content