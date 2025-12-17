from typing import Literal
from langchain_community.chat_models import ChatOllama
from .base_chain import BaseChain

QuizType = Literal["mcq", "short_answer", "true_false"]

class QuizChain(BaseChain):

    def __init__(self, retriever, model: str = "mistral", temperature: float = 0.2):
        self.retriever = retriever
        self.llm = ChatOllama(model=model, temperature=temperature)
    

    def run(
            self,
            topic: str,
            number_questions: int = 5,
            quiz_type: QuizType = "true_false",
            difficulty: str = "intermediate",
            top_k: int = 5,
    ) -> str:
        context_chuck = self.retriever.retrieve(topic, top_k=top_k)
        context = "\n\n".join(context_chuck)

        prompt = f"""
            You are a teaching assistant helping a student study.
            Generate a quiz using ONLY the context provided.

            Rules:
            - If the context does not contain enough information, say what is missing briefly.
            - Do NOT invent facts not present in the context.
            - Difficulty: {difficulty}
            - Quiz type: {quiz_type}
            - Number of questions: {number_questions}

            For mcq:
            - Provide 4 options (A-D)
            - Mark the correct answer
            - Provide a 1-2 sentence explanation referencing the context

            For short_answer:
            - Provide expected key points for grading
            - provide a sample answer

            For true_false:
            - Mark True/False and provide a short justification

            Context:
            {context}

            Topic:
            {topic}

            Output format:
            1) Question...
            A) ...
            B) ...
            C) ...
            D) ...
            Answer: X
            Explanation: ...

            Quiz:
            """
        return self.llm.invoke(prompt).content