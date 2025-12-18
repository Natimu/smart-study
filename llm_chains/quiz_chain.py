from typing import Literal
from langchain_community.chat_models import ChatOllama
from .base_chain import BaseChain

QuizType = Literal["mcq", "short_answer", "true_false"]

class QuizChain(BaseChain):

    def __init__(self, retriever, model: str = "mistral", temperature: float = 0.1):
        self.retriever = retriever
        self.llm = ChatOllama(model=model, temperature=temperature)

    def _format_instructions(self, quiz_type: str) -> str:
        if quiz_type == "mcq":
            return """
                Output STRICTLY in the following format.
                Do NOT include any other question types.

                Each question MUST be multiple-choice.

                Format:
                1) Question text
                A) Option A
                B) Option B
                C) Option C
                D) Option D
                Correct Answer: <A/B/C/D>
                Explanation: 1–2 sentences referencing the context
                """
        elif quiz_type == "short_answer":
            return """
                Output STRICTLY in the following format.

                Format:
                1) Question text
                Expected Points:
                - bullet point
                - bullet point
                Sample Answer: concise paragraph
                """
        elif quiz_type == "true_false":
            return """
                Output STRICTLY in the following format.

                Format:
                1) Statement
                Answer: True/False
                Justification: 1–2 sentences
                """
        else:
            raise ValueError(f"Unsupported quiz type: {quiz_type}")


    def run(
            self,
            topic: str,
            num_questions: int = 5,
            quiz_type: QuizType = "true_false",
            difficulty: str = "intermediate",
            top_k: int = 5,
    ) -> str:
        context_chucks = self.retriever.retrieve(topic, top_k=top_k)
        context = "\n\n".join(context_chucks)

        format_rules = self._format_instructions(quiz_type)

        prompt = f"""
            You are a teaching assistant helping a student study.

            Rules:
            - Use ONLY the context provided.
            - Do NOT invent facts.
            - Difficulty: {difficulty}
            - Generate EXACTLY {num_questions} questions.
            - Follow the format EXACTLY.
            - Do NOT mix formats.

            {format_rules}

            Context:
            {context}

            Topic:
            {topic}

            Quiz:
            """
        return self.llm.invoke(prompt).content