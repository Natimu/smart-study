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
                {
                    "id": 1,
                    "type": "mcq",
                    "question": "...",
                    "options": {
                        "A": "...",
                        "B": "...",
                        "C": "...",
                        "D": "..."
                    },
                    "grading": {
                        "correct_option": "B"
                    },
                    "explanation": "...",
                    "source_chunks": [{
                        "chunk_id": "string",
                        "document": "string",
                        "page": 1
                    }]
                }

                
                """
        elif quiz_type == "short_answer":
            return """
                {
                    "id": 3,
                    "type": "short_answer",
                    "question": "...",
                    "grading": {
                        "expected_points": [
                        "...",
                        "...",
                        "..."
                        ],
                        "keywords": [
                        "...",
                        "...",
                        "...",
                        "..."
                        ],
                        "max_score": 3
                    },
                    "sample_answer": "....",
                    "explanation": "...",
                    "source_chunks": [{
                        "chunk_id": "string",
                        "document": "string",
                        "page": 1
                    }]
                    }

                """
        elif quiz_type == "true_false":
          return """
                {
                    "id": 2,
                    "type": "true_false",
                    "question": "...",
                    "grading": {
                        "correct_answer": false
                    },
                    "explanation": "...",
                    "source_chunks": [{
                        "chunk_id": "string",
                        "document": "string",
                        "page": 1
                    }]
                }

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
            - The length of the "questions" array MUST be exactly {num_questions}.
            - Follow the format EXACTLY.
            - Do NOT mix formats.
            - quiz_id MUST be a unique string identifier (UUID or timestamp-based string).
            - Return STRICTLY valid JSON that matches the following schema.
            - Question "id" values must be sequential integers starting from 1
            - For true_false questions, grading.correct_answer MUST be a boolean (true or false), not a string.
            - Each object inside the "questions" array MUST follow the schema below.

                Do not include markdown, comments, or extra text.

                {{
                "quiz_id": "...",
                "difficulty": "{difficulty}",
                "quiz_type": "{quiz_type}",
                "questions": [
                    {format_rules}    
                ]
                }}

            Context:
            {context}

            Topic:
            {topic}

            Quiz:
            """
        return self.llm.invoke(prompt).content