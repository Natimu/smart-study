from typing import Literal
import json
import re
from langchain_community.chat_models import ChatOllama
from .base_chain import BaseChain

QuizType = Literal["mcq", "short_answer", "true_false"]
COMMON_FIELDS = {"id", "type", "prompt", "grading", "explanation"}
REQUIRED_TOP_KEYS = {"quiz_id", "difficulty", "quiz_type", "questions"}

class StructureError(Exception):
    """JSON is malformed or structurally invalid"""
    pass

class ContentError(Exception):
    """JSON is valid but content violates quiz rules"""
    pass


class QuizChain(BaseChain):

    def __init__(self, retriever, model: str = "llama3.2:3b", temperature: float = 0.1):
        self.retriever = retriever
        self.llm = ChatOllama(model=model, temperature=temperature)

    def _question_schema(self, quiz_type: str) -> str:
        if quiz_type == "mcq":
            return """
                {
                    "id": 1,
                    "type": "mcq",
                    "prompt": "What is the main purpose of X?",
                    "options": {
                        "A": "First option",
                        "B": "Second option",
                        "C": "Third option",
                        "D": "Fourth option"
                    },
                    "grading": {
                        "correct_option": "B"
                    },
                    "explanation": "This is the explanation."
                }
                """
        elif quiz_type == "short_answer":
            return """
                {
                    "id": 1,
                    "type": "short_answer",
                    "prompt": "Explain the concept of X.",
                    "grading": {
                        "expected_points": [
                            "First key point",
                            "Second key point",
                            "Third key point"
                        ],
                        "keywords": [
                            "keyword1",
                            "keyword2",
                            "keyword3"
                        ],
                        "max_score": 3
                    },
                    "sample_answer": "A sample answer demonstrating the expected response.",
                    "explanation": "This is the explanation."
                }
                """
        elif quiz_type == "true_false":
            return """
                {
                    "id": 1,
                    "type": "true_false",
                    "prompt": "X is responsible for Y.",
                    "grading": {
                        "correct_answer": true
                    },
                    "explanation": "This is the explanation."
                }
                """
        else:
            raise ValueError(f"Unsupported quiz type: {quiz_type}")

    def _full_quiz_schema(self, quiz_type: str) -> str:
        return f"""
        {{
          "quiz_id": "string",
          "difficulty": "string",
          "quiz_type": "{quiz_type}",
          "questions": [
            {self._question_schema(quiz_type)}
          ]
        }}
        """

    def run(
            self,
            topic: str,
            num_questions: int = 5,
            quiz_type: QuizType = "true_false",
            difficulty: str = "intermediate",
            top_k: int = 3,
    ) -> dict:
        context_chunks = self.retriever.retrieve(topic, top_k=top_k)
        
        if not context_chunks:
            raise ValueError(f"No context found for topic: {topic}")
        
        context = "\n\n".join(context_chunks)
        
        extra_rules = ""
        if quiz_type == "mcq":
            extra_rules = """
    - This quiz is MULTIPLE-CHOICE ONLY.
    - Do NOT generate true/false or yes/no questions.
    - Each question MUST have four meaningful answer choices.
    - All four options must be plausible but only one correct.
    - Do NOT make options like "All of the above" or "None of the above".
                """
        elif quiz_type == "true_false":
            extra_rules = """
    - This quiz is TRUE/FALSE ONLY.
    - Questions must be statements that can be definitively true or false.
    - Avoid ambiguous statements.
                """
        elif quiz_type == "short_answer":
            extra_rules = """
    - This quiz is SHORT ANSWER ONLY.
    - Questions should require explanation or description.
    - Provide 3-5 expected points in the grading section.
    - Include 4-6 relevant keywords.
                """
        
        prompt = f"""You are a teaching assistant creating a quiz for students.

STRICT RULES:
- Use ONLY information from the context provided below
- Do NOT invent, assume, or add any facts not in the context
- Difficulty level: {difficulty}
- Generate EXACTLY {num_questions} questions - no more, no less
- ALL questions MUST be type "{quiz_type}"
- Do NOT mix question types under any circumstances
- Question IDs must be 1, 2, 3, ... up to {num_questions}
- Each explanation must be one clear sentence
{extra_rules}

OUTPUT FORMAT:
- Return ONLY valid JSON
- NO markdown code blocks (no ```json or ```)
- NO additional commentary or text
- Start directly with {{

REQUIRED JSON STRUCTURE:
{self._full_quiz_schema(quiz_type)}

CONTEXT:
{context}

TOPIC: {topic}

Generate the quiz now:"""

        MAX_RETRIES = 3
        raw = None

        for attempt in range(MAX_RETRIES):
            print(f"\n{'='*60}")
            print(f"ATTEMPT {attempt + 1}/{MAX_RETRIES}")
            print(f"{'='*60}")
            
            if raw is None:
                print("Invoking LLM...")
                raw = self.llm.invoke(prompt).content
                print(f"\nRaw Response (first 800 chars):")
                print("-" * 60)
                print(raw[:800])
                print("-" * 60)

            try:
                # Clean the response
                cleaned = self._clean_llm_response(raw)
                
                # Parse JSON
                data = json.loads(cleaned)
                print(f"✓ JSON parsed successfully")
                
                # Auto-fix common issues
                self._auto_fix_common_issues(data)
                print(f"✓ Auto-fixes applied")
                
                # Validate
                self._validate_quiz(data, quiz_type, num_questions)
                print(f"✓ Validation passed")
                print(f"\n{'='*60}")
                print(f"SUCCESS! Generated {len(data['questions'])} questions")
                print(f"{'='*60}\n")
                
                return data
                
            except json.JSONDecodeError as e:
                print(f"✗ JSON Decode Error: {e}")
                print(f"  Attempting repair...")
                raw = self._repair_json(raw, quiz_type, num_questions)
                
            except StructureError as e:
                print(f"✗ Structure Error: {e}")
                print(f"  Attempting repair...")
                raw = self._repair_json(raw, quiz_type, num_questions)
                
            except ContentError as e:
                print(f"✗ Content Error: {e}")
                print(f"  Starting fresh...")
                raw = None
                
            except Exception as e:
                print(f"✗ Unexpected Error: {type(e).__name__}: {e}")
                print(f"  Starting fresh...")
                raw = None

        raise RuntimeError(
            f"Failed to generate valid quiz after {MAX_RETRIES} attempts. "
            f"Try: (1) reducing num_questions, (2) using simpler quiz_type, "
            f"(3) checking if context is relevant to topic"
        )

    def _clean_llm_response(self, raw: str) -> str:
        """Remove markdown code blocks and other formatting"""
        # Remove markdown code blocks
        cleaned = re.sub(r'^```(?:json)?\s*\n', '', raw, flags=re.MULTILINE)
        cleaned = re.sub(r'\n```\s*$', '', cleaned, flags=re.MULTILINE)
        
        # Remove any leading/trailing whitespace
        cleaned = cleaned.strip()
        
        # If there's text before the first {, remove it
        first_brace = cleaned.find('{')
        if first_brace > 0:
            cleaned = cleaned[first_brace:]
        
        # If there's text after the last }, remove it
        last_brace = cleaned.rfind('}')
        if last_brace != -1 and last_brace < len(cleaned) - 1:
            cleaned = cleaned[:last_brace + 1]
        
        return cleaned

    # Validation methods
    def _validate_root_keys(self, data):
        if not isinstance(data, dict):
            raise StructureError("Quiz must be a JSON object")
        
        missing = REQUIRED_TOP_KEYS - data.keys()
        if missing:
            raise StructureError(f"Missing top level keys: {missing}")
        
        if not isinstance(data["questions"], list):
            raise StructureError("Questions must be an array")
        
        if len(data["questions"]) == 0:
            raise ContentError("Questions array is empty")

    def _validate_num_questions(self, data, expected_count):
        count = len(data["questions"])
        if count != expected_count:
            raise ContentError(
                f"Expected {expected_count} questions, got {count}. "
                f"Must generate exactly {expected_count} questions."
            )

    def _validate_common_question_fields(self, q, expected_type):
        if not isinstance(q, dict):
            raise StructureError("Each question must be an object")

        missing = COMMON_FIELDS - q.keys()
        if missing:
            raise StructureError(f"Missing question fields: {missing}")

        if not isinstance(q.get("grading"), dict):
            raise StructureError("grading must be an object")

        if q["type"] != expected_type:
            raise ContentError(
                f"Question type mismatch: expected '{expected_type}', got '{q['type']}'"
            )
        
        # Validate prompt is not empty
        if not q.get("prompt") or not q["prompt"].strip():
            raise ContentError("Question prompt cannot be empty")
        
        # Validate explanation is not empty
        if not q.get("explanation") or not q["explanation"].strip():
            raise ContentError("Question explanation cannot be empty")

    def _validate_mcq(self, q):
        options = q.get("options")
        if not isinstance(options, dict):
            raise StructureError("MCQ must include options as an object")

        if set(options.keys()) != {"A", "B", "C", "D"}:
            raise StructureError("MCQ options must be exactly A, B, C, D")
        
        # Check all options have content
        for key, val in options.items():
            if not val or not str(val).strip():
                raise ContentError(f"MCQ option {key} cannot be empty")

        correct = q["grading"].get("correct_option")
        if correct not in {"A", "B", "C", "D"}:
            raise ContentError(
                f"Invalid correct_option for MCQ: '{correct}'. Must be A, B, C, or D"
            )

    def _validate_true_false(self, q):
        answer = q["grading"].get("correct_answer")
        if not isinstance(answer, bool):
            raise StructureError(
                f"correct_answer must be boolean (true/false), got: {type(answer).__name__}"
            )

    def _validate_short_answer(self, q):
        grading = q["grading"]

        expected_points = grading.get("expected_points")
        if not expected_points or not isinstance(expected_points, list):
            raise ContentError("Short answer must have expected_points as a list")
        
        if len(expected_points) < 2:
            raise ContentError("Short answer must have at least 2 expected points")

        keywords = grading.get("keywords")
        if not keywords or not isinstance(keywords, list):
            raise ContentError("Short answer must have keywords as a list")
        
        if len(keywords) < 2:
            raise ContentError("Short answer must have at least 2 keywords")

        max_score = grading.get("max_score")
        if not isinstance(max_score, int) or max_score < 1:
            raise StructureError("Short answer must have max_score as a positive integer")
        
        # Check for sample_answer
        if not q.get("sample_answer") or not q["sample_answer"].strip():
            raise ContentError("Short answer must have a sample_answer")

    def _validate_quiz(self, data, quiz_type, num_questions):
        self._validate_root_keys(data)
        self._validate_num_questions(data, num_questions)

        for i, q in enumerate(data["questions"], 1):
            try:
                self._validate_common_question_fields(q, quiz_type)

                if quiz_type == "mcq":
                    self._validate_mcq(q)
                elif quiz_type == "true_false":
                    self._validate_true_false(q)
                elif quiz_type == "short_answer":
                    self._validate_short_answer(q)
                    
            except (StructureError, ContentError) as e:
                raise type(e)(f"Question {i}: {str(e)}")
        
        # Validate sequential IDs
        expected_ids = list(range(1, num_questions + 1))
        actual_ids = [q["id"] for q in data["questions"]]

        if actual_ids != expected_ids:
            raise StructureError(
                f"Question IDs must be {expected_ids}, got {actual_ids}"
            )

    def _auto_fix_common_issues(self, data):
        """Automatically fix common LLM mistakes"""
        if not isinstance(data, dict):
            return
        
        questions = data.get("questions", [])
        if not isinstance(questions, list):
            return
        
        for q in questions:
            if not isinstance(q, dict):
                continue
            
            # Fix: "question" field instead of "prompt"
            if "question" in q and "prompt" not in q:
                q["prompt"] = q.pop("question")
            
            # Fix: MCQ options as list instead of dict
            if q.get("type") == "mcq" and isinstance(q.get("options"), list):
                opts = q["options"]
                if len(opts) == 4:
                    q["options"] = {
                        "A": opts[0],
                        "B": opts[1],
                        "C": opts[2],
                        "D": opts[3],
                    }
            
            # Fix: String "true"/"false" instead of boolean
            if q.get("type") == "true_false":
                grading = q.get("grading", {})
                answer = grading.get("correct_answer")
                if isinstance(answer, str):
                    if answer.lower() == "true":
                        grading["correct_answer"] = True
                    elif answer.lower() == "false":
                        grading["correct_answer"] = False
            
            # Fix: Ensure id is integer
            if "id" in q:
                try:
                    q["id"] = int(q["id"])
                except (ValueError, TypeError):
                    pass

    def _repair_json(self, broken_json: str, quiz_type: str, num_questions: int) -> str:
        """Ask LLM to repair malformed JSON"""
        repair_prompt = f"""You are a JSON repair assistant. Your ONLY job is to fix JSON syntax errors.

RULES:
- Fix ONLY JSON structure and syntax errors (missing commas, quotes, brackets, etc.)
- Do NOT change any content, wording, or answers
- Do NOT add or remove questions
- Do NOT change question types
- Keep all existing field values exactly as they are
- Return ONLY valid JSON, no markdown, no comments

The JSON should have:
- quiz_id, difficulty, quiz_type, questions (array)
- Exactly {num_questions} questions
- All questions must be type "{quiz_type}"

BROKEN JSON:
{broken_json}

REPAIRED JSON:"""
        
        print("Asking LLM to repair JSON...")
        return self.llm.invoke(repair_prompt).content