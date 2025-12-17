from subjects.subject_manager import SubjectManager
from embedding.local_embedder import LocalEmbedder
from llm_chains.local_explanation import LocalExplanation
from llm_chains.quiz_chain import QuizChain
from llm_chains.summary_chain import SummaryChain

embedder = LocalEmbedder()
manager = SubjectManager(embedder)

# Create subject once


# Upload files
manager.ingest_files("networks", [
    "Chapter_05.pdf",
    "Chapter_06.pdf",
    "Chapter_07.pdf"
])
retriever = manager.get_retriever("networks")

# # Query

# explainer = LocalExplanation(retriever)

# answer = explainer.run("what is Error detection and correction in link layer")
# print(answer)

quiz = QuizChain(retriever)
summary = SummaryChain(retriever)

print("\n=== QUIZ ===\n")
print(quiz.run("Wireless and Mobile Networks", number_questions=6, quiz_type="mcq", difficulty="exam", top_k=6))

print("\n=== CHEAT SHEET ===\n")
print(summary.run("Mobile IP", top_k=8))
