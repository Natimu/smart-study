from subjects.subject_manager import SubjectManager
from embedding.local_embedder import LocalEmbedder
from llm_chains.quiz_chain import QuizChain
from llm_chains.summary_chain import SummaryChain
# from llm_chains.quiz_gener_chain import QuizChain

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
print(quiz.run(topic="Open Shortest Path First", num_questions=3, quiz_type="true_false", difficulty="exam", top_k=3))

# print("\n=== CHEAT SHEET ===\n")
# print(summary.run("Intra-AS Routing in the Internet", top_k=8))
