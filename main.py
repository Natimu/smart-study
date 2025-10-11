from ingestion.pdf_parser import PDFParser
from embedding.openai_embedder import OpenAIEmbedder
from retrieval.vector_retriever import VectorRetriever
from llm_chains.explanation_chain import ExplanationChain
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Parse PDF
parser = PDFParser()
text = parser.parse("Natnael.pdf")

# 2. Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_text(text)

# 3. Create embeddings using your wrapper class
embedder = OpenAIEmbedder()

vectorstore = Chroma.from_texts(
    chunks,
    embedding=embedder,             
    persist_directory="./db"
)


# 4. Setup retriever and explanation chain
retriever = VectorRetriever(vectorstore)
explainer = ExplanationChain(retriever)

# 5. Ask a question
query = "Explain the TCP/IP model"
answer = explainer.run(query)
print(answer)
