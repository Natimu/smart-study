import os
import json
from typing import List
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ingestion.pdf_parser import PDFParser
from retrieval.vector_retriever import VectorRetriever

class SubjectManager:
    SUBJECT_METADATA_FILE = "metadata.json"

    def __init__(self, embedder):
       self.embedder = embedder
       self._load_metadata()

# Handel metadata
    def _load_metadata(self):
        if not os.path.exists(self.SUBJECT_METADATA_FILE):
            self.metadata = {"subjects": {}}
            self._save_metadata()
        else:
            with open(self.SUBJECT_METADATA_FILE, "r") as f:
                self.metadata = json.load(f)
                if "subjects" not in self.metadata:
                    self.metadata["subjects"] = {}
                    self._save_metadata()


    def _save_metadata(self):
        with open(self.SUBJECT_METADATA_FILE, "w") as f:
            json.dump(self.metadata, f, indent=2)

# Manage Subjects

    def create_subject(self, subject_id: str, display_name: str):
        if subject_id in self.metadata["subjects"]:
            raise ValueError("Subject already exist")
        
        path = f"./db/{subject_id}"
        os.makedirs(path, exist_ok=True)

        self.metadata["subjects"][subject_id] ={
            "name": display_name,
            "path": path,
            "files":[]
        }
        self._save_metadata()

    def list_subjects(self):
        return self.metadata["subjects"]
    def subject_exist(self, subject_id: str) -> bool:
        return subject_id in self.metadata["subjects"]
    

# file ingestion
    def ingest_files(self, subject_id: str, file_paths: List[str]):
        if subject_id not in self.metadata["subjects"]:
            raise ValueError("Subject does not exist")
        
        subject = self.metadata["subjects"][subject_id]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 800,
            chunk_overlap = 100
        )

        all_chunks = []
        for file_path in file_paths:
            file_name = os.path.basename(file_path)

            # if file is already in the chunk skip
            if file_name in subject["files"]:
                continue

            parser = PDFParser()
            text = parser.parse(file_path)
            chunks = splitter.split_text(text)
            all_chunks.extend(chunks)

            subject["files"].append(file_name)
        if not all_chunks:
            return
        
        Chroma.from_texts(
            all_chunks, 
            embedding = self.embedder, 
            persist_directory=subject["path"]
        )

        self._save_metadata()

# Retrieval
    def get_retriever(self, subject_id: str) -> VectorRetriever:
        if subject_id not in self.metadata["subjects"]:
            raise ValueError("Subject does not exist")
        path = self.metadata["subjects"][subject_id]["path"]

        vectorstore = Chroma(
            persist_directory = path,
            embedding_function = self.embedder
        )
        return VectorRetriever(vectorstore)

