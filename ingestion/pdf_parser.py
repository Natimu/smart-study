from PyPDF2 import PdfReader
from .base_parser import BaseParser

class PDFParser(BaseParser):
    def parse(self, file_path: str) -> str:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
