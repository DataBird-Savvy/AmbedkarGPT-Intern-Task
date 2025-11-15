from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from transformers import AutoTokenizer
from dotenv import load_dotenv
import os

load_dotenv()


class DocumentProcessor:
    """
    A class for loading and splitting documents using RecursiveCharacterTextSplitter.
    """

    def __init__(self):
        self.chunk_size: int =  int(os.getenv("CHUNK_SIZE"))
        self.chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP"))
        self.model_name: str = os.getenv("EMBEDDING_MODEL")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            
        )

    def load_documents(self, path: str) -> List[Document]:
        loader = TextLoader(path, encoding="utf-8")
        return loader.load()

    def split_documents(self, docs: List[Document]) -> List[Document]:
        chunk_docs = self.splitter.split_documents(docs)
        print(
            f"[INFO] Loaded {len(docs)} document(s) "
            f"and split into {len(chunk_docs)} chunk(s)."
        )
        # print(chunk_docs)
        return chunk_docs

    def process(self, path: str) -> List[Document]:
        docs = self.load_documents(path)
        return self.split_documents(docs)


if __name__ == "__main__":
    processor = DocumentProcessor()
    documents = processor.process("data/speech.txt")