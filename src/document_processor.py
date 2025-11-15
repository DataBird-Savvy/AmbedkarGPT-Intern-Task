from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from transformers import AutoTokenizer
from logger import logger
from exception import RAGException
import config



class DocumentProcessor:
    """
    A class for loading and splitting documents using RecursiveCharacterTextSplitter.
    """

    def __init__(self):
        self.chunk_size: int =  config.CHUNK_SIZE
        self.chunk_overlap: int = config.CHUNK_OVERLAP
        self.model_name: str = config.EMBEDDING_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            
        )

    def load_documents(self, path: str) -> List[Document]:
        try:
            loader = TextLoader(path, encoding="utf-8")
            logger.info(f"Loaded document from {path}")
            return loader.load()
        except Exception as e:
            logger.error(f"Error loading document from {path}: {str(e)}")
            raise RAGException(f"Failed to load document: {str(e)}")
           

    def split_documents(self, docs: List[Document]) -> List[Document]:
        try:
            chunk_docs = self.splitter.split_documents(docs)
            logger.info(
                f"[INFO] Loaded {len(docs)} document(s) "
                f"and split into {len(chunk_docs)} chunk(s)."
            )
            
            return chunk_docs
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            raise RAGException(f"Failed to split documents: {str(e)}")

    def process(self, path: str) -> List[Document]:
        try:
            logger.info(f"Starting document processing for {path}") 
            
            docs = self.load_documents(path)
            return self.split_documents(docs)
        except RAGException as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise RAGException(f"Document processing failed: {str(e)}")


if __name__ == "__main__":
    processor = DocumentProcessor()
    documents = processor.process("data/speech.txt")