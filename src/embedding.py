from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from logger import logger
from exception import RAGException
import os
import sys
import traceback

load_dotenv()


class HuggingFaceEmbeddingsLoader:
    def __init__(self):
        
        self.model_name = os.getenv("EMBEDDING_MODEL")
        
        

    def load_hf_embeddings(self):

        try:
            logger.info(f"[INFO] Loading embedding model: {self.model_name} (this may take a moment)...")
            hf_embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
            logger.info(f"[INFO] Successfully loaded HuggingFaceEmbeddings model: {self.model_name}")
            return hf_embeddings 
        except Exception as e:
            raise RAGException(e, sys) from e
if __name__ == "__main__":
    hf_loader = HuggingFaceEmbeddingsLoader()
    hf_embeddings = hf_loader.load_hf_embeddings()