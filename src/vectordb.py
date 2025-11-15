import config
from langchain_chroma.vectorstores import Chroma
import os
from logger import logger
from exception import RAGException


class VectorDB:
    def __init__(self,chunks=None, embeddings=None):
        self.persist_directory = config.CHROMA_DIR
        self.vectordb = None
        self.chunks = chunks
        self.embeddings = embeddings

    def build_or_load_vectorstore(self):
        try:
            persist_path = os.path.abspath(self.persist_directory)
            os.makedirs(persist_path, exist_ok=True)
            logger.info(f"[INFO] Creating/loading Chroma DB at: {persist_path}")

            if self.chunks and self.embeddings:
                
                self.vectordb = Chroma.from_documents(
                    collection_name=config.COLLECTION_NAME,
                    documents=self.chunks,
                    embedding=self.embeddings,
                    persist_directory=persist_path
                )
                
            else:
                
                self.vectordb = Chroma(
                    collection_name=config.COLLECTION_NAME,
                    persist_directory=persist_path
                )

            logger.info(f"[INFO] Chroma vectorstore is ready at: {persist_path}")
            return self.vectordb

        except Exception as e:
            logger.error(f"Error in building/loading vectorstore: {e}")
            raise RAGException(f"VectorDB Error: {e}", error_detail=e)
            
                
       
        
if __name__ == "__main__":
    from embedding import HuggingFaceEmbeddingsLoader
    from document_processor import DocumentProcessor
    embeddings = HuggingFaceEmbeddingsLoader().load_hf_embeddings()
    chunks= DocumentProcessor().process(config.SPEECH_FILE)
    vectordb = VectorDB(chunks=chunks, embeddings=embeddings)
    vectordb.build_or_load_vectorstore()