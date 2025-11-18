from llm import LLM as get_mistral_llm
from logger import logger
from vectordb import VectorDB
from retriever import Retriever


class RAGPipeline:
    def __init__(self,):
        
        self.llm = get_mistral_llm()
        self.context= None
    
    
    # Retrieves relevant context from the Vector DB based on the query.
    def process_context(self,query: str):
        logger.info("Processing context for RAG Pipeline...")
        db = VectorDB().build_or_load_vectorstore()
        retriever = Retriever(db)
        retriever.retrieve(query)
        
    
    # Generates an answer using the LLM with the retrieved context.
    def ask(self, query: str):
    
        prompt = f"""
        You are a context-bound QA assistant.  
        Use only the provided context to answer.

        If the context does not contain the answer, respond: "Information not found."

        Context:
        {self.context }

        Question: {query}

        Answer:

        """

        logger.info("Calling LLM for answer...")
        result = self.llm.generate(prompt)
        
        return result


if __name__ == "__main__":

    query = "According to the passage, why is abolishing caste linked to rejecting the sanctity of the shastras?"
    rag = RAGPipeline()
    rag.process_context(query)
    answer = rag.ask(query)
    print(f"Answer:\n{answer}\n")