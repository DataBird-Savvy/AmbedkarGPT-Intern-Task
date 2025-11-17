from llm import LLM as get_mistral_llm
from logger import logger



class RAGPipeline:
    def __init__(self,context : None):
        self.context = context
        self.llm = get_mistral_llm()

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
        print("Result:", result)
        return result


if __name__ == "__main__":
    from vectordb import VectorDB
    from retriever import Retriever
    db = VectorDB().build_or_load_vectorstore()
    retriever = Retriever(db)
    query = "According to the passage, why is abolishing caste linked to rejecting the sanctity of the shastras?"
    results = retriever.retrieve(query)
    rag = RAGPipeline(results)
    answer = rag.ask(query)
    print(f"Answer:\n{answer}\n")