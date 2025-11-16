from langchain_cohere import ChatCohere, CohereRerank
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from dotenv import load_dotenv
import os
import config
load_dotenv()
from logger import logger
from exception import RAGException
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


class Retriever:
    def __init__(self, vectordb,vector_k=4, alpha=0.5):
        self.vectordb = vectordb
        # self.co = cohere.Client(os.getenv("COHERE_API_KEY"))
        self.vector_k = config.VECTOR_K
    # Helper function for printing docs


    def pretty_print_docs(self,docs):
        print(
            f"\n{'-' * 100}\n".join(
                [f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]
            )
        )
    def retrieve(self, query):
        logger.info("Retrieving relevant documents from vector database.")
        
        #  Vector similarity search
        top_docs = self.vectordb.similarity_search(query, k=self.vector_k)
        logger.info(f"Retrieved {len(top_docs)} docs from vector search.")
        
        # 2️⃣ BM25 rerank using query tokens
        
        bm25_retriever = BM25Retriever.from_documents(top_docs)
        bm25_docs = bm25_retriever.invoke(query) 
        logger.info("BM25Retriever reranking applied based on query.")

              
        #  Cohere ReRank final rerank
        reranker = CohereRerank(model="rerank-english-v3.0")

        response = reranker.rerank(
            query=query,
            documents=bm25_docs,
        )

        # Sort using Cohere score
        reranked_docs = sorted(
            response, key=lambda x: x["relevance_score"], reverse=True
        )

        final_docs = [bm25_docs[d["index"]] for d in reranked_docs]

        self.pretty_print_docs(final_docs)

        return final_docs
        
        
    
    
if __name__ == "__main__":
    from vectordb import VectorDB
    db = VectorDB().build_or_load_vectorstore()
    retriever = Retriever(db)
    query = "What is Ambedkar's view on social justice?"
    results = retriever.retrieve(query)
    for i, res in enumerate(results):
        print(f"Result {i+1}:\n{res.page_content}\n")









