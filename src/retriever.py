from langchain_cohere import  CohereRerank
import config
from logger import logger
from exception import RAGException
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
from langchain_core.documents import Document
load_dotenv()





class Retriever:
    def __init__(self, vectordb):
        self.vectordb = vectordb
        self.vector_k = config.VECTOR_K
    


    def pretty_print_docs(self,docs):
        print(
            f"\n{'-' * 100}\n".join(
                [f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]
            )
        )
    def retrieve(self, query):
        logger.info("Retrieving relevant documents from vector database.")
        
        try:
            vector_docs = self.vectordb.similarity_search(query, k=self.vector_k)
            logger.info(f"Retrieved {len(vector_docs)} docs from vector search.")
            
            
            
            all_docs = self.vectordb._collection.get(include=["documents"])["documents"]
            all_docs = [Document(page_content=text) for text in all_docs]
            bm25_retriever = BM25Retriever.from_documents(all_docs)
            bm25_docs = bm25_retriever.invoke(query) 
            logger.info("BM25Retriever reranking applied based on query.")
            
            
            # Combine and remove duplicates
            combined_docs = {doc.page_content: doc for doc in vector_docs + bm25_docs}
            combined_docs = list(combined_docs.values())


                
            
            reranker = CohereRerank(model=config.COHERE_RERANK_MODEL)
            strings = [d.page_content for d in combined_docs]
            response = reranker.rerank(
                query=query,
                documents=strings,
            )

            
            reranked_docs = sorted(
                response, key=lambda x: x["relevance_score"], reverse=True
            )

            final_docs = [bm25_docs[d["index"]] for d in reranked_docs]

            self.pretty_print_docs(final_docs)

            return final_docs
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            raise RAGException(str(e), e)

        
        
    
    
if __name__ == "__main__":
    from vectordb import VectorDB
    db = VectorDB().build_or_load_vectorstore()
    retriever = Retriever(db)
    query = "What is Ambedkar's view on social justice?"
    results = retriever.retrieve(query)
   








