from langchain_cohere import CohereRerank
import config
from logger import logger
from exception import RAGException
from langchain_community.retrievers import BM25Retriever
from graph_retriever import Graph as KGRetriever
from dotenv import load_dotenv
from langchain_core.documents import Document
load_dotenv()


class Retriever:
    # Initializes the retriever with a vector database and top-k value.
    def __init__(self, vectordb):
        self.vectordb = vectordb
        self.vector_k = config.VECTOR_K

    # Formats retrieved documents for readable logging.
    def pretty_print_docs(self, docs):
        return "\n" + "\n".join(
            [f"{'-'*100}\nDocument {i+1}:\n\n{d.page_content}"
             for i, d in enumerate(docs)]
        )

    # Retrieves documents using vector search + BM25 + Cohere reranker.
    def retrieve(self, query):
        logger.info("Retrieving relevant documents from vector database.")
        
        try:
            # Step 1: Vector similarity search
            vector_docs = self.vectordb.similarity_search(query, k=self.vector_k)
            logger.info(f"Retrieved {len(vector_docs)} docs from vector search.")
            logger.info(f"Vector Docs: {self.pretty_print_docs(vector_docs)}")

            # Step 2: BM25 keyword search
            all_docs = self.vectordb._collection.get(include=["documents"])["documents"]
            all_docs = [Document(page_content=text) for text in all_docs]
            bm25_retriever = BM25Retriever.from_documents(all_docs)
            bm25_docs = bm25_retriever.invoke(query)
            logger.info(f"Retrieved {len(bm25_docs)} docs from keyword search.")
            logger.info(f"BM25 Docs: {self.pretty_print_docs(bm25_docs)}")
            
            # Step 3. Knowledge Graph Retrieval (no cypher needed)
            kg_docs = KGRetriever().kg_retrieve(query)
            logger.info(f"Retrieved {len(kg_docs)} docs from graph.")
            logger.info(f"KG Docs: {self.pretty_print_docs(kg_docs)}")

            # Step 4: Combine (remove duplicates)
            combined_docs = {doc.page_content: doc for doc in vector_docs + bm25_docs+ kg_docs}
            combined_docs = list(combined_docs.values())

            # Step 5: Cohere reranking
            reranker = CohereRerank(model=config.COHERE_RERANK_MODEL)
            strings = [d.page_content for d in combined_docs]

            response = reranker.rerank(
                query=query,
                documents=strings,
            )

            # Step 6: Sort by relevance score
            reranked_docs = sorted(
                response, key=lambda x: x["relevance_score"], reverse=True
            )

            final_docs = [bm25_docs[d["index"]] for d in reranked_docs]
            
            logger.info(
                f"Final hybrid retrieval (Vector + BM25 + KG): {self.pretty_print_docs(final_docs)}"
            )

          

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
