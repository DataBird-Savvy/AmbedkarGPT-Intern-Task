from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from config import *
from logger import logger

class Graph:
    def __init__(self):
        
        self.vector_k = VECTOR_K

        # Updated non-deprecated Neo4jGraph
        self.graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASS
        )

        self.llm = OllamaLLM(model="llama3")

        self.graph_transformer = LLMGraphTransformer(llm=self.llm)

        # Updated non-deprecated GraphCypherQAChain
        self.kg_chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=False,
            allow_dangerous_requests=True  
        )

    def save_to_graph(self, text: str):
        try:
            docs = [Document(page_content=text)]
            triples = self.graph_transformer.convert_to_graph_documents(docs)
            self.graph.add_graph_documents(triples)
            logger.info("Saved extracted triples to Neo4j.")
            return True

        except Exception as e:
            logger.error(f"Graph save error: {e}")
            return False

    def kg_retrieve(self, query):
        try:
            response = self.kg_chain.invoke({"query": query})
            text = str(response)
            logger.info("KG retrieval done using GraphCypherQAChain.")
            return [Document(page_content=text)]

        except Exception as e:
            logger.error(f"KG retrieval error: {e}")
            return []

        
if __name__ == "__main__":
    graph = Graph()
    with open("data/speech.txt", "r") as f:
        text = f.read()
    graph.save_to_graph(text)
    answer_docs = graph.kg_retrieve("What is described as the “real remedy” for solving the caste problem?")
    for doc in answer_docs:
        print(doc.page_content)
    
    
  