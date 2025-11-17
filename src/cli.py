import argparse
from rag_pipeline import RAGPipeline
from logger import logger
from vectordb import VectorDB
from retriever import Retriever

def main():
    
    db = VectorDB().build_or_load_vectorstore()
    retriever = Retriever(db)
    query = "According to the passage, why is abolishing caste linked to rejecting the sanctity of the shastras?"
    
    parser = argparse.ArgumentParser(description="AmbedkarGPT CLI")
    parser.add_argument("--query", "-q", required=True, help="Your question")

    args = parser.parse_args()
    query = args.query
    results = retriever.retrieve(query)
    rag = RAGPipeline(results)
    
   

    try:
        answer = rag.ask(query)
        print("\n================= ANSWER =================\n")
        print(answer)
        print("\n==========================================\n")

    except Exception as e:
        logger.error(f"CLI Error: {str(e)}")
        print("Error occurred. Check logs.")

if __name__ == "__main__":
    
    main()
