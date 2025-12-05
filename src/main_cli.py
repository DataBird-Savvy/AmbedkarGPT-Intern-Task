import argparse
from rag_pipeline import RAGPipeline
from logger import logger



# This function initializes the CLI, retrieves context using the RAG pipeline, 
# generates the answer, and prints the final output.
def main():
    
    rag = RAGPipeline()
    
    parser = argparse.ArgumentParser(description="AmbedkarGPT CLI")
    parser.add_argument("--query", "-q", required=True, help="Your question")

    args = parser.parse_args()
    query = args.query
    
    rag.process_context(query)
         

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
