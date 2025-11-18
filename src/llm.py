import requests
from logger import logger
from exception import RAGException
import config


class LLM:
    """
    Wrapper for Ollama Mistral chat model.
    """

    def __init__(self, model_name=None):
        self.model = model_name or config.LLM_MODEL  
        self.base_url = config.OLLAMA_URL            

    def generate(self, prompt: str) -> str:
        """
        Sends prompt to Ollama and returns response text.
        """

        try:
            logger.info(f"Sending request to Ollama model: {self.model}")

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=180
            )

            if response.status_code != 200:
                raise RAGException(
                    f"Ollama error {response.status_code}: {response.text}"
                )

            data = response.json()
            return data.get("response", "").strip()

        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            raise RAGException("Ollama request failed", str(e))


if __name__ == "__main__":
    llm = LLM()
    test_prompt = "What is the capital of France?"
    print(llm.generate(test_prompt))