import requests
import os
import time
from dotenv import load_dotenv
import pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.embeddings import Embeddings

load_dotenv()

# Hugging Face API setup
HF_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
HF_HEADERS = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}

class HuggingFaceAPIEmbeddings(Embeddings):
    def embed_query(self, text: str, retries: int = 3, delay: int = 2):
        for attempt in range(retries):
            response = requests.post(
                f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL}",
                headers=HF_HEADERS,
                json={"inputs": text},
            )
            if response.status_code == 200:
                embeddings = response.json()
                if isinstance(embeddings, list):
                    if isinstance(embeddings[0], list):
                        return embeddings[0]
                    return embeddings
                else:
                    raise ValueError(f"Unexpected response format: {embeddings}")
            elif response.status_code == 503:
                print(f"Hugging Face API unavailable (503). Retry {attempt + 1}/{retries} after {delay}s...")
                time.sleep(delay)
            else:
                raise ValueError(f"Request failed: {response.status_code} - {response.text}")
        raise ConnectionError("Failed to get embeddings after multiple retries.")

    def embed_documents(self, texts: list[str]):
        return [self.embed_query(text) for text in texts]

# Initialize embeddings
embedding = HuggingFaceAPIEmbeddings()

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

# Initialize PineconeVectorStore
vectorstore = PineconeVectorStore(index=index, embedding=embedding, namespace="nav_indexed")

print(f"Vectorstore initialized: {vectorstore is not None}")
