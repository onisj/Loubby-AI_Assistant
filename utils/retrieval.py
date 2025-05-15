from pymongo import MongoClient
import os
from utils.embeddings import get_embedding
import numpy as np
from dotenv import load_dotenv

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
db = client["loubby_db"]
collection = db["docs_collection"]

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def retrieve_docs(query, top_k=3):
    try:
        query_embedding = get_embedding(query)
        all_docs = list(collection.find({}, {"text": 1, "embedding": 1}))
        if not all_docs:
            print("No documents found in collection")  # Debug
            return ["No relevant documents found"]
        similarities = []
        for doc in all_docs:
            doc_embedding = doc.get("embedding")
            if not doc_embedding:
                continue
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((doc["text"], similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [doc[0] for doc in similarities[:top_k]]
    except Exception as e:
        print(f"Error in retrieve_docs: {str(e)}")  # Debug
        return ["Error retrieving documents"]