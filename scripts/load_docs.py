# scripts/load_docs.py
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.embedding import get_embedding
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
db = client["loubby_db"]
collection = db["docs_collection"]

loader = TextLoader("../docs/loubby_docs.md")  # Path relative to scripts/
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

for i, chunk in enumerate(chunks):
    embedding = get_embedding(chunk.page_content)
    doc = {
        "text": chunk.page_content,
        "metadata": {"id": i, "section": "Job Search" if "job" in chunk.page_content.lower() else "Course Syncing"},
        "embedding": embedding
    }
    collection.insert_one(doc)

print("Docs loaded into MongoDB!")