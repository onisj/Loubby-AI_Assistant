import os
import pinecone
import uuid_utils as uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone, Serverlessspec
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from groq import Groq
from langchain.embeddings.openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from statistics import mean

# load environment variables 
load_dotenv()
app = FastAPI()

#initialize Groq client
groq_client = Groq(api_ket=os.getenv("GROQ_API-KEY"))

#Initialize sentence Transformer for embeddings
embeddings = SentenceTransformer('all-MiniLM-L6-v2')

#Initialize Pinecone Serverless
pc = Pinecone(api_key="API_KEY")
index_name  = "ai-feedback index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  #Adjust based on our embedding model
        metric="cosine",
        spec=Serverlessspec(cloud="aws", region="us-east-1")    
    )

# Connect to pinecone index
index = pc.Index(index_name)

# Pydantic models 
class QueryRequest(BaseModel):
    query: str

class FeedbackRequest(BaseModel):
    query: str
    response: str
    rating: int
    comment: str = ""

#Generate response using Groq API
def generate_response(query: str) -> str:
    prompt = PromptTemplate(
        input_variables=["query"],
        template="Answer clearly and concisely: {query}"
    )
    formatted_prompt = prompt.format(query=query)

    completion = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": formatted _prompt}
         ],
     temperature=0.7
     )
    return completion.choices[0].message.content 

# API endpoint
@app.post("/query")
async def query_assistant(request: QueryRequest):
    response = generate_response(request.query)
    return {"query": request.query, "respnse": response}

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    #genrate embeddings for query-respose pair
    text_to_embed = f"Query: {request.query}\nResponse: {request.response}"
    embedding = embeddings.encode(text_to_embed).tolist() #convert to list for pinecone

    #store in pinecone
    metadata = {
        "query": request.query,
        "response": request.response,
        "rating": request.rating,
        "comment": request.comment
    }
    unique_id = str(uuid())
    index.upsert([(unique_id, embedding, metadata)])

    return {"message": "Feedbck recorded successfully"}

#initialize embeddings
embeddings = SentenceTransformer('all-MiniLM-L6-v2')

#initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API-KEY"))
index = pc.Index("feedback-index")

def analyze_feedback():
    #Fetch feedback (using a zero vector as a placeholder)
    results = index.query(vector=[0] * 384, top_k=1000, include_metadata=True)["matches"]

    ratings = [match["metadata"]["rating"] for match in results]
    avg_rating = mean(ratings) if ratings else 0
    print(f"Average Rating: {avg_rating}")

    low_rated = [match for match in results if match["metadata"]["rating"] < 3]
    for entry in low_rated:
        meta = entry["metadata"]
        print(f"Low-rated: Query: {meta['query']}, Response: {meta['resonse']}, comment: {meta['comment']}")

    if low_rated:
        sample_text = f"Query: {low_rated[0]['metadata']['query']}\nResponse: {low_rated[0]['metadata']['response']}"
        sample_embedding = embeddings.encode(sample_text).tolist()
        similar = index.query(vector=sample_embedding, top_k=5, include_metadata=True)
        print("siimilar low-rated response:", [match["metadata"] for match in similar["matches"]])

if __name__ == "__name__":
    analyze_feedback()

