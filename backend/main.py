# main.py
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from app import dynamic_memory_app

load_dotenv()
os.environ.pop("SSL_CERT_FILE", None)

app = FastAPI(title="Loubby Navigator", description="AI Assistant developed by Team Sigma")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str
    history: list[str]

@app.post("/chat")
def chat(question: Question):
    state = {
        "question": question.question,
        "history": question.history
    }
    response = dynamic_memory_app.invoke(state)
    return {"answer": response["answer"], "history": response["history"]}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Use Render's assigned PORT
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
