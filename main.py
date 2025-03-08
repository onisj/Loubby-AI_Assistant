# main.py
from fastapi import FastAPI, Request, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
from app import dynamic_memory_app
from pydantic import BaseModel
from deep_translator import GoogleTranslator
from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="templates"), name="static")

# MongoDB setup for feedback
mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client["loubby_navigator"]
feedback_collection = db["feedback"]

SUPPORTED_LANGUAGES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
    "pt": "Portuguese", "nl": "Dutch", "ru": "Russian", "zh-CN": "Chinese (Simplified)",
    "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "hi": "Hindi"
}

class ChatRequest(BaseModel):
    question: str
    history: list = []
    lang: str = "en"

class FeedbackRequest(BaseModel):
    rating: int
    comment: str = ""

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open(Path("templates/index.html")) as f:
        return HTMLResponse(content=f.read())

@app.get("/healthz")
def health_check():
    return {"status": "healthy"}

@app.post("/chat")
async def chat(request: Request, body: ChatRequest):
    question = body.question
    history = body.history
    lang = body.lang.lower()

    if lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {lang}. Supported: {list(SUPPORTED_LANGUAGES.keys())}")

    translated_query = GoogleTranslator(source=lang, target="en").translate(question) if lang != "en" else question
    state = {"question": translated_query, "history": history}
    response = dynamic_memory_app.invoke(state)
    answer_en = response["answer"]
    answer = GoogleTranslator(source="en", target=lang).translate(answer_en) if lang != "en" else answer_en
    response["answer"] = answer
    return response

@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    if not 1 <= request.rating <= 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
    feedback_data = {"rating": request.rating, "comment": request.comment, "timestamp": datetime.utcnow()}
    result = feedback_collection.insert_one(feedback_data)
    return {"message": "Feedback recorded", "feedback_id": str(result.inserted_id)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            question, lang = data.split("|") if "|" in data else (data, "en")
            lang = lang.lower()
            if lang not in SUPPORTED_LANGUAGES:
                await websocket.send_text(f"Unsupported language: {lang}")
                continue
            translated_query = GoogleTranslator(source=lang, target="en").translate(question) if lang != "en" else question
            state = {"question": translated_query, "history": []}  # History could be added via WS
            response = dynamic_memory_app.invoke(state)
            answer_en = response["answer"]
            answer = GoogleTranslator(source="en", target=lang).translate(answer_en) if lang != "en" else answer_en
            await websocket.send_text(answer)
    except Exception as e:
        await websocket.send_text(f"Error: {str(e)}")
    finally:
        await websocket.close()