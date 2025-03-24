import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from groq import Groq
from deep_translator import GoogleTranslator
import logging
import asyncio
from pinecone import Pinecone, ServerlessSpec
from pymongo import MongoClient
from datetime import datetime
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.responses import HTMLResponse
from starlette.websockets import WebSocketDisconnect
from docx import Document
from functools import lru_cache
import shutil
import re
import uvicorn

logging.basicConfig(level=logging.INFO, filename="logs/app.log", filemode="a", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

os.makedirs("logs", exist_ok=True)
os.makedirs("responses", exist_ok=True)
os.makedirs("static", exist_ok=True)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

app = FastAPI(title="Loubby Navigation Assistant")
app.mount("/static", StaticFiles(directory="static"), name="static")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

groq_client = Groq(api_key=GROQ_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "loubby-navigation"
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(name=INDEX_NAME, dimension=384, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
index = pc.Index(INDEX_NAME)

mongo_client = MongoClient(MONGO_URI)
db = mongo_client["loubby_assistant"]
feedback_collection = db["feedback"]
logger.info("MongoDB connected successfully")

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

SUPPORTED_LANGUAGES = {
    'afrikaans': 'af', 'albanian': 'sq', 'amharic': 'am', 'arabic': 'ar', 'armenian': 'hy', 'assamese': 'as', 'aymara': 'ay', 'azerbaijani': 'az',
    'bambara': 'bm', 'basque': 'eu', 'belarusian': 'be', 'bengali': 'bn', 'bhojpuri': 'bho', 'bosnian': 'bs', 'bulgarian': 'bg', 'catalan': 'ca',
    'cebuano': 'ceb', 'chichewa': 'ny', 'chinese (simplified)': 'zh-CN', 'chinese (traditional)': 'zh-TW', 'corsican': 'co', 'croatian': 'hr',
    'czech': 'cs', 'danish': 'da', 'dhivehi': 'dv', 'dogri': 'doi', 'dutch': 'nl', 'english': 'en', 'esperanto': 'eo', 'estonian': 'et', 'ewe': 'ee',
    'filipino': 'tl', 'finnish': 'fi', 'french': 'fr', 'frisian': 'fy', 'galician': 'gl', 'georgian': 'ka', 'german': 'de', 'greek': 'el',
    'guarani': 'gn', 'gujarati': 'gu', 'haitian creole': 'ht', 'hausa': 'ha', 'hawaiian': 'haw', 'hebrew': 'iw', 'hindi': 'hi', 'hmong': 'hmn',
    'hungarian': 'hu', 'icelandic': 'is', 'igbo': 'ig', 'ilocano': 'ilo', 'indonesian': 'id', 'irish': 'ga', 'italian': 'it', 'japanese': 'ja',
    'javanese': 'jw', 'kannada': 'kn', 'kazakh': 'kk', 'khmer': 'km', 'kinyarwanda': 'rw', 'konkani': 'gom', 'korean': 'ko', 'krio': 'kri',
    'kurdish (kurmanji)': 'ku', 'kurdish (sorani)': 'ckb', 'kyrgyz': 'ky', 'lao': 'lo', 'latin': 'la', 'latvian': 'lv', 'lingala': 'ln',
    'lithuanian': 'lt', 'luganda': 'lg', 'luxembourgish': 'lb', 'macedonian': 'mk', 'maithili': 'mai', 'malagasy': 'mg', 'malay': 'ms',
    'malayalam': 'ml', 'maltese': 'mt', 'maori': 'mi', 'marathi': 'mr', 'meiteilon (manipuri)': 'mni-Mtei', 'mizo': 'lus', 'mongolian': 'mn',
    'myanmar': 'my', 'nepali': 'ne', 'norwegian': 'no', 'odia (oriya)': 'or', 'oromo': 'om', 'pashto': 'ps', 'persian': 'fa', 'polish': 'pl',
    'portuguese': 'pt', 'punjabi': 'pa', 'quechua': 'qu', 'romanian': 'ro', 'russian': 'ru', 'samoan': 'sm', 'sanskrit': 'sa', 'scots gaelic': 'gd',
    'sepedi': 'nso', 'serbian': 'sr', 'sesotho': 'st', 'shona': 'sn', 'sindhi': 'sd', 'sinhala': 'si', 'slovak': 'sk', 'slovenian': 'sl',
    'somali': 'so', 'spanish': 'es', 'sundanese': 'su', 'swahili': 'sw', 'swedish': 'sv', 'tajik': 'tg', 'tamil': 'ta', 'tatar': 'tt', 'telugu': 'te',
    'thai': 'th', 'tigrinya': 'ti', 'tsonga': 'ts', 'turkish': 'tr', 'turkmen': 'tk', 'twi': 'ak', 'ukrainian': 'uk', 'urdu': 'ur', 'uyghur': 'ug',
    'uzbek': 'uz', 'vietnamese': 'vi', 'welsh': 'cy', 'xhosa': 'xh', 'yiddish': 'yi', 'yoruba': 'yo', 'zulu': 'zu'
}

@lru_cache()
def load_docx(file_path):
    logger.info(f"Loading {file_path}")
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

@lru_cache()
def get_chunks():
    applicant_text = load_docx("Applicant_Journey.docx")
    recruiter_text = load_docx("Recruiter_Dashboard.docx")
    combined_text = applicant_text + "\n\n" + recruiter_text
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return splitter.split_text(combined_text)

@lru_cache()
def initialize_embeddings():
    stats = index.describe_index_stats()
    vector_count = stats['namespaces'].get('loubby', {}).get('vector_count', 0)
    logger.info(f"Pinecone loubby namespace vector count: {vector_count}")
    if vector_count == 0:
        logger.info("Initializing embeddings for Applicant_Journey.docx and Recruiter_Dashboard.docx")
        chunks = get_chunks()
        chunk_embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            index.upsert(vectors=[(f"chunk_{i}", embedding, {"text": chunk, "rating_score": 0.0})], namespace="loubby")
        logger.info("Embeddings initialized")
    else:
        logger.info("Embeddings already initialized")

initialize_embeddings()

COMMON_QUERIES = {
    "What is Loubby?": {
        "embedding": embedding_model.encode(["What is Loubby?"])[0],
        "response_en": "Loubby is a job navigation platform that offers a recruitment portal for job seekers and an employee portal for hired candidates. It provides features such as profile setup, mobile access, submission of assignments, and access to a virtual interview room. To create an account, users can visit [loubby.ai/recruitment] and follow the sign-up process. Once a job offer is accepted, users receive employee portal login credentials for continued use."
    }
}

class ChatRequest(BaseModel):
    query: str
    lang: str = "en"
    include_audio: bool = False

class FeedbackRequest(BaseModel):
    rating: int
    comment: str = ""

@lru_cache(maxsize=32)
def generate_feedback_summary(limit=5):
    recent_feedback = feedback_collection.find().sort("timestamp", -1).limit(limit)
    summaries = []
    for fb in recent_feedback:
        rating = fb["rating"]
        comment = fb.get("comment", "")
        if rating < 3 and comment: summaries.append(f"Critique: '{comment}' (rating: {rating})")
        elif rating >= 4 and comment: summaries.append(f"Praise: '{comment}' (rating: {rating})")
    return "Recent feedback: " + "; ".join(summaries) if summaries else "No recent feedback available."

def update_chunk_ratings(query: str, rating: int):
    query_embedding = embedding_model.encode([query])[0]
    search_results = index.query(vector=query_embedding.tolist(), top_k=5, include_metadata=True, namespace="loubby")
    for match in search_results["matches"]:
        chunk_id = match["id"]
        current_score = match["metadata"].get("rating_score", 0.0)
        new_score = current_score + (rating - 3) * 0.1
        index.update(id=chunk_id, set_metadata={"rating_score": new_score}, namespace="loubby")
        logger.info(f"Updated chunk {chunk_id} rating to {new_score}")

def clean_response(response: str) -> str:
    patterns = [
        r"(Internal Feedback Summary|feedback summaries|recent feedback|critiques about|for internal use only|praise for)[^.!?]*[.!?]",
        r"\b(rating: \d|ai video|nice interface)\b[^.!?]*[.!?]"
    ]
    cleaned = response
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()

def normalize_language_code(lang: str) -> str:
    """Normalize language codes to match SUPPORTED_LANGUAGES."""
    lang = lang.lower()
    if lang.startswith("en"):  # Handle en-US, en-GB, etc.
        return "en"
    if lang.startswith("es"):  # Handle es-ES, es-MX, etc.
        return "es"
    if lang.startswith("pt"):  # Handle pt-PT, pt-BR, etc.
        return "pt"
    if lang.startswith("zh"):  # Handle zh-CN, zh-TW
        return lang  # Keep specific Chinese variants
    return lang  # Default to input if no special case

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    logger.info("Attempting WebSocket connection")
    await websocket.accept()
    logger.info("WebSocket connection established")
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Raw WebSocket data received: '{data}'")
            query, target_lang = data.split("|") if "|" in data else (data, "en")
            target_lang = normalize_language_code(target_lang)
            logger.info(f"Processed query: '{query}', Normalized target lang: '{target_lang}'")
            if not query.strip():
                await websocket.send_text("Error: Empty query received")
                logger.warning("Empty query received")
                continue
            if target_lang not in SUPPORTED_LANGUAGES.values():
                await websocket.send_text(f"Error: {target_lang} --> No support for the provided language. Please select one of the supported languages: {SUPPORTED_LANGUAGES}")
                logger.warning(f"Unsupported language: {target_lang}")
                continue
            translated_query = GoogleTranslator(source=target_lang, target="en").translate(query) if target_lang != "en" else query

            if translated_query in COMMON_QUERIES:
                response_en = COMMON_QUERIES[translated_query]["response_en"]
            else:
                query_embedding = COMMON_QUERIES.get(translated_query, {}).get("embedding", embedding_model.encode([translated_query])[0])
                search_results = index.query(vector=query_embedding.tolist(), top_k=5, include_metadata=True, namespace="loubby")
                context = " ".join([match["metadata"]["text"] for match in sorted(
                    search_results["matches"], key=lambda x: x["metadata"].get("rating_score", 0.0), reverse=True)])
                feedback_summary = generate_feedback_summary()
                prompt = (
                    f"You are an assistant for the Loubby platform, a job navigation tool. Answer strictly based on this context: {context}\n"
                    f"Internal Feedback Summary (FOR INTERNAL USE ONLY, DO NOT INCLUDE IN RESPONSE): {feedback_summary}\n"
                    f"Query: {translated_query}\n"
                    "Provide a clear, concise answer relevant to Loubby features. Under no circumstances mention feedback, ratings, or unrelated topics like Python or songs unless explicitly asked."
                )
                try:
                    response_en = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, lambda: groq_client.chat.completions.create(
                            model="mixtral-8x7b-32768",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=300,
                            temperature=0.7
                        ).choices[0].message.content),
                        timeout=5.0
                    )
                    response_en = clean_response(response_en)
                except Exception as e:
                    logger.error(f"Groq API error: {str(e)}")
                    response_en = "Sorry, I couldn’t process your request right now."

            response = GoogleTranslator(source="en", target=target_lang).translate(response_en)
            await websocket.send_text(response)
            logger.info(f"Sent response: '{response}'")
    except WebSocketDisconnect as e:
        logger.info(f"WebSocket disconnected: {e.code} - {e.reason}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        if websocket.client_state == 1:  # CONNECTED
            await websocket.send_text(f"Error: {str(e)}")
    finally:
        if websocket.client_state == 1:  # CONNECTED
            await websocket.close()
            logger.info("WebSocket closed cleanly")

@app.get("/")
async def get_frontend():
    logger.info("Serving index.html")
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    try:
        if not 1 <= request.rating <= 5:
            raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
        feedback_data = {"rating": request.rating, "comment": request.comment, "timestamp": datetime.utcnow()}
        result = feedback_collection.insert_one(feedback_data)
        update_chunk_ratings(request.comment, request.rating)
        logger.info(f"Feedback stored with ID: {result.inserted_id}")
        return {"message": "Feedback recorded", "feedback_id": str(result.inserted_id)}
    except Exception as e:
        logger.error(f"Feedback error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
@limiter.limit("10/minute")
async def chat(request: Request, chat_request: ChatRequest):
    try:
        query = chat_request.query
        target_lang = normalize_language_code(chat_request.lang)
        logger.info(f"Received query: '{query}', Normalized target lang: '{target_lang}'")
        if target_lang not in SUPPORTED_LANGUAGES.values():
            return {"answer": f"Error: {target_lang} --> No support for the provided language. Please select one of the supported languages: {SUPPORTED_LANGUAGES}"}
        translated_query = GoogleTranslator(source=target_lang, target="en").translate(query) if target_lang != "en" else query
        if translated_query in COMMON_QUERIES:
            response_en = COMMON_QUERIES[translated_query]["response_en"]
        else:
            query_embedding = COMMON_QUERIES.get(translated_query, {}).get("embedding", embedding_model.encode([translated_query])[0])
            search_results = index.query(vector=query_embedding.tolist(), top_k=5, include_metadata=True, namespace="loubby")
            context = " ".join([match["metadata"]["text"] for match in sorted(
                search_results["matches"], key=lambda x: x["metadata"].get("rating_score", 0.0), reverse=True)])
            feedback_summary = generate_feedback_summary()
            prompt = (
                f"You are an assistant for the Loubby platform, a job navigation tool. Answer strictly based on this context: {context}\n"
                f"Internal Feedback Summary (FOR INTERNAL USE ONLY, DO NOT INCLUDE IN RESPONSE): {feedback_summary}\n"
                f"Query: {translated_query}\n"
                "Provide a clear, concise answer relevant to Loubby features. Under no circumstances mention feedback, ratings, or unrelated topics like Python or songs unless explicitly asked."
            )
            response_en = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, lambda: groq_client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.7
                ).choices[0].message.content),
                timeout=5.0
            )
            response_en = clean_response(response_en)
        response = GoogleTranslator(source="en", target=target_lang).translate(response_en)
        logger.info(f"Sent response: '{response}'")
        return {"answer": response}
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")