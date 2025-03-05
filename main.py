from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from groq import Groq
from pinecone import Pinecone, ServerlessSpec
from google.cloud import speech, texttospeech
from langdetect import detect
import logging
import asyncio
import uuid
from pymongo import MongoClient
from datetime import datetime
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import os
from starlette.responses import HTMLResponse
from docx import Document
import numpy as np
import pyaudio
from aiortc import RTCPeerConnection, RTCSessionDescription, AudioStreamTrack

# Logging setup
logging.basicConfig(level=logging.INFO, filename="logs/app.log", filemode="a", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


os.makedirs("logs", exist_ok=True)
os.makedirs("responses", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://loubby:loubby@loubby-cluster.oolrx.mongodb.net/?retryWrites=true&w=majority&appName=loubby-cluster")

# Initialize clients
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

# Sentence Transformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Google Cloud clients
speech_client = speech.SpeechClient()
tts_client = texttospeech.TextToSpeechClient()

# Load Loubby Documentation
def load_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)

LOUBBY_DOCS = load_docx("Loubby.docx")

# Chunk and vectorize documentation
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_text(LOUBBY_DOCS)
chunk_embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
    index.upsert(vectors=[(f"chunk_{i}", embedding, {"text": chunk, "rating_score": 0.0})], namespace="loubby")

# Pydantic models
class ChatRequest(BaseModel):
    query: str
    lang: str = "auto"

class FeedbackRequest(BaseModel):
    rating: int
    comment: str = ""

# Feedback summary
def generate_feedback_summary(limit=5):
    recent_feedback = feedback_collection.find().sort("timestamp", -1).limit(limit)
    summaries = []
    for fb in recent_feedback:
        rating = fb["rating"]
        comment = fb.get("comment", "")
        if rating < 3 and comment:
            summaries.append(f"Critique: '{comment}' (rating: {rating})")
        elif rating >= 4 and comment:
            summaries.append(f"Praise: '{comment}' (rating: {rating})")
    return "Recent feedback: " + "; ".join(summaries) if summaries else "No recent feedback available."

#chunk ratings
def update_chunk_ratings(query: str, rating: int):
    query_embedding = embedding_model.encode([query])[0]
    search_results = index.query(vector=query_embedding.tolist(), top_k=3, include_metadata=True, namespace="loubby")
    for match in search_results["matches"]:
        chunk_id = match["id"]
        current_score = match["metadata"].get("rating_score", 0.0)
        new_score = current_score + (rating - 3) * 0.1
        index.update(id=chunk_id, set_metadata={"rating_score": new_score}, namespace="loubby")
        logger.info(f"Updated chunk {chunk_id} rating to {new_score}")

# Chat endpoint with Google Cloud TTS
@app.post("/chat")
@limiter.limit("10/minute")
async def chat(request: Request, chat_request: ChatRequest):
    try:
        query = chat_request.query
        logger.info(f"Received query: {query}")

        query_lang = chat_request.lang if chat_request.lang != "auto" else detect(query)
        if query_lang != 'en':
            query = query  

        query_embedding = embedding_model.encode([query])[0]
        search_results = index.query(vector=query_embedding.tolist(), top_k=3, include_metadata=True, namespace="loubby")
        context = " ".join([match["metadata"]["text"] for match in sorted(
            search_results["matches"], key=lambda x: x["metadata"].get("rating_score", 0.0), reverse=True)])

        feedback_summary = generate_feedback_summary()
        prompt = f"Context: {context}\nFeedback Summary: {feedback_summary}\nQuery: {query}\nProvide a clear, concise answer."
        response = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7
        ).choices[0].message.content

        if query_lang != 'en':
            synthesis_input = texttospeech.SynthesisInput(text=response)
            voice = texttospeech.VoiceSelectionParams(language_code=query_lang, ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
            audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)
            tts_response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
            audio_content = tts_response.audio_content
            response_dict = {"query": chat_request.query, "answer": response, "lang": query_lang, "audio": audio_content.hex()}
        else:
            response_dict = {"query": chat_request.query, "answer": response, "lang": query_lang}

        logger.info(f"Generated response: {response}")
        return response_dict

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# WebRTC voice/video chat
pcs = set()

class AIAudioTrack(AudioStreamTrack):
    def __init__(self):
        super().__init__()
        self.audio_queue = asyncio.Queue()

    async def recv(self):
        frame = await self.audio_queue.get()
        return frame

@app.post("/offer")
async def offer(request: Request):
    data = await request.json()
    offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
    pc = RTCPeerConnection()
    pcs.add(pc)

    ai_track = AIAudioTrack()

    @pc.on("icecandidate")
    async def on_icecandidate(candidate):
        if candidate and hasattr(request.app, 'websocket') and request.app.websocket:
            await request.app.websocket.send_json({"ice": candidate.to_json()})

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state: {pc.connectionState}")
        if pc.connectionState == "closed":
            pcs.discard(pc)

    await pc.setRemoteDescription(offer)
    pc.addTrack(ai_track)

    async def process_audio():
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        output_stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, output=True, frames_per_buffer=1024)

        while True:
            try:
                audio_data = stream.read(1024, exception_on_overflow=False)
                audio_config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code="en-US"
                )
                audio = speech.RecognitionAudio(content=audio_data)
                response = speech_client.recognize(config=audio_config, audio=audio)

                if response.results:
                    query = response.results[0].alternatives[0].transcript
                    logger.info(f"Recognized query: {query}")

                    query_lang = detect(query)
                    if query_lang != 'en':
                        query = query 

                    query_embedding = embedding_model.encode([query])[0]
                    search_results = index.query(vector=query_embedding.tolist(), top_k=3, include_metadata=True, namespace="loubby")
                    context = " ".join([match["metadata"]["text"] for match in sorted(
                        search_results["matches"], key=lambda x: x["metadata"].get("rating_score", 0.0), reverse=True)])

                    feedback_summary = generate_feedback_summary()
                    prompt = f"Context: {context}\nFeedback Summary: {feedback_summary}\nQuery: {query}\nProvide a clear, concise answer."
                    ai_response = groq_client.chat.completions.create(
                        model="mixtral-8x7b-32768",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=150,
                        temperature=0.7
                    ).choices[0].message.content

                    synthesis_input = texttospeech.SynthesisInput(text=ai_response)
                    voice = texttospeech.VoiceSelectionParams(language_code=query_lang if query_lang != 'en' else "en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
                    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)
                    tts_response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
                    audio_content = tts_response.audio_content

                    await ai_track.audio_queue.put(np.frombuffer(audio_content, dtype=np.int16))
                    output_stream.write(audio_content)

            except Exception as e:
                logger.error(f"Voice chat error: {str(e)}")
                break

        stream.stop_stream()
        stream.close()
        output_stream.stop_stream()
        output_stream.close()
        p.terminate()

    asyncio.create_task(process_audio())
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

# Serve frontend
@app.get("/")
async def get_frontend():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

# WebSocket for signaling
@app.websocket("/ws")
async def websocket_handler(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")
    app.websocket = websocket
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received WebSocket message: {data}")
            # Handle WebSocket messages if needed
            await websocket.send_text("Echo: " + data)  
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        logger.info("WebSocket connection closed")
        app.websocket = None

# Feedback endpoint
@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    try:
        if not 1 <= request.rating <= 5:
            raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
        feedback_data = {
            "rating": request.rating,
            "comment": request.comment,
            "timestamp": datetime.utcnow()
        }
        result = feedback_collection.insert_one(feedback_data)
        update_chunk_ratings(request.comment, request.rating)
        logger.info(f"Feedback stored with ID: {result.inserted_id}")
        return {"message": "Feedback recorded", "feedback_id": str(result.inserted_id)}
    except Exception as e:
        logger.error(f"Feedback error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
