from fastapi import APIRouter, HTTPException, WebSocket
from utils.embeddings import get_embedding
from utils.retrieval import retrieve_docs
from utils.generation import generate_response
from utils.audio import voice_to_text, text_to_speech
from api.models import Query, Feedback
from fastapi.responses import FileResponse
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import VideoFrame
import os
import asyncio
import cv2
import subprocess

router = APIRouter()
pcs = set()

@router.post("/chat")
async def chat(query: Query, language: str = "en-US"):
    try:
        docs = retrieve_docs(query.text)
        response = generate_response(query.text, docs, language)
        return {"response": response, "language": language}
    except Exception as e:
        print(f"Error in /chat: {str(e)}")  # Debug
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")
@router.get("/voice")
async def voice_input(language: str = "en-US"):
    try:
        text = voice_to_text(language=language)
        docs = retrieve_docs(text)
        response = generate_response(text, docs, language)
        return {"response": response, "language": language}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing voice input: {str(e)}")

@router.get("/tts/{response_text}")
async def tts(response_text: str, language: str = "en-US"):
    try:
        audio_file = text_to_speech(response_text, language=language)
        if not os.path.exists(audio_file):
            raise HTTPException(status_code=500, detail="Generated audio file not found.")
        response = FileResponse(audio_file, media_type="audio/wav", filename="response.wav")
        os.remove(audio_file)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating TTS audio: {str(e)}")

@router.post("/feedback")
async def submit_feedback(feedback: Feedback):
    return {"status": "Feedback recorded"}

@router.websocket("/audio_stream")
async def audio_stream(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_json()
        audio_base64 = data.get("audio")
        language = data.get("language", "en-US")
        print(f"Received audio for language: {language}")  # Debug
        import base64
        audio_data = base64.b64decode(audio_base64)
        print(f"Decoded audio length: {len(audio_data)} bytes")  # Debug
        text = voice_to_text(audio_data, language)
        print(f"Recognized text: {text}")  # Debug
        docs = retrieve_docs(text)
        response = generate_response(text, docs, language)
        audio_response = text_to_speech(response, language)
        with open(audio_response, "rb") as f:
            await websocket.send_bytes(f.read())
        os.remove(audio_response)

class AIVideoResponseTrack(VideoStreamTrack):
    def __init__(self, query=None, language="en-US"):
        super().__init__()
        self.cap = None
        self.audio_path = None
        self.language = language
        if query:
            self.generate_response(query)

    def generate_response(self, query):
        docs = retrieve_docs(query)
        text_response = generate_response(query, docs, self.language)
        self.audio_path = text_to_speech(text_response, self.language)
        output_video = os.path.join(os.path.dirname(__file__), "../../static/output_video.mp4")
        wav2lip_path = os.path.join(os.path.dirname(__file__), "../lib/Wav2Lip/inference.py")
        checkpoint_path = os.path.join(os.path.dirname(__file__), "../lib/Wav2Lip/checkpoints/wav2lip.pth")
        face_path = os.path.join(os.path.dirname(__file__), "../../static/avatar.mp4")
        subprocess.run([
            "python", wav2lip_path,
            "--checkpoint_path", checkpoint_path,
            "--face", face_path,
            "--audio", self.audio_path,
            "--outfile", output_video
        ], check=True)
        self.cap = cv2.VideoCapture(output_video)

    async def recv(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            if ret:
                await asyncio.sleep(0.033)
                return VideoFrame.from_ndarray(frame, format="bgr24")
        return None

@router.websocket("/video_stream")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("track")
    def on_track(track):
        print(f"Received {track.kind} track")

    data = await websocket.receive_json()
    query = data.get("query", "Hello")
    language = data.get("language", "en-US")
    ai_video_track = AIVideoResponseTrack(query=query, language=language)
    pc.addTrack(ai_video_track)

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    await websocket.send_json({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

    data = await websocket.receive_json()
    await pc.setRemoteDescription(RTCSessionDescription(**data))

    try:
        while True:
            await asyncio.sleep(1)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        pcs.remove(pc)
        await pc.close()