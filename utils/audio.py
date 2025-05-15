from vosk import Model, KaldiRecognizer
import wave
import os
import tempfile
import subprocess
import numpy as np
import pyaudio

# Language model mapping for Vosk
LANGUAGE_MODELS = {
    "en-US": "models/vosk/vosk-model-small-en-us-0.15",
    "es-ES": "models/vosk/vosk-model-small-es-0.42",
    "fr-FR": "models/vosk/vosk-model-small-fr-0.22",
    # Add more from https://alphacephei.com/vosk/models
}

def voice_to_text(audio_data=None, language="en-US"):
    """Convert audio to text using Vosk."""
    model_path = LANGUAGE_MODELS.get(language, LANGUAGE_MODELS["en-US"])  # Fallback to English
    if not os.path.exists(model_path):
        raise ValueError(f"Vosk model for {language} not found at {model_path}")

    model = Model(model_path)
    rec = KaldiRecognizer(model, 16000)  # 16kHz sample rate

    if audio_data:  # From WebSocket/stream
        rec.AcceptWaveform(audio_data)
        result = rec.Result()
        import json
        return json.loads(result).get("text", "No speech detected")
    else:  # From microphone
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        print("Listening...")
        audio = b""
        for _ in range(5 * RATE // CHUNK):  # 5 seconds
            audio += stream.read(CHUNK)
        stream.stop_stream()
        stream.close()
        p.terminate()
        rec.AcceptWaveform(audio)
        result = rec.Result()
        return json.loads(result).get("text", "No speech detected")

def text_to_speech(text, language="en-US"):
    lang_code = language.split("-")[0]
    print(f"Speaking '{text}' in language: {lang_code}")  # Debug
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio_path = temp_audio.name
        subprocess.run([
            "espeak", "-v", lang_code, text, "-w", temp_audio_path
        ], check=True)
    return temp_audio_path