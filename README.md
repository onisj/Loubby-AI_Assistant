# Loubby Navigation Assistant

# Loubby Navigation Assistant

A two-way voice/video AI assistant built with FastAPI, WebRTC, and Google Cloud APIs, featuring a realistic animated avatar via a pre-recorded video clip with support for over 100 languages.

## Features
- **Text Chat:** Query via `/chat` endpoint with support for over 100 languages (e.g., English, Igbo, German, Japanese, Hindi, Arabic).
- **Voice/Video Chat:** Real-time audio interaction at `http://localhost:8001/` with a pre-recorded animated avatar (lip-sync from clip, live multilingual responses).
- **Feedback:** Submit ratings and comments via `/feedback`.
- **Health Check:** Verify server status at `/health`.

## Prerequisites
- Python 3.9+
- Conda (for environment management)
- Google Cloud account with billing enabled (Speech-to-Text, Text-to-Speech, Translate APIs)
- API keys: Grok (xAI), Pinecone, MongoDB Atlas
- `avatar_clip.mp4` (short realistic talking video)

## Setup
1. **Clone Repository:**
   ```bash
   git clone <repository-url>
   cd loubby-navigation-assistant
<<<<<<< HEAD

2. **Create Conda Environment:**
   ```bash
   conda create -n loubby python=3.9
   conda activate loubby

3. **Install Dependencies:**
   ```bash
   conda install portaudio
   pip install -r requirements.txt --no-cache-dir
   pip install websockets --no-cache-dir
   pip install google-cloud-speech google-cloud-texttospeech google-cloud-translate --no-cache-dir

4. **Configure Environment Variables:**
   Create .env in loubby-navigation-assistant/:
   ```plaintext
   GROQ_API_KEY=your_grok_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   MONGO_URI=mongodb+srv://loubby:<your_mongo_password>@loubby-cluster.oolrx.mongodb.net/?retryWrites=true&w=majority&appName=loubby-cluster
   GOOGLE_APPLICATION_CREDENTIALS=/Users/macmac/Documents/loubby-navigation-assistant/loubby-name.json

## Download Google Cloud service account key from Google Cloud Console.
5. **Enable Google Cloud APIs and Billing:**
   Enable Speech-to-Text, Text-to-Speech, and Translate APIs in Google Cloud Console.
   Add a billing card at Billing for project name—required for audio and translation.

6. **Place Video Clip:**
   Save avatar_clip.mp4 in static/:
   ```bash
   mkdir -p static
   mv avatar_clip.mp4 static/

## To prepare the video clip, you can use FFmpeg to resize and format:
   ```bash
  ffmpeg -i input.mp4 -vf "scale=200:200" -c:v libx264 -c:a aac static/avatar_clip.mp4

## Running the Application

1. Start Server:
   ```bash
   python main.py
## Server runs on http://0.0.0.0:8001.

2. Access Frontend:
   Open http://localhost:8001/ in a browser.
   Allow microphone access, speak or type in any supported language to interact.

3. Test API:
   Swagger: http://localhost:8001/docs.
   Example /chat requests:
   ```json
   {"query": "kedụ ihe bụ Loubby?", "lang": "auto"}  # Igbo
   {"query": "Was ist Loubby?", "lang": "auto"}      # German
   {"query": "ルービーとは？", "lang": "auto"}       # Japanese
   {"query": "क्या है लौबी?", "lang": "auto"}        # Hindi
   {"query": "ما هو لوبي؟", "lang": "auto"}         # Arabic   
