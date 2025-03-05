# Loubby Navigation Assistant

A two-way voice/video AI assistant built with FastAPI, WebRTC, and Google Cloud APIs, featuring a realistic animated avatar.

## Features
- **Text Chat:** Query via `/chat` endpoint with multilingual support (e.g., French).
- **Voice/Video Chat:** Real-time audio interaction at `http://localhost:8001/` with an animated AI avatar.
- **Feedback:** Submit ratings and comments via `/feedback`.
- **Health Check:** Verify server status at `/health`.

## Prerequisites
- Python 3.9+
- Conda (for environment management)
- Google Cloud account with billing enabled (Speech-to-Text, Text-to-Speech APIs)
- API keys: Grok (xAI), Pinecone, MongoDB Atlas

## Setup
1. **Clone Repository:**
   ```bash
   git clone <repository-url>
   cd loubby-navigation-assistant
cd loubby-navigation-assistant

2. **Create Conda Environment:
``bash

conda create -n loubby python=3.9
conda activate loubby

3. **Install Dependencies:
``bash

conda install portaudio
pip install -r requirements.txt --no-cache-dir

4. **Configure Environment Variables:
Create .env:

GROQ_API_KEY=your_grok_api_key
PINECONE_API_KEY=your_pinecone_api_key
MONGO_URI=mongodb+srv://loubby:<your_mongo_password>@loubby-cluster.oolrx.mongodb.net/?retryWrites=true&w=majority&appName=loubby-cluster
GOOGLE_APPLICATION_CREDENTIALS=/path/to/loubby-9f0a887723e7.json
#Download Google Cloud service account key and place it at the specified path.
 
5. **Enable Google Cloud APIs:
Enable Speech-to-Text and Text-to-Speech APIs in Google Cloud Console.
Add billing card for full functionality.

6. **Place Avatar:
Save ai_avatar.png in static/ directory.


##Running the Application
1. **Start Server:
``bash

python main.py

#Server runs on http://0.0.0.0:8001.

2. **Access Frontend:
Open http://localhost:8001/ in a browser.
Allow microphone access, speak to interact with the AI.

3. **Test API:
Swagger: http://localhost:8001/docs.
Example /chat request:
json

{"query": "what is loubby?", "lang": "fr"}   
