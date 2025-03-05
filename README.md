# Loubby Navigation Assistant

A two-way voice/video AI assistant built with FastAPI, WebRTC, and Google Cloud APIs, featuring a realistic animated avatar with live communication effects.

## Features
- **Text Chat:** Query via `/chat` endpoint with multilingual support (e.g., French).
- **Voice/Video Chat:** Real-time audio interaction at `http://localhost:8001/` with an animated AI avatar (head tilt and eye movement when speaking, breathing when idle).
- **Feedback:** Submit ratings and comments via `/feedback`.
- **Health Check:** Verify server status at `/health`.

## Prerequisites
- Python 3.9+
- Conda (for environment management)
- Google Cloud account with billing enabled (Speech-to-Text, Text-to-Speech APIs)
- API keys: Grok (xAI), Pinecone, MongoDB Atlas
- `ai_avatar.png` (e.g., from Vecteezy, preferably with a face for animation)

## Setup
1. **Clone Repository:**
   ```bash
   git clone <repository-url>
   cd loubby-navigation-assistant
