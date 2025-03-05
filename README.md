# Loubby Navigation Assistant

AI-powered backend to assist candidates in navigating the Loubby platform.

## Setup
1. **Prerequisites:**
   - Docker installed
   - MongoDB running: `docker run -d -p 27017:27017 --name mongo-container mongo`
   - (Optional) Jitsi Meet: See `docker-jitsi-meet` repo

2. **Build and Run:**
   ```bash
   docker build -t loubby-navigation-assistant .
   docker run -d -p 8000:8000 --env-file .env --link mongo-container:mongo loubby-navigation-assistant