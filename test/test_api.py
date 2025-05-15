from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_chat():
    response = client.post("/chat", json={"text": "How do I apply?"})
    assert response.status_code == 200
    assert "response" in response.json()
    
    
import os
print(os.path.exists("lib/Wav2Lip/inference.py"))
print(os.path.exists("lib/Wav2Lip/checkpoints/wav2lip.pth"))
print(os.path.exists("static/avatar.mp4"))