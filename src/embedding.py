"""
Embedding generation using Gemini API (REST) or mock fallback.
"""
import os
import requests
from .config import GEMINI_API_KEY

GEMINI_EMBEDDING_URL = "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent"

USE_MOCK = os.getenv("USE_MOCK_EMBEDDING", "true").lower() == "true"


def get_gemini_embedding(text: str, use_mock: bool = None) -> list:
    """
    Get embedding for a text using Gemini Embeddings REST API or mock.
    Set use_mock=True to force mock, False to force real, or None to use config/env.
    """
    if use_mock is None:
        use_mock = USE_MOCK or not GEMINI_API_KEY
    if use_mock:
        # MOCK: Return a fixed-size vector for testing
        return [0.1] * 768
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    data = {
        "model": "models/embedding-001",
        "content": {"parts": [{"text": text}]}
    }
    response = requests.post(GEMINI_EMBEDDING_URL, headers=headers, params=params, json=data)
    response.raise_for_status()
    embedding = response.json()["embedding"]["values"]
    return embedding 