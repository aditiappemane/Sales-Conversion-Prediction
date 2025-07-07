"""
Deployment script: serve the prediction system via API.
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import numpy as np
import joblib

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load fine-tuned model and classifier at startup
embedding_model = SentenceTransformer("models/fine-tuned-sales-embedding")
try:
    clf = joblib.load("models/fine-tuned-logreg.joblib")
except Exception:
    clf = None

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    transcript = data.get("transcript", "")
    if not transcript:
        return {"error": "No transcript provided."}
    embedding = embedding_model.encode(transcript)
    if clf is not None:
        proba = clf.predict_proba([embedding])[0][1]
        return {"probability": float(proba)}
    else:
        return {"error": "Classifier not trained. Please run evaluation pipeline to train and save the classifier."} 