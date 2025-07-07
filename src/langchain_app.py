"""
LangChain orchestration for embedding generation and prediction.
"""
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from sentence_transformers import SentenceTransformer
import joblib

# Load fine-tuned model and classifier
embedding_model = SentenceTransformer("models/fine-tuned-sales-embedding")
clf = joblib.load("models/fine-tuned-logreg.joblib")

# Example LangChain workflow (pseudo-chain)
def predict_conversion(transcript: str) -> float:
    embedding = embedding_model.encode(transcript)
    proba = clf.predict_proba([embedding])[0][1]
    return float(proba)

if __name__ == "__main__":
    transcript = input("Paste sales call transcript: ")
    prob = predict_conversion(transcript)
    print(f"Predicted conversion probability: {prob:.2%}") 