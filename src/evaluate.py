"""
Evaluation script: compare baseline vs. fine-tuned embeddings for conversion prediction.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import joblib
# Load data
DATA_PATH = "data/sales_calls.csv"
df = pd.read_csv(DATA_PATH).dropna(subset=["transcript", "label"])
df["label"] = df["label"].astype(int)

# Split data
X = df["transcript"].tolist()
y = df["label"].tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load models
baseline_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
fine_tuned_model = SentenceTransformer("models/fine-tuned-sales-embedding")

# Helper to get embeddings
def get_embeddings(model, texts):
    return [model.encode(text) for text in tqdm(texts)]

# Baseline
print("\n--- Baseline Embeddings ---")
X_train_base = get_embeddings(baseline_model, X_train)
X_test_base = get_embeddings(baseline_model, X_test)
clf_base = LogisticRegression(max_iter=1000)
clf_base.fit(X_train_base, y_train)
y_pred_base = clf_base.predict(X_test_base)
print(classification_report(y_test, y_pred_base))

# Fine-tuned
print("\n--- Fine-Tuned Embeddings ---")
X_train_ft = get_embeddings(fine_tuned_model, X_train)
X_test_ft = get_embeddings(fine_tuned_model, X_test)
clf_ft = LogisticRegression(max_iter=1000)
clf_ft.fit(X_train_ft, y_train)
y_pred_ft = clf_ft.predict(X_test_ft)
print(classification_report(y_test, y_pred_ft))

# Save fine-tuned classifier for API
joblib.dump(clf_ft, "models/fine-tuned-logreg.joblib") 