"""
Baseline pipeline: generic embeddings + classifier for conversion prediction.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from .data_utils import load_data, preprocess_transcripts
from .embedding import get_gemini_embedding

def main():
    df = load_data('sales_calls.csv')
    df = preprocess_transcripts(df)
    print('Loaded', len(df), 'records')

    # Generate embeddings
    print('Generating embeddings...')
    df['embedding'] = df['transcript'].apply(get_gemini_embedding)
    X = list(df['embedding'])
    y = df['conversion'].astype(int)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Evaluate
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main() 