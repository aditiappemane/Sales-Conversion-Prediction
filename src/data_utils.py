"""
Data loading and preprocessing utilities for sales call transcripts.
"""
import os
import pandas as pd
from typing import Tuple
from .config import DATA_DIR

def load_data(filename: str) -> pd.DataFrame:
    """Load sales call data from a CSV file."""
    path = os.path.join(DATA_DIR, filename)
    return pd.read_csv(path)

def preprocess_transcripts(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess transcripts (cleaning, lowercasing, etc.)."""
    # TODO: Implement cleaning steps
    df['transcript'] = df['transcript'].str.lower()
    return df 