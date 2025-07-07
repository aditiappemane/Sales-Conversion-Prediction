"""
Contrastive fine-tuning for sales call embeddings using Triplet Loss.
"""
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sentence_transformers import SentenceTransformer, losses, InputExample
from tqdm import tqdm

# ✅ Dataset: anchor = current call, positive = same label, negative = opposite label
class SalesContrastiveDataset(Dataset):
    def __init__(self, df, model):
        self.df = df.reset_index(drop=True)
        self.model = model
        self.label_groups = df.groupby("label").groups  # {0: [idxs], 1: [idxs]}
        self.samples = []

        for idx, row in df.iterrows():
            label = row["label"]
            opposite_label = 1 - label
            anchor = row["transcript"]

            pos_idxs = self.label_groups[label]
            neg_idxs = self.label_groups[opposite_label]

            # Skip if not enough samples
            if len(pos_idxs) < 2 or len(neg_idxs) < 1:
                continue

            # Get positive (not same as anchor)
            pos_idx = next(i for i in pos_idxs if i != idx)
            pos = df.loc[pos_idx, "transcript"]

            # Get random negative
            neg_idx = neg_idxs[0]
            neg = df.loc[neg_idx, "transcript"]

            self.samples.append((anchor, pos, neg))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        a, p, n = self.samples[idx]
        return InputExample(texts=[a, p, n])

# ✅ Training loop
def train_contrastive():
    # Load data
    df = pd.read_csv("data/sales_calls.csv")  # Must contain 'transcript' and 'label' columns
    df = df.dropna(subset=["transcript", "label"])
    df["label"] = df["label"].astype(int)

    # Load base model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Create dataset
    dataset = SalesContrastiveDataset(df, model)
    train_dataloader = DataLoader(dataset, shuffle=True, batch_size=16)

    # Define loss
    train_loss = losses.TripletLoss(model)

    # Train
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=10,
        show_progress_bar=True,
        output_path="models/fine-tuned-sales-embedding"
    )

if __name__ == '__main__':
    train_contrastive()
