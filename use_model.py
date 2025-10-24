# use_model_on_reviews_min.py
# Purpose: load the fine-tuned DistilBERT model and score a new dataset.
# Input : skytrax_subset.csv with a 'text' column
# Output: skytrax_subset_scored.csv with pred_label (0/1) and pred_prob_pos

import numpy as np
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Fixed settings for simplicity
MODEL_DIR   = "sentiment-model-distilbert"
INPUT_CSV   = "skytrax_subset.csv"
OUTPUT_CSV  = "skytrax_subset_scored.csv"
MAX_LENGTH  = 256
BATCH_SIZE  = 64

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

# Load data (expects a 'text' column)
df = pd.read_csv(INPUT_CSV)
if "text" not in df.columns:
    raise ValueError("Input file must contain a 'text' column.")
texts = df["text"].astype(str).tolist()

# Load model and tokenizer from the saved folder
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)

# Device and eval mode
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

preds = []
probs = []
use_amp = (device == "cuda")

