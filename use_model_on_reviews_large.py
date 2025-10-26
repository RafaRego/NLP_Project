# test_model_on_large_dataset.py
# Purpose: load the fine-tuned DistilBERT model and score the large cleaned dataset.
# Input : large_dataset_airline_clean.csv with a 'text_clean' column
# Output: large_dataset_scored.csv with pred_label (0/1) and pred_prob_pos

import numpy as np
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Fixed settings for simplicity
MODEL_DIR   = "./sentiment-model-distilbert"
INPUT_CSV   = "large_dataset_airline_clean.csv"
OUTPUT_CSV  = "large_dataset_scored.csv"
MAX_LENGTH  = 256
BATCH_SIZE  = 64

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

# Load data (expects a 'text_clean' column)
df = pd.read_csv(INPUT_CSV)
if "text_clean" not in df.columns:
    raise ValueError("Input file must contain a 'text_clean' column.")
texts = df["text_clean"].astype(str).tolist()

# Load model and tokenizer from the saved folder
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)

# Device and eval mode
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

preds = []
probs = []
use_amp = (device == "cuda")

# Process in batches
print(f"Processing {len(texts)} reviews on {device}...")
for i in range(0, len(texts), BATCH_SIZE):
    batch_texts = texts[i:i + BATCH_SIZE]
    
    # Tokenize batch
    encoded = tokenizer(
        batch_texts,
        max_length=MAX_LENGTH,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    # Move to device
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    
    # Get predictions
    with torch.no_grad():
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Convert logits to probabilities
    logits = outputs.logits.cpu().numpy()
    batch_probs = softmax(logits)
    batch_preds = np.argmax(batch_probs, axis=1)
    
    preds.extend(batch_preds)
    probs.extend(batch_probs[:, 1])  # Probability of positive class
    
    if (i // BATCH_SIZE + 1) % 10 == 0:
        print(f"Processed {i + len(batch_texts)}/{len(texts)} reviews...")

# Add predictions to dataframe
df["pred_label"] = preds
df["pred_prob_pos"] = probs

# Save results
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nDone! Results saved to {OUTPUT_CSV}")
print(f"Predicted positive: {sum(preds)} / {len(preds)} ({100*sum(preds)/len(preds):.1f}%)")