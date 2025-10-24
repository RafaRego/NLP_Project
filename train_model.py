
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from datasets import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

# Load the preprocessed data produced by preprocessing.py
# Expected columns: text_clean (str), label_bin (int: 0=negative, 1=positive)
df = pd.read_csv("tweets_clean.csv")
required_cols = {"text_clean", "label_bin"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in tweets_clean.csv: {sorted(missing)}")

# Drop empty texts to avoid tokenization issues
df = df[df["text_clean"].astype(str).str.strip() != ""].copy()
print(f"Loaded {len(df)} rows for training.")
print(df["label_bin"].value_counts().rename(index={0: 'neg(0)', 1: 'pos(1)'}).to_string())

# Stratified train/validation split (80/20)
train_df, val_df = train_test_split(
    df[["text_clean", "label_bin"]],
    test_size=0.20,
    stratify=df["label_bin"],
    random_state=42,
)
print(f"Train size: {len(train_df)} | Val size: {len(val_df)}")

# Convert to Hugging Face Datasets
train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds   = Dataset.from_pandas(val_df.reset_index(drop=True))

# Tokenizer and tokenization function (DistilBERT, as in the lab)
MODEL_NAME = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

def tokenize_batch(batch):
    return tokenizer(
        batch["text_clean"],
        truncation=True,
        padding=True,
        max_length=256,
    )

train_ds = train_ds.map(tokenize_batch, batched=True)
val_ds   = val_ds.map(tokenize_batch,   batched=True)

# Trainer expects the target column to be named "labels"
train_ds = train_ds.rename_column("label_bin", "labels")
val_ds   = val_ds.rename_column("label_bin", "labels")

# Keep only tensors needed for training
cols = ["input_ids", "attention_mask", "labels"]
train_ds.set_format(type="torch", columns=cols)
val_ds.set_format(type="torch", columns=cols)

# D) Model (DistilBERT for sequence classification with 2 labels)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# E) Metrics function (accuracy, precision, recall, F1)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

# F) Training configuration (lab-style, simple and reliable)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="clf_out",
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=24,     # if OOM, drop to 24 or 16
    per_device_eval_batch_size=32,
    weight_decay=0.0,
    eval_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="no",
    load_best_model_at_end=False,
    metric_for_best_model="f1",
    greater_is_better=True,
    seed=42,
    data_seed=42,
    report_to=[],
    fp16=True,                          # use mixed precision on NVIDIA
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# G) Train, evaluate, and save
trainer.train()
eval_metrics = trainer.evaluate()
print("Validation metrics:", {k: round(v, 4) for k, v in eval_metrics.items() if isinstance(v, float)})

save_dir = "sentiment-model-distilbert"
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)

with open(f"{save_dir}/val_metrics.json", "w", encoding="utf-8") as f:
    json.dump(eval_metrics, f, indent=2)

print(f"Model and tokenizer saved to: {save_dir}")
