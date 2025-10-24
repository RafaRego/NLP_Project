# preprocessing.py
# Step 1: Load the Twitter US Airline Sentiment CSV and inspect columns and label distribution.

import pandas as pd
import re
import html


#%%
# Load CSV (expects Tweets.csv in the working directory)
df = pd.read_csv("Tweets.csv")

# Basic shape and columns
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))

# Expected columns in the classic dataset
label_col = "airline_sentiment"
text_col  = "text"

missing = [c for c in [label_col, text_col] if c not in df.columns]
if missing:
    raise ValueError(f"Expected columns not found: {missing}")

# Preview a few rows (text truncated for readability)
with pd.option_context("display.max_colwidth", 120):
    print("\nHead (first 5 rows):")
    print(df[[text_col, label_col]].head(5))

# Null counts for a quick hygiene check
print("\nNull values (top 15 columns):")
print(df.isna().sum().sort_values(ascending=False).head(15))

# Label distribution (positive / negative / neutral)
labels = df[label_col].astype(str).str.lower()
vc = labels.value_counts(dropna=False)
total = len(labels)

print("\nClass distribution:")
dist = (
    vc.rename_axis("label")
      .reset_index(name="count")
      .assign(share=lambda x: (100 * x["count"] / total).round(2).astype(str) + "%")
)
print(dist.to_string(index=False))

# A couple of short examples per class (if present)
for cls in ["positive", "negative", "neutral"]:
    subset = df.loc[labels == cls, text_col].head(3)
    if not subset.empty:
        print(f"\nExamples for class = {cls}:")
        for s in subset.tolist():
            print(f"- {s[:160]}")

#%% Step 2: Preprocess tweet text (minimal, BERT-friendly)
# Goal: lowercase, remove URLs and @handles, drop '#' while keeping the hashtag word,
# collapse extra whitespace, and unescape simple HTML entities.

text_raw = df[text_col].astype(str)

# Patterns for simple, robust cleanup
url_re    = re.compile(r"https?://\S+")
handle_re = re.compile(r"@\w+")
spaces_re = re.compile(r"\s+")

def clean_tweet(t: str) -> str:
    t = html.unescape(t)                 # convert &amp; -> &, etc.
    t = url_re.sub("", t)                # remove URLs
    t = handle_re.sub("", t)             # remove @handles
    t = t.replace("#", "")               # keep hashtag words but drop '#'
    t = t.lower()                        # lowercase
    t = spaces_re.sub(" ", t).strip()    # collapse spaces
    return t

df["text_clean"] = text_raw.map(clean_tweet)

#%% Step 3: Keep only positive/negative and encode labels to {0,1}

# Normalize label strings
labels_norm = df[label_col].astype(str).str.lower()

# Filter to polarized tweets only
mask_polarized = labels_norm.isin(["positive", "negative"])
df = df.loc[mask_polarized].copy()

# Ensure cleaned text exists and is non-empty after cleaning
df = df[df["text_clean"].astype(str).str.strip() != ""]

# Binary mapping: negative -> 0, positive -> 1
label_map = {"negative": 0, "positive": 1}
df["label_bin"] = labels_norm.loc[df.index].map(label_map).astype(int)

