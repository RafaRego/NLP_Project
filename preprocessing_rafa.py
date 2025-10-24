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

#%% Remove retweets and exact duplicate texts

# Identify retweets using a robust pattern and (if present) the retweet_count column
rt_regex = re.compile(r'(?i)^\s*rt[\s:@]')  # matches "RT ", "rt:", "rt@", etc. at the start
is_retweet = df[text_col].astype(str).str.match(rt_regex)

# Show a few examples of true retweets that will be removed
rt_examples = df.loc[is_retweet, [text_col, "text_clean"]].head(5)
if not rt_examples.empty:
    print(f"\n[Step 3a] Retweets to remove (strict RT-prefix): {is_retweet.sum()} rows")
    print("[Step 3a] Examples of retweets:")
    with pd.option_context("display.max_colwidth", 160):
        print(rt_examples.to_string(index=False))
else:
    print("\n[Step 3a] No retweets detected by the strict RT-prefix rule.")

# Remove retweets
df = df.loc[~is_retweet].copy()

# Detect exact duplicates after cleaning (same text_clean string)
dup_mask = df.duplicated(subset=["text_clean"], keep="first")
n_dups = int(dup_mask.sum())

# Show a few duplicate groups (text and how many times it appears)
if n_dups > 0:
    dup_counts = (
        df.loc[dup_mask | df.duplicated(subset=["text_clean"], keep=False), "text_clean"]
          .value_counts()
    )
    top_dup_groups = dup_counts[dup_counts > 1].head(5)
    print(f"\n[Step 3a] Exact duplicate rows to remove: {n_dups}")
    print("[Step 3a] Examples of duplicate texts and their counts:")
    for txt, cnt in top_dup_groups.items():
        print(f"- ({cnt}×) {txt[:140]}")

    # Show one kept vs one removed for the first duplicate group
    example_text = top_dup_groups.index[0]
    group_rows = df.loc[df["text_clean"] == example_text, [text_col, "text_clean"]].head(2)
    if len(group_rows) == 2:
        print("\n[Step 3a] For the first duplicate group, one kept vs one removed example:")
        with pd.option_context("display.max_colwidth", 160):
            print(group_rows.to_string(index=False))
else:
    print("\n[Step 3a] No exact duplicates detected in text_clean.")

# Drop duplicate rows now (keep the first occurrence)
df = df.loc[~dup_mask].copy()

#%% NOTE TO SEBB ALSO CHECK IF WE ONLY HAVE ENGLISH IN THE DATASET I CHECK THE TWITTER WITH CHATGPT BUT I TRIED THE REVIEW
# ONE THERE AND IT WAS TOO HEAVY. BETTER TO DO IT IN CODE.
#%%
# Binary mapping: negative -> 0, positive -> 1
label_map = {"negative": 0, "positive": 1}
df["label_bin"] = labels_norm.loc[df.index].map(label_map).astype(int)


#%% Save cleaned dataset to CSV
df.to_csv("tweets_clean.csv", index=False)