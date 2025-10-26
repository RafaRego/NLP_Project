# preprocessing_large_dataset.py
# Preprocess large_dataset_airline.csv with 128k observations

import pandas as pd
import re
import html
from langdetect import detect, LangDetectException

#%% Load the large dataset
df = pd.read_csv("large_dataset_airline.csv")

print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))

# Check if we have the required columns
text_col = "text"

if text_col not in df.columns:
    raise ValueError(f"Column '{text_col}' not found in dataset")

print(f"\nFirst 5 rows of text column:")
with pd.option_context("display.max_colwidth", 120):
    print(df[[text_col]].head(5))

# Check for null values
print("\nNull values:")
print(df[[text_col]].isna().sum())

#%% Step 2: Preprocess text (same as tweet preprocessing)

text_raw = df[text_col].astype(str)

# Patterns for cleanup
url_re    = re.compile(r"https?://\S+")
handle_re = re.compile(r"@\w+")
spaces_re = re.compile(r"\s+")

def clean_text(t: str) -> str:
    t = html.unescape(t)                 # convert &amp; -> &, etc.
    t = url_re.sub("", t)                # remove URLs
    t = handle_re.sub("", t)             # remove @handles
    t = t.replace("#", "")               # keep hashtag words but drop '#'
    t = t.lower()                        # lowercase
    t = spaces_re.sub(" ", t).strip()    # collapse spaces
    return t

print("\n[Step 2] Cleaning text...")
df["text_clean"] = text_raw.map(clean_text)

# Show before/after examples
print("\nBefore and after cleaning (first 3 examples):")
for i in range(min(3, len(df))):
    print(f"\nOriginal: {df.iloc[i][text_col][:150]}")
    print(f"Cleaned:  {df.iloc[i]['text_clean'][:150]}")

#%% Step 3: Remove empty texts after cleaning
initial_count = len(df)
df = df[df["text_clean"].astype(str).str.strip() != ""].copy()
removed_empty = initial_count - len(df)
print(f"\n[Step 3] Removed {removed_empty} rows with empty text after cleaning")

#%% Step 4: Remove exact duplicates

# Detect exact duplicates after cleaning
dup_mask = df.duplicated(subset=["text_clean"], keep="first")
n_dups = int(dup_mask.sum())

if n_dups > 0:
    dup_counts = (
        df.loc[dup_mask | df.duplicated(subset=["text_clean"], keep=False), "text_clean"]
          .value_counts()
    )
    top_dup_groups = dup_counts[dup_counts > 1].head(5)
    print(f"\n[Step 4] Exact duplicate rows to remove: {n_dups}")
    print("[Step 4] Top 5 most duplicated texts:")
    for txt, cnt in top_dup_groups.items():
        print(f"- ({cnt}×) {txt[:100]}")
else:
    print("\n[Step 4] No exact duplicates detected in text_clean.")

# Drop duplicates
df = df.loc[~dup_mask].copy()
print(f"Rows after removing duplicates: {len(df)}")

#%% Step 5: Language detection - Remove non-English reviews

print("\n[Step 5] Detecting and removing non-English reviews...")
print("This may take a few minutes for large datasets...")

def detect_language(text):
    """Detect language of text, return 'en' for English or other language code"""
    try:
        # Only try detection if text is long enough
        if len(str(text).strip()) < 10:
            return 'unknown'
        return detect(str(text))
    except LangDetectException:
        return 'unknown'
    except Exception:
        return 'unknown'

# Detect language for all reviews
df['language'] = df['text_clean'].apply(detect_language)

# Show language distribution
print("\nLanguage distribution (top 10):")
lang_dist = df['language'].value_counts().head(10)
for lang, count in lang_dist.items():
    print(f"{lang}: {count} ({100*count/len(df):.2f}%)")

# Keep only English reviews
initial = len(df)
df = df[df['language'] == 'en'].copy()
removed = initial - len(df)
print(f"\nRemoved {removed} non-English reviews ({100*removed/initial:.2f}%)")
print(f"Remaining rows: {len(df)}")

# Show examples of removed non-English reviews (before filtering)
if removed > 0:
    print("\nExamples of detected non-English text (these were removed):")
    non_english_samples = df[df['language'] != 'en']['text_clean'].head(3)
    for text in non_english_samples:
        print(f"- {text[:120]}")

# Drop the language column (no longer needed)
df = df.drop(columns=['language'])

#%% Step 6: Handle labels (Recommended column with capital R)

if 'Recommended' in df.columns:
    print("\n[Step 6] Processing 'Recommended' labels...")
    
    # Check what values are in the column
    print("\nUnique values in Recommended column:")
    rec_values = df['Recommended'].value_counts(dropna=False)
    print(rec_values)
    
    # Map recommended to binary labels (handle various formats)
    df['label_bin'] = df['Recommended'].map({
        'yes': 1, 'no': 0,
        'Yes': 1, 'No': 0,
        'YES': 1, 'NO': 0,
        1: 1, 0: 0,
        True: 1, False: 0
    })
    
    # Remove rows with missing labels
    initial = len(df)
    df = df.dropna(subset=['label_bin']).copy()
    df['label_bin'] = df['label_bin'].astype(int)
    removed = initial - len(df)
    print(f"\nRemoved {removed} rows with missing/invalid labels")
    
    # Show label distribution
    label_dist = df['label_bin'].value_counts()
    print("\nLabel distribution:")
    print(f"Not Recommended (0): {label_dist.get(0, 0)} ({100*label_dist.get(0, 0)/len(df):.1f}%)")
    print(f"Recommended (1): {label_dist.get(1, 0)} ({100*label_dist.get(1, 0)/len(df):.1f}%)")
else:
    print("\n[Step 6] No 'Recommended' column found - skipping label processing")

#%% Step 7: Final statistics and save

print(f"\n{'='*60}")
print("FINAL DATASET STATISTICS")
print(f"{'='*60}")
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Text length statistics
df['text_length'] = df['text_clean'].str.len()
print(f"\nText length statistics:")
print(f"Mean: {df['text_length'].mean():.1f} characters")
print(f"Median: {df['text_length'].median():.1f} characters")
print(f"Min: {df['text_length'].min()} characters")
print(f"Max: {df['text_length'].max()} characters")

# Drop temporary text_length column
df = df.drop(columns=['text_length'])

# Save cleaned dataset
output_file = "large_dataset_airline_clean.csv"
df.to_csv(output_file, index=False)
print(f"\n✓ Cleaned dataset saved to: {output_file}")
print(f"{'='*60}")