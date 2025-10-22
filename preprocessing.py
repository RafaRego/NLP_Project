import pandas as pd
import html

df = pd.read_csv('Tweets.csv')
# 1. Remove rows with missing or empty text
df = df[df['text'].notna()]
df = df[df['text'].str.strip() != '']

# 2. Decode HTML entities
df['text'] = df['text'].apply(lambda x: html.unescape(x) if isinstance(x, str) else x)

# 3. Remove placeholder/corrupted data
df = df[~df['text'].str.contains('#{5,}', regex=True, na=False)]  # Removes rows with 5+ consecutive #

# 4. Fix encoding issues (if present)
def fix_encoding(text):
    if isinstance(text, str):
        try:
            return text.encode('latin1').decode('utf-8', errors='ignore')
        except:
            return text
    return text

df['text'] = df['text'].apply(fix_encoding)

# 5. Remove excessive whitespace
df['text'] = df['text'].str.strip()
df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)

# 6. (Optional) Handle URLs - choose one approach:
# Option A: Remove completely
df['text'] = df['text'].str.replace(r'https?://\S+', '', regex=True)
# Option B: Replace with token
# df['text'] = df['text'].str.replace(r'https?://\S+', '[URL]', regex=True)

# 7. Remove any remaining empty texts after cleaning
df = df[df['text'].str.strip() != '']

# 8. Tokenize with DistilBERT tokenizer
from transformers import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
encodings = tokenizer(
    df['text'].tolist(),
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors='pt'
)



print(f"Original rows: {len(pd.read_csv('Tweets.csv'))}")
print(f"Cleaned rows: {len(df)}")
print(f"\nSample cleaned texts:")
print(df['text'].head(20))

# View encodings for first 3 lines
for i in range(3):
    print(f"\n{'='*60}")
    print(f"EXAMPLE {i+1}")
    print(f"{'='*60}")
    
    print(f"\nOriginal text:")
    print(df['text'].iloc[i])
    
    print(f"\nInput IDs:")
    print(encodings['input_ids'][i])
    
    print(f"\nAttention Mask:")
    print(encodings['attention_mask'][i])
    
    print(f"\nTokens (as words):")
    print(tokenizer.convert_ids_to_tokens(encodings['input_ids'][i]))
    
    print(f"\nDecoded back to text:")
    print(tokenizer.decode(encodings['input_ids'][i]))
    
    print(f"\nNumber of real tokens: {encodings['attention_mask'][i].sum().item()}")