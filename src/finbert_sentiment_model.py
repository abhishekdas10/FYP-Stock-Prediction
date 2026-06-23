import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import pipeline

# Load dataset
df = pd.read_csv("data/raw/financial_sentiment.csv")

df = df.rename(columns={
    "Sentence": "text",
    "Sentiment": "sentiment"
})

df = df.dropna(subset=["text", "sentiment"])

# Use smaller test sample first because FinBERT can be slow on Mac
_, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment"]
)

# Optional: start with 300 rows for speed
test_df = test_df.sample(n=300, random_state=42)

print("Testing FinBERT on:", len(test_df), "rows")

finbert = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert"
)

texts = test_df["text"].tolist()
true_labels = test_df["sentiment"].str.lower().tolist()

predictions = []

for i, text in enumerate(texts):
    result = finbert(text[:512])[0]
    label = result["label"].lower()

    predictions.append(label)

    if (i + 1) % 50 == 0:
        print(f"Processed {i + 1}/{len(texts)}")

print("\nAccuracy:", accuracy_score(true_labels, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, predictions, labels=["negative", "neutral", "positive"]))

print("\nClassification Report:")
print(classification_report(true_labels, predictions))