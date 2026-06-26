import os
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

INPUT_FILE = "data/raw/financial_news.csv"
OUTPUT_FILE = "data/processed/news_sentiment.csv"

TEXT_COLUMN = "Combined_News"
DATE_COLUMN = "Date"

BATCH_SIZE = 16

os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv(INPUT_FILE)

df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
df = df.dropna(subset=[DATE_COLUMN, TEXT_COLUMN])

df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)

print("News rows loaded:", len(df))
print("Date range:", df[DATE_COLUMN].min(), "to", df[DATE_COLUMN].max())

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

labels = ["positive", "negative", "neutral"]

results = []

texts = df[TEXT_COLUMN].tolist()
dates = df[DATE_COLUMN].dt.date.astype(str).tolist()

for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch_texts = texts[i:i + BATCH_SIZE]
    batch_dates = dates[i:i + BATCH_SIZE]

    inputs = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)

    predicted_classes = torch.argmax(probs, dim=1)

    for j, pred_class in enumerate(predicted_classes):
        label = labels[pred_class.item()]
        confidence = probs[j][pred_class].item()

        if label == "positive":
            sentiment_score = confidence
        elif label == "negative":
            sentiment_score = -confidence
        else:
            sentiment_score = 0.0

        results.append({
            "Date": batch_dates[j],
            "Text": batch_texts[j],
            "Sentiment_Label": label,
            "Sentiment_Score": sentiment_score
        })

result_df = pd.DataFrame(results)
result_df.to_csv(OUTPUT_FILE, index=False)

print("FinBERT sentiment saved to:", OUTPUT_FILE)
print(result_df.head())
print("Shape:", result_df.shape)