import pandas as pd
import os

INPUT_FILE = "data/processed/news_sentiment.csv"
OUTPUT_FILE = "data/processed/daily_sentiment.csv"

os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

daily_sentiment = (
    df.groupby("Date")
    .agg(
        Average_Sentiment=("Sentiment_Score", "mean"),
        News_Count=("Sentiment_Score", "count"),
        Positive_Count=("Sentiment_Label", lambda x: (x == "positive").sum()),
        Neutral_Count=("Sentiment_Label", lambda x: (x == "neutral").sum()),
        Negative_Count=("Sentiment_Label", lambda x: (x == "negative").sum())
    )
    .reset_index()
)

daily_sentiment.to_csv(OUTPUT_FILE, index=False)

print("Daily sentiment created successfully!")
print(daily_sentiment.head())
print("\nShape:", daily_sentiment.shape)