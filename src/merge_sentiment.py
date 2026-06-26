import pandas as pd
import os

MARKET_FILE = "data/processed/gold_vix_2015_2020.csv"
SENTIMENT_FILE = "data/processed/daily_sentiment.csv"
OUTPUT_FILE = "data/processed/gold_vix_sentiment.csv"

os.makedirs("data/processed", exist_ok=True)

market = pd.read_csv(MARKET_FILE)
sentiment = pd.read_csv(SENTIMENT_FILE)

market["Date"] = pd.to_datetime(market["Date"])
sentiment["Date"] = pd.to_datetime(sentiment["Date"])

merged = pd.merge(
    market,
    sentiment,
    on="Date",
    how="left"
)

sentiment_cols = [
    "Average_Sentiment",
    "News_Count",
    "Positive_Count",
    "Neutral_Count",
    "Negative_Count"
]

merged[sentiment_cols] = merged[sentiment_cols].fillna(0)

count_columns = [
    "News_Count",
    "Positive_Count",
    "Neutral_Count",
    "Negative_Count"
]

merged[count_columns] = merged[count_columns].astype(int)

merged.to_csv(OUTPUT_FILE, index=False)

print("Merged market + sentiment data saved successfully!")
print(merged.head())
print("\nShape:", merged.shape)
print("\nMissing values:")
print(merged.isnull().sum())