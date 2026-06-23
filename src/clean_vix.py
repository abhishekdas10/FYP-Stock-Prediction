import pandas as pd

df = pd.read_csv("data/raw/vix_data.csv")

print("Before cleaning:")
print(df.head())

# Fix columns from yfinance format
df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

df = df.dropna(subset=["Date"])

numeric_cols = ["Close", "High", "Low", "Open", "Volume"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()
df["Date"] = pd.to_datetime(df["Date"])

# Rename columns so they don't conflict with gold data
df = df.rename(columns={
    "Close": "VIX_Close",
    "High": "VIX_High",
    "Low": "VIX_Low",
    "Open": "VIX_Open",
    "Volume": "VIX_Volume"
})

df.to_csv("data/processed/cleaned_vix_data.csv", index=False)

print("\nAfter cleaning:")
print(df.head())
print(df.dtypes)
print("Shape:", df.shape)
print("Saved cleaned VIX data")