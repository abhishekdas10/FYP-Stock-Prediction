from pathlib import Path
import yfinance as yf
import pandas as pd

# Make sure the raw data folder exists
Path("data/raw").mkdir(parents=True, exist_ok=True)

ticker = "GC=F"

# Download historical data
df = yf.download(ticker, start="2020-01-01", end="2025-01-01")

# Reset index so Date becomes a normal column
df = df.reset_index()

# Save to CSV
output_file = "data/raw/gc_f_gold_futures.csv"
df.to_csv(output_file, index=False)

print("Downloaded data for:", ticker)
print("Shape:", df.shape)
print(df.head())
print("Saved to:", output_file)