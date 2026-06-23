from pathlib import Path
import yfinance as yf

Path("data/raw").mkdir(parents=True, exist_ok=True)

ticker = "^VIX"

df = yf.download(ticker, start="2020-01-01", end="2025-01-01")
df = df.reset_index()

output_file = "data/raw/vix_data.csv"
df.to_csv(output_file, index=False)

print("Downloaded VIX data")
print(df.head())
print("Shape:", df.shape)
print("Saved to:", output_file)