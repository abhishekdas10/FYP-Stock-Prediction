import pandas as pd
import matplotlib.pyplot as plt

# Load processed data
df = pd.read_csv("data/processed/gold_vix_sentiment.csv")

df["Date"] = pd.to_datetime(df["Date"])

plt.figure(figsize=(15,6))

plt.plot(
    df["Date"],
    df["Close"],
    linewidth=2,
    label="Gold Close Price"
)

plt.title("Gold Closing Price (2015–2020)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "data/processed/gold_price_chart.png",
    dpi=300
)

plt.show()