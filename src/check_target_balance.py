import pandas as pd

df = pd.read_csv("data/processed/cleaned_gold_data.csv")

df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
df = df.dropna()

print("Target value counts:")
print(df["Target"].value_counts())

print("\nTarget proportions:")
print(df["Target"].value_counts(normalize=True))