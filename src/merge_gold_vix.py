import pandas as pd

gold = pd.read_csv("data/processed/cleaned_gold_data.csv")
vix = pd.read_csv("data/processed/cleaned_vix_data.csv")

gold["Date"] = pd.to_datetime(gold["Date"])
vix["Date"] = pd.to_datetime(vix["Date"])

# Merge on Date
merged = pd.merge(gold, vix, on="Date", how="inner")

merged.to_csv("data/processed/gold_vix_merged.csv", index=False)

print("Merged dataset:")
print(merged.head())
print("Shape:", merged.shape)
print("Saved to data/processed/gold_vix_merged.csv")