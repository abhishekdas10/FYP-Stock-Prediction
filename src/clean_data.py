import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/gc_f_gold_futures.csv")

print("Before cleaning:")
print(df.head())

# 🔹 Fix column names (flatten if needed)
df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

# 🔹 Remove rows where Date is missing
df = df.dropna(subset=["Date"])

# 🔹 Convert numeric columns properly
cols = ["Close", "High", "Low", "Open", "Volume"]

for col in cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 🔹 Drop any remaining NaNs
df = df.dropna()

# 🔹 Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"])

print("\nAfter cleaning:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nShape:", df.shape)

# Save cleaned data
df.to_csv("data/processed/cleaned_gold_data.csv", index=False)

print("\nSaved cleaned dataset!")