import pandas as pd

file_path = "data/raw/financial_news.csv"

df = pd.read_csv(file_path)

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

# Try to detect date column
possible_date_cols = ["Date", "date", "Published", "published", "Timestamp", "timestamp"]
date_col = None

for col in possible_date_cols:
    if col in df.columns:
        date_col = col
        break

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    print("\nDate column:", date_col)
    print("Start date:", df[date_col].min())
    print("End date:", df[date_col].max())
else:
    print("\nNo date column found.")