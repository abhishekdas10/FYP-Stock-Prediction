import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/gc_f_gold_futures.csv")

print("First 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns)

print("\nShape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)