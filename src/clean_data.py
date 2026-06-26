import os
import pandas as pd

def clean_gold_data():
    input_path = os.path.join("data", "raw", "gold_2015_2020.csv")
    output_path = os.path.join("data", "processed", "cleaned_gold_2015_2020.csv")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing input file: {input_path}. Please run download_data.py first.")
        
    print(f"Cleaning gold data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Columns to convert to numeric
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            raise KeyError(f"Expected column '{col}' not found in the raw gold data.")
            
    # Drop rows where critical numeric columns are null (invalid rows)
    initial_shape = df.shape
    df = df.dropna(subset=numeric_cols)
    
    # Save processed file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Gold data cleaning complete. Original rows: {initial_shape[0]}, Cleaned rows: {df.shape[0]}. Saved to {output_path}")

if __name__ == "__main__":
    clean_gold_data()