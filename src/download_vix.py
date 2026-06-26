import os
import yfinance as yf
import pandas as pd

def download_vix_data():
    print("Downloading VIX Volatility (^VIX) data...")
    ticker = "^VIX"
    start_date = "2015-01-01"
    end_date = "2020-05-26"  # Exclusive in yfinance, so this includes 2020-12-31
    
    # Download data
    df = yf.download(ticker, start=start_date, end=end_date)
    
    if df.empty:
        raise ValueError("Failed to download VIX data. The returned DataFrame is empty.")
        
    # Reset index to make Date a column
    df = df.reset_index()
    
    # Filter to make sure we don't have any date beyond 2020-12-31
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date'] <= '2020-12-31']
    
    # Save output
    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "vix_2015_2020.csv")
    
    # Flatten MultiIndex columns if returned
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[1] == '' else col[0] for col in df.columns]
        
    df.to_csv(output_path, index=False)
    print(f"Successfully downloaded VIX data. Shape: {df.shape}. Saved to {output_path}")

if __name__ == "__main__":
    download_vix_data()