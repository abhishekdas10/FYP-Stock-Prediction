# Stock Market Trend Prediction Using Sentiment Analysis and Machine Learning

**Final Year Project — Griffith College Cork**
**Student:** Saugat | **Supervisor:** Atif

---

## Project Overview

This project investigates whether financial news sentiment, captured using **FinBERT**, can improve the prediction of **Gold market trends** when combined with technical indicators and macroeconomic features.

### Research Question
> *Does financial news sentiment improve gold market trend prediction across multiple time horizons?*

---

## Pipeline

```
Yahoo Finance (Gold + VIX)  ──┐
Financial News + FinBERT    ──┼──▶ Feature Engineering ──▶ XGBoost ──▶ Trend Prediction
DXY (US Dollar Index)       ──┘
```

---

## Key Results

| Horizon | Accuracy | vs Majority Baseline (44%) |
|---------|----------|---------------------------|
| 1 day   | 49.53%   | +5.5%                     |
| 5 days  | **50.40%** | **+6.4%**               |
| 10 days | 47.92%   | +3.9%                     |
| 20 days | 45.54%   | +1.5%                     |

The **5-day horizon** achieved the best accuracy, suggesting sentiment features provide stronger predictive signal over short-to-medium term horizons.

---

## Project Structure

```
FYP_Stock_Prediction/
│
├── data/
│   ├── raw/                        # Original downloaded datasets
│   │   ├── gold_2015_2020.csv
│   │   ├── vix_2015_2020.csv
│   │   └── financial_news.csv
│   └── processed/                  # Cleaned and merged datasets
│       ├── gold_vix_sentiment.csv  # Main input file
│       ├── news_sentiment.csv      # FinBERT output
│       └── multi_horizon_results.csv
│
└── src/
    ├── download_data.py            # Download Gold data from Yahoo Finance
    ├── download_vix.py             # Download VIX data from Yahoo Finance
    ├── clean_data.py               # Clean Gold data
    ├── clean_vix.py                # Clean VIX data
    ├── merge_gold_vix.py           # Merge Gold + VIX
    ├── finbert_news.py             # Run FinBERT on financial news
    ├── create_daily_sentiment.py   # Aggregate sentiment by day
    ├── merge_sentiment.py          # Merge sentiment with market data
    ├── multi_horizon_model.py      # Main model — multi-horizon XGBoost
    ├── prediction_model.py         # Baseline single-horizon model
    └── dashboard.py                # Streamlit dashboard
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline (if starting fresh)
```bash
python src/download_data.py
python src/download_vix.py
python src/clean_data.py
python src/clean_vix.py
python src/merge_gold_vix.py
python src/finbert_news.py          # Warning: slow — processes 49,637 articles
python src/create_daily_sentiment.py
python src/merge_sentiment.py
```

### 3. Run the improved multi-horizon model
```bash
python src/multi_horizon_model.py
```

### 4. Launch the dashboard
```bash
streamlit run src/dashboard.py
```

---

## Model Details

- **Algorithm:** XGBoost Classifier
- **Validation:** TimeSeriesSplit (5 folds) — no data leakage
- **Class balancing:** `scale_pos_weight`
- **Label thresholding:** Moves < 0.2% removed as noise
- **Horizons:** 1, 5, 10, 20 trading days
- **Features:** 32 total — market, VIX, technical indicators, FinBERT sentiment, DXY

---

## Technologies

| Category | Tools |
|----------|-------|
| Language | Python 3.10 |
| ML | XGBoost, scikit-learn |
| NLP | FinBERT (HuggingFace Transformers) |
| Data | Yahoo Finance, yfinance |
| Dashboard | Streamlit |
| Visualisation | Matplotlib, Seaborn |

---

## Limitations

- News dataset contains general financial news rather than gold-specific articles
- Dataset period limited to 2015–2020
- Gold prices influenced by macro variables not fully captured (interest rates, inflation)
