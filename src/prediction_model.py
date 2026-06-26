import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

INPUT_FILE = "data/processed/gold_vix_sentiment.csv"

df = pd.read_csv(INPUT_FILE)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# Basic technical indicators
df["Return"] = df["Close"].pct_change()
df["MA_5"] = df["Close"].rolling(window=5).mean()
df["MA_10"] = df["Close"].rolling(window=10).mean()
df["MA_20"] = df["Close"].rolling(window=20).mean()

df["Price_Range"] = df["High"] - df["Low"]
df["Open_Close_Diff"] = df["Close"] - df["Open"]
df["Volatility_5"] = df["Return"].rolling(window=5).std()
df["Momentum_5"] = df["Close"] - df["Close"].shift(5)

# EMA and MACD
df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = df["EMA_12"] - df["EMA_26"]
df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

# RSI 14
delta = df["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()

rs = avg_gain / avg_loss
df["RSI_14"] = 100 - (100 / (1 + rs))

# Target: next-day direction
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

df = df.dropna()

features = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",

    "VIX_Open",
    "VIX_High",
    "VIX_Low",
    "VIX_Close",

    "Return",
    "MA_5",
    "MA_10",
    "MA_20",
    "EMA_12",
    "EMA_26",
    "MACD",
    "Signal_Line",
    "RSI_14",
    "Price_Range",
    "Open_Close_Diff",
    "Volatility_5",
    "Momentum_5",

    "Average_Sentiment",
    "News_Count",
    "Positive_Count",
    "Neutral_Count",
    "Negative_Count"
]

X = df[features]
y = df["Target"]

split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nTarget balance:")
print(y.value_counts())
print(y.value_counts(normalize=True))

# Majority baseline
majority_class = y_train.mode()[0]
baseline_preds = [majority_class] * len(y_test)
baseline_accuracy = accuracy_score(y_test, baseline_preds)

print("\nMajority baseline accuracy:", baseline_accuracy)

# Time-series cross validation
tscv = TimeSeriesSplit(n_splits=3)

model = XGBClassifier(
    eval_metric="logloss",
    random_state=42
)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [2, 3],
    "learning_rate": [0.03, 0.05],
    "subsample": [0.8],
    "colsample_bytree": [0.8]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=tscv,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

preds = best_model.predict(X_test)

accuracy = accuracy_score(y_test, preds)
cm = confusion_matrix(y_test, preds)
report = classification_report(y_test, preds)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nFinal XGBoost Accuracy:")
print(accuracy)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)

# Save predictions
os.makedirs("data/processed", exist_ok=True)

predictions = X_test.copy()
predictions["Date"] = df.iloc[split:]["Date"].values
predictions["Actual"] = y_test.values
predictions["Predicted"] = preds

if hasattr(best_model, "predict_proba"):
    predictions["Prediction_Confidence"] = best_model.predict_proba(X_test).max(axis=1)

predictions.to_csv("data/processed/final_predictions.csv", index=False)

# Save model results
results = pd.DataFrame({
    "Model": [
        "Majority Baseline",
        "Gold + VIX + FinBERT XGBoost"
    ],
    "Accuracy": [
        baseline_accuracy,
        accuracy
    ]
})

results.to_csv("data/processed/model_results.csv", index=False)

print("\nSaved:")
print("data/processed/final_predictions.csv")
print("data/processed/model_results.csv")

# -----------------------------
# Feature Importance
# -----------------------------

importance = pd.DataFrame({
    "Feature": features,
    "Importance": best_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(importance.head(10))

importance.to_csv(
    "data/processed/feature_importance.csv",
    index=False
)

plt.figure(figsize=(10,8))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance")

plt.tight_layout()

plt.savefig(
    "data/processed/feature_importance.png",
    dpi=300
)

plt.show()