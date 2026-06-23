import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/processed/cleaned_gold_data.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Feature engineering
df["Return"] = df["Close"].pct_change()
df["MA_5"] = df["Close"].rolling(5).mean()
df["MA_10"] = df["Close"].rolling(10).mean()
df["MA_20"] = df["Close"].rolling(20).mean()

df["Price_Range"] = df["High"] - df["Low"]
df["Open_Close_Diff"] = df["Close"] - df["Open"]
df["Volatility_5"] = df["Return"].rolling(5).std()
df["Momentum_5"] = df["Close"] - df["Close"].shift(5)
df["Lag_1"] = df["Close"].shift(1)
df["Lag_2"] = df["Close"].shift(2)

# Target: next day movement
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

df = df.dropna()

features = [
    "Open", "High", "Low", "Close", "Volume",
    "Return", "MA_5", "MA_10", "MA_20",
    "Price_Range", "Open_Close_Diff",
    "Volatility_5", "Momentum_5",
    "Lag_1", "Lag_2"
]

X = df[features]
y = df["Target"]

split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=5
    )
}

for name, model in models.items():
    print("\n==============================")
    print(name)
    print("==============================")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))