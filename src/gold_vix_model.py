import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("data/processed/gold_vix_merged.csv")

# Feature Engineering
df["Return"] = df["Close"].pct_change()
df["MA_5"] = df["Close"].rolling(5).mean()
df["MA_10"] = df["Close"].rolling(10).mean()
df["MA_20"] = df["Close"].rolling(20).mean()

df["Price_Range"] = df["High"] - df["Low"]
df["Open_Close_Diff"] = df["Close"] - df["Open"]

# Target
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

df = df.dropna()

features = [
    "Open", "High", "Low", "Close", "Volume",
    "VIX_Close", "VIX_High", "VIX_Low", "VIX_Open",
    "Return", "MA_5", "MA_10", "MA_20",
    "Price_Range", "Open_Close_Diff"
]

X = df[features]
y = df["Target"]

split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))

print("\nClassification Report:")
print(classification_report(y_test, preds))