import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("data/processed/cleaned_gold_data.csv")

# Feature engineering
df["Return"] = df["Close"].pct_change()
df["MA_5"] = df["Close"].rolling(5).mean()
df["MA_10"] = df["Close"].rolling(10).mean()

# Target
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

df = df.dropna()

features = ["Open", "High", "Low", "Close", "Volume", "Return", "MA_5", "MA_10"]
X = df[features]
y = df["Target"]

# Time-based split
split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# Train & evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.4f}")