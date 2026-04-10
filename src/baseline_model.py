import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load cleaned data
df = pd.read_csv("data/processed/cleaned_gold_data.csv")

# Convert Date again just to be safe
df["Date"] = pd.to_datetime(df["Date"])

# Create features
df["Return"] = df["Close"].pct_change()
df["MA_5"] = df["Close"].rolling(5).mean()
df["MA_10"] = df["Close"].rolling(10).mean()

# Create target:
# 1 = next day close is higher than today
# 0 = next day close is not higher than today
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

# Drop rows with NaN values caused by rolling/shift
df = df.dropna()

# Feature columns
features = ["Open", "High", "Low", "Close", "Volume", "Return", "MA_5", "MA_10"]

X = df[features]
y = df["Target"]

# Train/test split without shuffling because this is time-series style data
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
preds = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, preds))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))
print("\nClassification Report:")
print(classification_report(y_test, preds))