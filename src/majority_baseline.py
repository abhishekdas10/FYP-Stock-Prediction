import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("data/processed/gold_vix_merged.csv")

df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
df = df.dropna()

y = df["Target"]

split = int(len(df) * 0.8)

y_train = y.iloc[:split]
y_test = y.iloc[split:]

majority_class = y_train.mode()[0]

preds = [majority_class] * len(y_test)

print("Majority class from training data:", majority_class)
print("Majority baseline accuracy:", accuracy_score(y_test, preds))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))

print("\nClassification Report:")
print(classification_report(y_test, preds))