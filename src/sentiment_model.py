import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("data/raw/financial_sentiment.csv")

print("First rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

# Rename columns if needed
df = df.rename(columns={
    "Sentence": "text",
    "Sentiment": "sentiment"
})

# Clean data
df = df.dropna(subset=["text", "sentiment"])

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

# Features and target
X = df["text"]
y = df["sentiment"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# TF-IDF + Logistic Regression model
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Evaluate
print("\nAccuracy:", accuracy_score(y_test, preds))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))

print("\nClassification Report:")
print(classification_report(y_test, preds))