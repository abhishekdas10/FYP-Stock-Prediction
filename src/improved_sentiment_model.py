import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

df = pd.read_csv("data/raw/financial_sentiment.csv")

df = df.rename(columns={
    "Sentence": "text",
    "Sentiment": "sentiment"
})

df = df.dropna(subset=["text", "sentiment"])

X = df["text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", LogisticRegression(max_iter=2000))
])

params = {
    "tfidf__max_features": [3000, 5000, 8000],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "classifier__C": [0.5, 1, 2, 5],
    "classifier__class_weight": [None, "balanced"]
}

grid = GridSearchCV(
    pipeline,
    params,
    cv=3,
    scoring="f1_weighted",
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

print("Best parameters:")
print(grid.best_params_)

best_model = grid.best_estimator_
preds = best_model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, preds))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))

print("\nClassification Report:")
print(classification_report(y_test, preds))