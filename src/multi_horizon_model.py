import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

warnings.filterwarnings("ignore")

# ── Paths (same conventions as your existing scripts) ─────────────────────────
INPUT_FILE  = "data/processed/gold_vix_sentiment.csv"
OUTPUT_DIR  = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
HORIZONS        = [1, 5, 10, 20]   # trading-day prediction horizons
LABEL_THRESHOLD = 0.002             # ignore moves smaller than 0.2% (noise filter)
START_DATE      = "2015-01-01"
END_DATE        = "2020-12-31"
N_SPLITS        = 5                 # TimeSeriesSplit folds

# ── Feature list (matches prediction_model.py exactly, plus DXY below) ───────
BASE_FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "VIX_Open", "VIX_High", "VIX_Low", "VIX_Close",
    "Return", "MA_5", "MA_10", "MA_20",
    "EMA_12", "EMA_26", "MACD", "Signal_Line", "RSI_14",
    "Price_Range", "Open_Close_Diff", "Volatility_5", "Momentum_5",
    "Average_Sentiment", "News_Count",
    "Positive_Count", "Neutral_Count", "Negative_Count",
]

DXY_FEATURES = [
    "DXY_Close", "DXY_Return", "DXY_MA_5", "DXY_Volatility", "Gold_DXY_Ratio"
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def load_and_engineer(filepath: str) -> pd.DataFrame:
    """Load gold_vix_sentiment.csv and add all technical features,
    exactly matching the feature engineering in prediction_model.py."""
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Technical indicators (same as prediction_model.py)
    df["Return"]         = df["Close"].pct_change()
    df["MA_5"]           = df["Close"].rolling(5).mean()
    df["MA_10"]          = df["Close"].rolling(10).mean()
    df["MA_20"]          = df["Close"].rolling(20).mean()
    df["Price_Range"]    = df["High"] - df["Low"]
    df["Open_Close_Diff"]= df["Close"] - df["Open"]
    df["Volatility_5"]   = df["Return"].rolling(5).std()
    df["Momentum_5"]     = df["Close"] - df["Close"].shift(5)
    df["EMA_12"]         = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"]         = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]           = df["EMA_12"] - df["EMA_26"]
    df["Signal_Line"]    = df["MACD"].ewm(span=9, adjust=False).mean()

    delta    = df["Close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs       = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    print(f"[load] Base dataset: {df.shape[0]} rows")
    return df


def add_dxy(df: pd.DataFrame) -> tuple:
    """
    Download DXY (US Dollar Index) from Yahoo Finance and merge.
    Gold moves inversely to the dollar — the strongest macro relationship.
    Returns (enriched_df, dxy_feature_cols_actually_added).
    """
    print("[dxy] Downloading DXY from Yahoo Finance...")
    try:
        dxy = yf.download("DX-Y.NYB", start=START_DATE, end=END_DATE,
                          progress=False, auto_adjust=True)
        if dxy.empty:
            raise ValueError("Empty result")
    except Exception as e:
        print(f"[dxy] WARNING: Download failed ({e}). Skipping DXY features.")
        return df, []

    # Flatten MultiIndex columns if present
    if isinstance(dxy.columns, pd.MultiIndex):
        dxy.columns = [col[0] for col in dxy.columns]

    dxy = dxy[["Close"]].copy()
    dxy.columns = ["DXY_Close"]
    dxy.index = pd.to_datetime(dxy.index).tz_localize(None)
    dxy.index.name = "Date"
    dxy = dxy.reset_index()

    df = df.merge(dxy, on="Date", how="left")
    df["DXY_Close"] = df["DXY_Close"].ffill()

    df["DXY_Return"]     = df["DXY_Close"].pct_change()
    df["DXY_MA_5"]       = df["DXY_Close"].rolling(5).mean()
    df["DXY_Volatility"] = df["DXY_Close"].pct_change().rolling(5).std()
    df["Gold_DXY_Ratio"] = df["Close"] / df["DXY_Close"]

    added = [c for c in DXY_FEATURES if c in df.columns]
    print(f"[dxy] Added features: {added}")
    return df, added


# ══════════════════════════════════════════════════════════════════════════════
# 2. LABEL CREATION WITH THRESHOLDING
# ══════════════════════════════════════════════════════════════════════════════

def make_labels(df: pd.DataFrame, horizon: int, threshold: float) -> pd.DataFrame:
    """
    Label = 1 (Up)   if price rises  > threshold over `horizon` days.
    Label = 0 (Down) if price falls  > threshold over `horizon` days.
    Rows where the move is too small are dropped — pure noise.

    Why this helps: the baseline labels every single day including 0.01% moves
    that are essentially random. Removing them makes the remaining labels cleaner.
    """
    future_return = df["Close"].pct_change(horizon).shift(-horizon)
    df = df.copy()
    df["Target"] = np.where(
        future_return >  threshold, 1,
        np.where(future_return < -threshold, 0, np.nan)
    )
    n_before = len(df)
    df = df.dropna(subset=["Target"]).reset_index(drop=True)
    df["Target"] = df["Target"].astype(int)

    up   = (df["Target"] == 1).sum()
    down = (df["Target"] == 0).sum()
    print(f"  rows kept={len(df)} | dropped(noise)={n_before - len(df)} | Up={up} Down={down}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_evaluate(df: pd.DataFrame, feature_cols: list, horizon: int) -> dict:
    """
    TimeSeriesSplit cross-validation — never leaks future data into training.
    scale_pos_weight corrects class imbalance so both Up and Down get predicted.
    """
    X = df[feature_cols].fillna(0)
    y = df["Target"]

    neg = (y == 0).sum()
    pos = (y == 1).sum()
    spw = neg / pos if pos > 0 else 1.0

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    accs, f1s, y_true_all, y_prob_all = [], [], [], []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBClassifier(
            n_estimators     = 300,
            max_depth        = 4,
            learning_rate    = 0.05,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            scale_pos_weight = spw,
            eval_metric      = "logloss",
            random_state     = 42,
            n_jobs           = -1,
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average="weighted"))
        y_true_all.extend(y_test.tolist())
        y_prob_all.extend(y_prob.tolist())

    # Final model on full data (for feature importance + dashboard)
    final_model = XGBClassifier(
        n_estimators     = 300,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = spw,
        eval_metric      = "logloss",
        random_state     = 42,
        n_jobs           = -1,
    )
    final_model.fit(X, y, verbose=False)

    return {
        "horizon"     : horizon,
        "accuracy"    : float(np.mean(accs)),
        "f1"          : float(np.mean(f1s)),
        "y_true"      : np.array(y_true_all),
        "y_prob"      : np.array(y_prob_all),
        "feature_cols": feature_cols,
        "importances" : final_model.feature_importances_,
        "model"       : final_model,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. PLOTS  (all saved to data/processed/ — same location as prediction_model.py)
# ══════════════════════════════════════════════════════════════════════════════

def _savefig(fig, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


def plot_confusion_matrix(result: dict):
    h      = result["horizon"]
    y_true = result["y_true"]
    y_pred = (result["y_prob"] >= 0.5).astype(int)
    cm     = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Down", "Up"],
                yticklabels=["Down", "Up"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {h}-Day Horizon")
    _savefig(fig, f"confusion_matrix_{h}d.png")


def plot_roc_curve(result: dict):
    h = result["horizon"]
    fpr, tpr, _ = roc_curve(result["y_true"], result["y_prob"])
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {h}-Day Horizon")
    ax.legend()
    _savefig(fig, f"roc_curve_{h}d.png")


def plot_pr_curve(result: dict):
    h = result["horizon"]
    precision, recall, _ = precision_recall_curve(result["y_true"], result["y_prob"])
    ap = average_precision_score(result["y_true"], result["y_prob"])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall Curve — {h}-Day Horizon")
    ax.legend()
    _savefig(fig, f"pr_curve_{h}d.png")


def plot_feature_importance(result: dict, top_n: int = 15):
    h = result["horizon"]
    feat_df = (
        pd.DataFrame({"Feature": result["feature_cols"],
                      "Importance": result["importances"]})
        .sort_values("Importance", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(feat_df["Feature"][::-1], feat_df["Importance"][::-1])
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_n} Features — {h}-Day Horizon")
    _savefig(fig, f"feature_importance_{h}d.png")


def plot_train_test_split(df: pd.DataFrame, horizon: int):
    """Show last TimeSeriesSplit fold on the gold price chart."""
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    splits = list(tscv.split(df))
    train_idx, test_idx = splits[-1]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["Date"].iloc[train_idx], df["Close"].iloc[train_idx],
            label="Train", color="steelblue")
    ax.plot(df["Date"].iloc[test_idx],  df["Close"].iloc[test_idx],
            label="Test",  color="coral")
    ax.set_title(f"Train / Test Split — {horizon}-Day Horizon (Final Fold)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Gold Price (USD)")
    ax.legend()
    _savefig(fig, f"train_test_split_{horizon}d.png")


def plot_horizon_comparison(results: list):
    """Bar chart comparing Accuracy and F1 across all horizons — key dissertation figure."""
    horizons = [r["horizon"]  for r in results]
    accs     = [r["accuracy"] for r in results]
    f1s      = [r["f1"]       for r in results]

    x     = np.arange(len(horizons))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, accs, width, label="Accuracy",       color="steelblue")
    bars2 = ax.bar(x + width/2, f1s,  width, label="F1 (weighted)",  color="coral")

    ax.axhline(0.44, color="gray", linestyle="--", linewidth=1.2,
               label="Majority baseline (~44%)")
    ax.set_xlabel("Prediction Horizon (trading days)")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Across Prediction Horizons")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}d" for h in horizons])
    ax.set_ylim(0, 1)
    ax.legend()

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=9)

    _savefig(fig, "horizon_comparison.png")


def plot_actual_vs_predicted(df: pd.DataFrame, result: dict):
    """Plot actual vs predicted direction on gold price in the last test fold."""
    h    = result["horizon"]
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    splits  = list(tscv.split(df))
    _, test_idx = splits[-1]

    test_df   = df.iloc[test_idx].copy()
    n_preds   = len(result["y_true"]) // N_SPLITS   # approx last fold size
    y_true_fold = result["y_true"][-n_preds:]
    y_pred_fold = (result["y_prob"][-n_preds:] >= 0.5).astype(int)

    plot_dates = test_df["Date"].values[-n_preds:]
    plot_close = test_df["Close"].values[-n_preds:]

    correct = y_true_fold == y_pred_fold

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(plot_dates, plot_close, color="gray", lw=1, label="Gold Close")
    ax.scatter(plot_dates[correct],  plot_close[correct],
               color="green", s=15, label="Correct", zorder=5)
    ax.scatter(plot_dates[~correct], plot_close[~correct],
               color="red",   s=15, label="Wrong",   zorder=5)
    ax.set_title(f"Actual vs Predicted Direction — {h}-Day Horizon (Last Fold)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Gold Price (USD)")
    ax.legend()
    _savefig(fig, f"actual_vs_predicted_{h}d.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  Multi-Horizon Gold Trend Prediction (Improved)")
    print("=" * 60)

    # Load base data with feature engineering
    df = load_and_engineer(INPUT_FILE)

    # Add DXY macro feature
    df, dxy_cols = add_dxy(df)

    feature_cols = BASE_FEATURES + dxy_cols

    all_results = []

    for horizon in HORIZONS:
        print(f"\n── Horizon: {horizon} trading day(s) " + "─" * 30)

        # Label with noise threshold
        df_h = make_labels(df, horizon, LABEL_THRESHOLD)

        if len(df_h) < 150:
            print(f"  [skip] Not enough samples after thresholding.")
            continue

        result = train_evaluate(df_h, feature_cols, horizon)
        all_results.append(result)

        print(f"  Accuracy : {result['accuracy']:.4f}")
        print(f"  F1       : {result['f1']:.4f}")
        print(classification_report(
            result["y_true"],
            (result["y_prob"] >= 0.5).astype(int),
            target_names=["Down", "Up"]
        ))

        # Per-horizon plots
        plot_confusion_matrix(result)
        plot_roc_curve(result)
        plot_pr_curve(result)
        plot_feature_importance(result)
        plot_train_test_split(df_h, horizon)
        plot_actual_vs_predicted(df_h, result)

    # Cross-horizon comparison (the key dissertation figure)
    if len(all_results) > 1:
        plot_horizon_comparison(all_results)

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"{'Horizon':<12} {'Accuracy':>10} {'F1 (weighted)':>15}")
    print("-" * 55)
    for r in all_results:
        print(f"{str(r['horizon'])+'d':<12} {r['accuracy']:>10.4f} {r['f1']:>15.4f}")
    print("=" * 55)

    # Save CSV summary
    rows = [{"Horizon_Days": r["horizon"],
             "Accuracy": round(r["accuracy"], 4),
             "F1_Weighted": round(r["f1"], 4)}
            for r in all_results]
    csv_path = os.path.join(OUTPUT_DIR, "multi_horizon_results.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n[saved] {csv_path}")
    print(f"[done]  All plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()