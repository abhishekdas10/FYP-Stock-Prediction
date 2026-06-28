import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gold Trend Prediction",
    page_icon="📈",
    layout="wide"
)

# ── Constants ─────────────────────────────────────────────────────────────────
HORIZONS        = [1, 5, 10, 20]
LABEL_THRESHOLD = 0.002
START_DATE      = "2015-01-01"
END_DATE        = "2020-12-31"
N_SPLITS        = 5

DATA_PATH = "data/processed/gold_vix_sentiment.csv"

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

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("📈 Gold Trend Prediction")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "🏠 Home",
    "📊 Dataset Overview",
    "🥇 Gold Price Chart",
    "📰 Sentiment Analysis",
    "🔍 Feature Importance",
    "📉 Model Evaluation",
    "🔭 Horizon Comparison",
    "🎯 Prediction Results",
    "🔮 Predict a Date",
])
st.sidebar.markdown("---")
st.sidebar.caption("FYP — Griffith College Cork\nStudent: Abhishek \nSupervisor: Atif")


# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL (cached so they only run once)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_base_data():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["Return"]          = df["Close"].pct_change()
    df["MA_5"]            = df["Close"].rolling(5).mean()
    df["MA_10"]           = df["Close"].rolling(10).mean()
    df["MA_20"]           = df["Close"].rolling(20).mean()
    df["Price_Range"]     = df["High"] - df["Low"]
    df["Open_Close_Diff"] = df["Close"] - df["Open"]
    df["Volatility_5"]    = df["Return"].rolling(5).std()
    df["Momentum_5"]      = df["Close"] - df["Close"].shift(5)
    df["EMA_12"]          = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"]          = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]            = df["EMA_12"] - df["EMA_26"]
    df["Signal_Line"]     = df["MACD"].ewm(span=9, adjust=False).mean()

    delta    = df["Close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs       = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    return df


@st.cache_data(show_spinner=False)
def load_dxy():
    try:
        dxy = yf.download("DX-Y.NYB", start=START_DATE, end=END_DATE,
                          progress=False, auto_adjust=True)
        if dxy.empty:
            return None
        if isinstance(dxy.columns, pd.MultiIndex):
            dxy.columns = [col[0] for col in dxy.columns]
        dxy = dxy[["Close"]].copy()
        dxy.columns = ["DXY_Close"]
        dxy.index = pd.to_datetime(dxy.index).tz_localize(None)
        dxy.index.name = "Date"
        return dxy.reset_index()
    except Exception:
        return None


def enrich_with_dxy(df, dxy_df):
    if dxy_df is None:
        return df, []
    df = df.merge(dxy_df, on="Date", how="left")
    df["DXY_Close"]    = df["DXY_Close"].ffill()
    df["DXY_Return"]   = df["DXY_Close"].pct_change()
    df["DXY_MA_5"]     = df["DXY_Close"].rolling(5).mean()
    df["DXY_Volatility"] = df["DXY_Close"].pct_change().rolling(5).std()
    df["Gold_DXY_Ratio"] = df["Close"] / df["DXY_Close"]
    added = [c for c in DXY_FEATURES if c in df.columns]
    return df, added


def make_labels(df, horizon, threshold):
    future_return = df["Close"].pct_change(horizon).shift(-horizon)
    df = df.copy()
    df["Target"] = np.where(
        future_return >  threshold, 1,
        np.where(future_return < -threshold, 0, np.nan)
    )
    df = df.dropna(subset=["Target"]).reset_index(drop=True)
    df["Target"] = df["Target"].astype(int)
    return df


@st.cache_data(show_spinner=False)
def run_model(horizon):
    df_base = load_base_data()
    dxy_df  = load_dxy()
    df_base, dxy_cols = enrich_with_dxy(df_base, dxy_df)
    feature_cols = BASE_FEATURES + dxy_cols

    df_h = make_labels(df_base, horizon, LABEL_THRESHOLD)
    X    = df_h[feature_cols].fillna(0)
    y    = df_h["Target"]

    neg = (y == 0).sum()
    pos = (y == 1).sum()
    spw = neg / pos if pos > 0 else 1.0

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    accs, f1s, y_true_all, y_prob_all = [], [], [], []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average="weighted"))
        y_true_all.extend(y_test.tolist())
        y_prob_all.extend(y_prob.tolist())

    # Final model on full data for feature importance
    final_model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric="logloss",
        random_state=42, n_jobs=-1,
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
        "df"          : df_h,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════════════════════

# ── HOME ──────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("📈 Stock Market Trend Prediction")
    st.subheader("Using Sentiment Analysis and Machine Learning")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset Period", "2015 – 2020")
    col2.metric("ML Model", "XGBoost")
    col3.metric("NLP Model", "FinBERT")

    st.markdown("---")
    st.markdown("""
    ### Project Overview
    This project investigates whether **financial news sentiment** can improve the prediction
    of **Gold market trends** when combined with technical indicators and macroeconomic features.

    ### Pipeline
    ```
    Yahoo Finance (Gold + VIX)  ──┐
    Financial News + FinBERT    ──┼──▶ Feature Engineering ──▶ XGBoost ──▶ Trend Prediction
    DXY (US Dollar Index)       ──┘
    ```

    ### Key Features
    - **FinBERT** sentiment analysis on 49,637 financial news articles
    - **XGBoost** classifier with TimeSeriesSplit cross-validation
    - **Multi-horizon** prediction: 1, 5, 10, and 20 trading days
    - **DXY** added as a macroeconomic feature (gold–dollar inverse relationship)
    - **Label thresholding** to remove noisy near-zero price movements

    ### Research Question
    > *Does financial news sentiment, as captured by FinBERT, improve gold market trend prediction
    > over multiple time horizons?*
    """)

    st.markdown("---")
    st.markdown("""
    | Horizon | Accuracy | vs Majority Baseline |
    |---------|----------|----------------------|
    | 1 day   | 49.53%   | +5.5%                |
    | 5 days  | **50.40%**   | **+6.4%**        |
    | 10 days | 47.92%   | +3.9%                |
    | 20 days | 45.54%   | +1.5%                |
    """)


# ── DATASET OVERVIEW ──────────────────────────────────────────────────────────
elif page == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")
    st.markdown("---")

    with st.spinner("Loading dataset..."):
        df = load_base_data()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Features", len(BASE_FEATURES))
    col3.metric("Date From", str(df["Date"].min().date()))
    col4.metric("Date To",   str(df["Date"].max().date()))

    st.markdown("---")
    st.subheader("Sample Data")
    st.dataframe(df[["Date", "Open", "High", "Low", "Close", "Volume",
                      "VIX_Close", "Average_Sentiment", "News_Count"]].head(10),
                 use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Missing Values")
        missing = df[BASE_FEATURES].isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.success("No missing values in feature columns.")
        else:
            st.dataframe(missing.rename("Missing Count"))

    with col2:
        st.subheader("Feature Statistics")
        st.dataframe(df[["Close", "VIX_Close", "Average_Sentiment",
                          "RSI_14", "MACD"]].describe().round(3),
                     use_container_width=True)

    st.markdown("---")
    st.subheader("Class Balance per Horizon")
    cols = st.columns(4)
    for i, h in enumerate(HORIZONS):
        df_h = make_labels(df, h, LABEL_THRESHOLD)
        up   = (df_h["Target"] == 1).sum()
        down = (df_h["Target"] == 0).sum()
        with cols[i]:
            st.metric(f"{h}-Day Horizon", f"{len(df_h)} rows")
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.pie([up, down], labels=["Up", "Down"],
                   colors=["#2ecc71", "#e74c3c"],
                   autopct="%1.1f%%", startangle=90)
            ax.set_title(f"{h}d")
            st.pyplot(fig)
            plt.close(fig)


# ── GOLD PRICE CHART ──────────────────────────────────────────────────────────
elif page == "🥇 Gold Price Chart":
    st.title("🥇 Gold Price History (2015–2020)")
    st.markdown("---")

    with st.spinner("Loading data..."):
        df = load_base_data()

    chart_type = st.selectbox("Select chart", [
        "Closing Price", "Price with Moving Averages",
        "RSI (14)", "MACD", "VIX Overlay"
    ])

    fig, ax = plt.subplots(figsize=(14, 5))

    if chart_type == "Closing Price":
        ax.plot(df["Date"], df["Close"], color="gold", lw=1.5)
        ax.set_title("Gold Closing Price (2015–2020)")
        ax.set_ylabel("Price (USD)")

    elif chart_type == "Price with Moving Averages":
        ax.plot(df["Date"], df["Close"], color="gold",    lw=1.5, label="Close")
        ax.plot(df["Date"], df["MA_5"],  color="blue",    lw=1,   label="MA 5")
        ax.plot(df["Date"], df["MA_20"], color="red",     lw=1,   label="MA 20")
        ax.set_title("Gold Price with Moving Averages")
        ax.set_ylabel("Price (USD)")
        ax.legend()

    elif chart_type == "RSI (14)":
        ax.plot(df["Date"], df["RSI_14"], color="purple", lw=1.2)
        ax.axhline(70, color="red",   linestyle="--", lw=1, label="Overbought (70)")
        ax.axhline(30, color="green", linestyle="--", lw=1, label="Oversold (30)")
        ax.set_title("RSI (14-Day)")
        ax.set_ylabel("RSI")
        ax.legend()

    elif chart_type == "MACD":
        ax.plot(df["Date"], df["MACD"],        color="blue",  lw=1.2, label="MACD")
        ax.plot(df["Date"], df["Signal_Line"], color="red",   lw=1.2, label="Signal Line")
        ax.axhline(0, color="gray", linestyle="--", lw=0.8)
        ax.set_title("MACD")
        ax.legend()

    elif chart_type == "VIX Overlay":
        ax2 = ax.twinx()
        ax.plot(df["Date"],  df["Close"],     color="gold",  lw=1.5, label="Gold")
        ax2.plot(df["Date"], df["VIX_Close"], color="red",   lw=1,   label="VIX", alpha=0.7)
        ax.set_ylabel("Gold Price (USD)", color="gold")
        ax2.set_ylabel("VIX", color="red")
        ax.set_title("Gold Price vs VIX")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")

    ax.set_xlabel("Date")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── SENTIMENT ANALYSIS ────────────────────────────────────────────────────────
elif page == "📰 Sentiment Analysis":
    st.title("📰 Sentiment Analysis")
    st.markdown("---")

    with st.spinner("Loading data..."):
        df = load_base_data()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Articles", "49,637")
    col2.metric("NLP Model", "FinBERT")
    col3.metric("Labels", "Positive / Neutral / Negative")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        labels  = ["Neutral", "Positive", "Negative"]
        sizes   = [19986, 17842, 11809]
        colors  = ["#f39c12", "#2ecc71", "#e74c3c"]
        ax.pie(sizes, labels=labels, colors=colors,
               autopct="%1.1f%%", startangle=90)
        ax.set_title("FinBERT Sentiment Labels")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Average Daily Sentiment Over Time")
        sentiment_monthly = (
            df.set_index("Date")["Average_Sentiment"]
            .resample("M").mean()
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(sentiment_monthly["Date"],
                sentiment_monthly["Average_Sentiment"],
                color="steelblue", lw=1.5)
        ax.axhline(0, color="gray", linestyle="--", lw=1)
        ax.fill_between(sentiment_monthly["Date"],
                        sentiment_monthly["Average_Sentiment"],
                        0,
                        where=sentiment_monthly["Average_Sentiment"] >= 0,
                        color="green", alpha=0.2)
        ax.fill_between(sentiment_monthly["Date"],
                        sentiment_monthly["Average_Sentiment"],
                        0,
                        where=sentiment_monthly["Average_Sentiment"] < 0,
                        color="red", alpha=0.2)
        ax.set_title("Monthly Average Sentiment Score")
        ax.set_ylabel("Sentiment Score")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.subheader("Sentiment vs Gold Price")
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax2 = ax1.twinx()
    ax1.plot(df["Date"], df["Close"],            color="gold",     lw=1.5, label="Gold Price")
    ax2.plot(df["Date"], df["Average_Sentiment"], color="steelblue", lw=0.8, alpha=0.6, label="Sentiment")
    ax1.set_ylabel("Gold Price (USD)", color="gold")
    ax2.set_ylabel("Sentiment Score",  color="steelblue")
    ax1.set_title("Gold Price vs Daily Sentiment Score")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── FEATURE IMPORTANCE ────────────────────────────────────────────────────────
elif page == "🔍 Feature Importance":
    st.title("🔍 Feature Importance")
    st.markdown("---")

    horizon = st.selectbox("Select prediction horizon", HORIZONS,
                           format_func=lambda h: f"{h}-Day")

    with st.spinner(f"Training model for {horizon}-day horizon... (this may take a minute)"):
        result = run_model(horizon)

    feat_df = (
        pd.DataFrame({"Feature": result["feature_cols"],
                      "Importance": result["importances"]})
        .sort_values("Importance", ascending=False)
    )

    top_n = st.slider("Show top N features", 5, len(feat_df), 15)
    feat_top = feat_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.4)))
    colors = ["#e74c3c" if "Sentiment" in f or "sentiment" in f
              else "#3498db" if "DXY" in f
              else "#2ecc71"
              for f in feat_top["Feature"][::-1]]
    ax.barh(feat_top["Feature"][::-1], feat_top["Importance"][::-1], color=colors)
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_n} Features — {horizon}-Day Horizon")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71",  label="Technical / Market"),
        Patch(facecolor="#3498db",  label="DXY (Macro)"),
        Patch(facecolor="#e74c3c",  label="Sentiment"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.subheader("Full Feature Rankings")
    st.dataframe(feat_df.reset_index(drop=True), use_container_width=True)


# ── MODEL EVALUATION ──────────────────────────────────────────────────────────
elif page == "📉 Model Evaluation":
    st.title("📉 Model Evaluation")
    st.markdown("---")

    horizon = st.selectbox("Select prediction horizon", HORIZONS,
                           format_func=lambda h: f"{h}-Day")

    with st.spinner(f"Training model for {horizon}-day horizon..."):
        result = run_model(horizon)

    y_true = result["y_true"]
    y_prob = result["y_prob"]
    y_pred = (y_prob >= 0.5).astype(int)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",     f"{result['accuracy']:.4f}")
    col2.metric("F1 (weighted)",f"{result['f1']:.4f}")
    col3.metric("Majority Baseline", "0.4400")
    col4.metric("Improvement",  f"+{result['accuracy']-0.44:.4f}")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Down", "Up"],
                    yticklabels=["Down", "Up"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {horizon}d")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve — {horizon}d")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col3:
        st.subheader("Precision–Recall Curve")
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.plot(recall, precision, lw=2, label=f"AP = {ap:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR Curve — {horizon}d")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.subheader("Classification Report")
    report = classification_report(y_true, y_pred,
                                   target_names=["Down", "Up"],
                                   output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)


# ── HORIZON COMPARISON ────────────────────────────────────────────────────────
elif page == "🔭 Horizon Comparison":
    st.title("🔭 Horizon Comparison")
    st.markdown("---")
    st.info("This runs all 4 horizon models. It will take a few minutes — results are cached after first run.")

    if st.button("▶ Run All Horizons", type="primary"):
        all_results = []
        progress = st.progress(0)
        status   = st.empty()

        for i, h in enumerate(HORIZONS):
            status.text(f"Training {h}-day horizon model...")
            result = run_model(h)
            all_results.append(result)
            progress.progress((i + 1) / len(HORIZONS))

        status.text("Done!")
        st.session_state["all_results"] = all_results

    if "all_results" in st.session_state:
        all_results = st.session_state["all_results"]

        horizons = [r["horizon"]  for r in all_results]
        accs     = [r["accuracy"] for r in all_results]
        f1s      = [r["f1"]       for r in all_results]

        # Summary table
        summary_df = pd.DataFrame({
            "Horizon"      : [f"{h}d" for h in horizons],
            "Accuracy"     : [f"{a:.4f}" for a in accs],
            "F1 (weighted)": [f"{f:.4f}" for f in f1s],
            "vs Baseline"  : [f"+{a-0.44:.4f}" for a in accs],
        })
        st.dataframe(summary_df, use_container_width=True)

        st.markdown("---")

        # Comparison bar chart
        x     = np.arange(len(horizons))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        bars1 = ax.bar(x - width/2, accs, width, label="Accuracy",      color="steelblue")
        bars2 = ax.bar(x + width/2, f1s,  width, label="F1 (weighted)", color="coral")
        ax.axhline(0.44, color="gray", linestyle="--", lw=1.2, label="Majority baseline (44%)")
        ax.set_xlabel("Prediction Horizon")
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Across Prediction Horizons")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{h}d" for h in horizons])
        ax.set_ylim(0, 0.75)
        ax.legend()
        for bar in list(bars1) + list(bars2):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Best horizon callout
        best = max(all_results, key=lambda r: r["accuracy"])
        st.success(f"✅ Best horizon: **{best['horizon']}-day** with accuracy **{best['accuracy']:.4f}**")


# ── PREDICTION RESULTS ────────────────────────────────────────────────────────
elif page == "🎯 Prediction Results":
    st.title("🎯 Prediction Results")
    st.markdown("---")

    horizon = st.selectbox("Select prediction horizon", HORIZONS,
                           format_func=lambda h: f"{h}-Day")

    with st.spinner(f"Running model for {horizon}-day horizon..."):
        result = run_model(horizon)

    df_h   = result["df"]
    y_true = result["y_true"]
    y_prob = result["y_prob"]
    y_pred = (y_prob >= 0.5).astype(int)

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{result['accuracy']:.4f}")
    col2.metric("Total Predictions", len(y_true))
    col3.metric("Correct",
                f"{(y_true == y_pred).sum()} / {len(y_true)}")

    st.markdown("---")

    # Actual vs Predicted on gold price (last fold)
    st.subheader("Actual vs Predicted Direction on Gold Price")
    tscv   = TimeSeriesSplit(n_splits=N_SPLITS)
    splits = list(tscv.split(df_h))
    _, test_idx = splits[-1]

    test_df     = df_h.iloc[test_idx].copy()
    n_fold      = len(test_idx)
    y_true_fold = y_true[-n_fold:]
    y_pred_fold = y_pred[-n_fold:]
    correct     = y_true_fold == y_pred_fold

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(test_df["Date"].values, test_df["Close"].values,
            color="gray", lw=1, label="Gold Price")
    ax.scatter(test_df["Date"].values[correct],
               test_df["Close"].values[correct],
               color="green", s=15, label="Correct", zorder=5)
    ax.scatter(test_df["Date"].values[~correct],
               test_df["Close"].values[~correct],
               color="red", s=15, label="Wrong", zorder=5)
    ax.set_title(f"Actual vs Predicted — {horizon}-Day Horizon (Last Fold)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Gold Price (USD)")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.subheader("Prediction Confidence Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(y_prob[y_true == 1], bins=30, alpha=0.6,
            color="green", label="Actual Up")
    ax.hist(y_prob[y_true == 0], bins=30, alpha=0.6,
            color="red",   label="Actual Down")
    ax.axvline(0.5, color="black", linestyle="--", lw=1.5, label="Decision boundary")
    ax.set_xlabel("Predicted Probability (Up)")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Confidence Distribution")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)



# ── PREDICT A DATE ────────────────────────────────────────────────────────────
elif page == "🔮 Predict a Date":
    st.title("🔮 Predict a Date")
    st.markdown("---")
    st.markdown("Pick a date within the dataset range and see what the model predicts.")

    with st.spinner("Loading data..."):
        df = load_base_data()
        dxy_df = load_dxy()
        df, dxy_cols = enrich_with_dxy(df, dxy_df)

    feature_cols = BASE_FEATURES + dxy_cols

    col1, col2 = st.columns(2)
    with col1:
        selected_date = st.date_input(
            "Select a date",
            value=pd.Timestamp("2018-01-02").date(),
            min_value=pd.Timestamp("2015-01-05").date(),
            max_value=pd.Timestamp("2020-12-01").date(),
        )
    with col2:
        horizon = st.selectbox("Prediction horizon", HORIZONS,
                               format_func=lambda h: f"{h}-Day")

    if st.button("▶ Run Prediction", type="primary"):
        selected_ts = pd.Timestamp(selected_date)

        # Find closest available date in dataset
        available = df[df["Date"] <= selected_ts]
        if available.empty:
            st.error("No data available before this date. Try a later date.")
        else:
            row_idx = available.index[-1]
            actual_date = df.loc[row_idx, "Date"]

            if actual_date != selected_ts:
                st.info(f"No trading data for {selected_date} — using closest available date: {actual_date.date()}")

            # Check enough future data exists
            future_idx = row_idx + horizon
            if future_idx >= len(df):
                st.error(f"Not enough future data for a {horizon}-day prediction from this date. Try an earlier date.")
            else:
                # Train model on all data BEFORE selected date (no leakage)
                df_train = make_labels(df.iloc[:row_idx], horizon, LABEL_THRESHOLD)

                if len(df_train) < 100:
                    st.error("Not enough training data before this date. Try a later date.")
                else:
                    X_train = df_train[feature_cols].fillna(0)
                    y_train = df_train["Target"]

                    neg = (y_train == 0).sum()
                    pos = (y_train == 1).sum()
                    spw = neg / pos if pos > 0 else 1.0

                    model = XGBClassifier(
                        n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        scale_pos_weight=spw, eval_metric="logloss",
                        random_state=42, n_jobs=-1,
                    )
                    with st.spinner("Training model on historical data up to selected date..."):
                        model.fit(X_train, y_train, verbose=False)

                    # Predict for selected date
                    X_pred = df.loc[[row_idx], feature_cols].fillna(0)
                    prob   = model.predict_proba(X_pred)[0][1]
                    pred   = 1 if prob >= 0.5 else 0

                    # Actual outcome
                    future_price  = df.loc[future_idx, "Close"]
                    current_price = df.loc[row_idx,    "Close"]
                    actual_return = (future_price - current_price) / current_price
                    actual_label  = 1 if actual_return > LABEL_THRESHOLD else (0 if actual_return < -LABEL_THRESHOLD else None)

                    st.markdown("---")

                    # Prediction result
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Date",          str(actual_date.date()))
                    col2.metric("Gold Price",    f"${current_price:.2f}")
                    col3.metric("Sentiment Score",
                                f"{df.loc[row_idx, 'Average_Sentiment']:.3f}"
                                if "Average_Sentiment" in df.columns else "N/A")

                    st.markdown("---")

                    pred_col, actual_col, conf_col = st.columns(3)

                    with pred_col:
                        if pred == 1:
                            st.success(f"## ↑ Predicted: UP")
                        else:
                            st.error(f"## ↓ Predicted: DOWN")
                        st.caption(f"Over next {horizon} trading days")

                    with conf_col:
                        st.metric("Confidence (Up)", f"{prob:.1%}")
                        st.metric("Confidence (Down)", f"{1-prob:.1%}")

                    with actual_col:
                        if actual_label == 1:
                            st.success(f"## ↑ Actual: UP")
                        elif actual_label == 0:
                            st.error(f"## ↓ Actual: DOWN")
                        else:
                            st.warning(f"## → Actual: FLAT")
                        st.caption(f"Price {actual_date.date()} → {df.loc[future_idx, 'Date'].date()}")

                    st.markdown("---")

                    # Price movement detail
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Price on Selected Date", f"${current_price:.2f}")
                    col2.metric(f"Price after {horizon} Days",
                                f"${future_price:.2f}",
                                delta=f"{actual_return:+.2%}")
                    correct = (pred == actual_label) if actual_label is not None else None
                    if correct is True:
                        col3.success("✅ Prediction was CORRECT")
                    elif correct is False:
                        col3.error("❌ Prediction was WRONG")
                    else:
                        col3.warning("⚠️ Actual move was too small to classify")

                    st.markdown("---")

                    # Mini gold price chart around the selected date
                    st.subheader("Gold Price Around Selected Date")
                    window_start = max(0, row_idx - 30)
                    window_end   = min(len(df), future_idx + 10)
                    df_window    = df.iloc[window_start:window_end]

                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(df_window["Date"], df_window["Close"],
                            color="gold", lw=1.5, label="Gold Price")
                    ax.axvline(actual_date,              color="blue",  linestyle="--", lw=1.5, label="Selected Date")
                    ax.axvline(df.loc[future_idx,"Date"], color="green" if actual_label == 1 else "red",
                               linestyle="--", lw=1.5, label=f"+{horizon} Days")
                    ax.scatter([actual_date], [current_price], color="blue",  s=80, zorder=5)
                    ax.scatter([df.loc[future_idx,"Date"]], [future_price],
                               color="green" if actual_label == 1 else "red", s=80, zorder=5)
                    ax.set_title(f"Gold Price — {actual_date.date()} to +{horizon} days")
                    ax.set_ylabel("Price (USD)")
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)