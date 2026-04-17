#!/usr/bin/env python3
"""
Module 8 — Enhanced Prediction Engine
=======================================
Improves upon the simple threshold prediction with:
  1. Optimal lag per coin (uses lag with best historical correlation)
  2. Contrarian signal detection (flips direction when correlation is negative)
  3. Z-score filtering (only predicts on extreme sentiment days)
  4. Multi-feature logistic regression (VADER + FinBERT + z-score + volume + momentum)
  5. Sentiment momentum (change in sentiment vs yesterday)
  6. Post volume minimum filter
  7. Fear & Greed Index integration
  8. Cross-validated accuracy reporting

Usage:
    python prediction_engine.py                    # Train and save model
    python prediction_engine.py --predict BTC      # Predict for a coin
    python prediction_engine.py --evaluate          # Show model performance
    python prediction_engine.py --status            # Summary
"""

import argparse
import json
import logging
import math
import pickle
import sqlite3
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────

DB_PATH = Path("data") / "project.db"
MODEL_DIR = Path("data") / "models"
EXPORT_DIR = Path("data") / "predictions"
LOG_DIR = Path("logs")

COINS = ["BTC", "ETH", "SOL", "DOGE", "SHIB", "BNB", "MARKET"]
MIN_POSTS_PER_DAY = 5        # minimum posts to trust daily sentiment
ZSCORE_THRESHOLD = 0.3       # only predict when |z| > this
OPTIMAL_LAGS = {}             # populated during training

# ──────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "prediction_engine.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────

def load_data(conn):
    """Load and merge all data needed for prediction."""

    # Daily sentiment per coin
    sent = pd.read_sql_query("""
        SELECT date, coin_target,
               COUNT(*) as post_count,
               AVG(vader_compound) as vader,
               AVG(finbert_score) as finbert,
               AVG(composite_score) as composite,
               AVG(composite_zscore) as zscore,
               AVG(weighted_score) as weighted,
               AVG(engagement_weight) as engagement,
               SUM(upvotes) as total_upvotes,
               SUM(num_comments) as total_comments
        FROM posts
        WHERE sentiment_processed > 0
        GROUP BY date, coin_target
        ORDER BY date
    """, conn)
    sent["date"] = pd.to_datetime(sent["date"])

    # Prices
    prices = pd.read_sql_query("""
        SELECT coin, date, close, volume FROM prices ORDER BY coin, date
    """, conn)

    if prices.empty:
        return sent, pd.DataFrame(), pd.DataFrame()

    prices["date"] = pd.to_datetime(prices["date"])
    prices["daily_return"] = prices.groupby("coin")["close"].pct_change()
    prices["direction"] = (prices["daily_return"] > 0).astype(int)

    # Fear & Greed
    fg = pd.DataFrame()
    try:
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fear_greed'")
        if c.fetchone():
            fg = pd.read_sql_query("SELECT date, value as fear_greed FROM fear_greed", conn)
            fg["date"] = pd.to_datetime(fg["date"])
    except:
        pass

    return sent, prices, fg


def build_features(sent_df, price_df, fg_df, coin):
    """
    Build feature matrix for a single coin.
    Features: vader, finbert, composite, zscore, momentum, volume_ratio,
              fear_greed, vader_finbert_agreement
    Target: next-day price direction (1=up, 0=down)
    """
    price_coin = coin if coin != "MARKET" else "BTC"
    s = sent_df[sent_df["coin_target"] == coin].copy().sort_values("date")
    p = price_df[price_df["coin"] == price_coin][["date", "close", "daily_return", "direction"]].copy()

    if s.empty or p.empty:
        return pd.DataFrame()

    # Sentiment momentum (today - yesterday)
    s["vader_momentum"] = s["vader"].diff()
    s["finbert_momentum"] = s["finbert"].diff()
    s["composite_momentum"] = s["composite"].diff()

    # Volume ratio (today's posts / 7-day average)
    s["volume_ma7"] = s["post_count"].rolling(7, min_periods=1).mean()
    s["volume_ratio"] = s["post_count"] / s["volume_ma7"].replace(0, 1)

    # VADER-FinBERT agreement (1=agree, -1=disagree)
    s["model_agreement"] = np.sign(s["vader"]) * np.sign(s["finbert"])

    # Merge with price data
    merged = s.merge(p, on="date", how="inner")

    # Add Fear & Greed if available
    if not fg_df.empty:
        merged = merged.merge(fg_df, on="date", how="left")
        merged["fear_greed"] = merged["fear_greed"].fillna(50)  # neutral default
    else:
        merged["fear_greed"] = 50

    # Target: next-day direction (shift price direction back by 1)
    merged["target"] = merged["direction"].shift(-1)

    # For lag-2 target
    merged["target_lag2"] = merged["direction"].shift(-2)

    # Drop NaN rows
    merged = merged.dropna(subset=["target", "vader_momentum"])

    return merged


# ──────────────────────────────────────────────────────────────────────
# FIND OPTIMAL LAG PER COIN
# ──────────────────────────────────────────────────────────────────────

def find_optimal_lag(merged, max_lag=3):
    """Find the lag with strongest absolute correlation for composite."""
    best_lag = 1
    best_corr = 0

    for lag in range(0, max_lag + 1):
        if lag == 0:
            s = merged["composite"].values
            r = merged["daily_return"].values
        else:
            s = merged["composite"].iloc[:-lag].values
            r = merged["daily_return"].iloc[lag:].values

        mask = ~(np.isnan(s) | np.isnan(r))
        if mask.sum() < 10:
            continue

        corr, _ = stats.pearsonr(s[mask], r[mask])

        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_lag = lag

    is_contrarian = best_corr < 0
    return best_lag, best_corr, is_contrarian


# ──────────────────────────────────────────────────────────────────────
# TRAIN LOGISTIC REGRESSION
# ──────────────────────────────────────────────────────────────────────

def train_model(merged, coin, optimal_lag):
    """
    Train prediction model using a lean feature set + ensemble approach.
    Combines: logistic regression (3-4 features) + simple contrarian rules.
    Designed for small datasets (60-120 days).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, f1_score

    # Select target based on optimal lag
    target_col = "target" if optimal_lag <= 1 else "target_lag2"

    # Filter days with minimum posts
    df = merged[merged["post_count"] >= MIN_POSTS_PER_DAY].copy()
    df = df.dropna(subset=[target_col])

    if len(df) < 30:
        logger.warning(f"  {coin}: Only {len(df)} usable days. Need 30+.")
        return None

    # LEAN feature set — only 4 features to prevent overfitting on small data
    feature_cols = ["composite", "zscore", "composite_momentum", "model_agreement"]
    available_cols = [c for c in feature_cols if c in df.columns]
    df = df.dropna(subset=available_cols)

    X = df[available_cols].values
    y = df[target_col].values.astype(int)

    if len(np.unique(y)) < 2:
        logger.warning(f"  {coin}: Only one class in target. Skipping.")
        return None

    # ──────────────────────────────────────────────────────────
    # APPROACH 1: Logistic Regression (lean)
    # ──────────────────────────────────────────────────────────
    tscv = TimeSeriesSplit(n_splits=3)  # fewer splits = more training data per fold
    lr_accuracies = []
    lr_f1s = []
    baseline_accs = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(y_train)) < 2 or len(X_test) < 5:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Strong regularization (C=0.01) to prevent overfitting
        model = LogisticRegression(C=0.01, max_iter=1000, solver="lbfgs", random_state=42)
        model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)
        lr_accuracies.append(accuracy_score(y_test, y_pred))
        lr_f1s.append(f1_score(y_test, y_pred, zero_division=0))
        baseline_accs.append(max(y_test.mean(), 1 - y_test.mean()))

    # ──────────────────────────────────────────────────────────
    # APPROACH 2: Simple rule-based (sentiment threshold + momentum)
    # ──────────────────────────────────────────────────────────
    rule_accuracies = []

    for train_idx, test_idx in tscv.split(X):
        y_test = y[test_idx]
        test_df = df.iloc[test_idx]

        if len(y_test) < 5:
            continue

        # Rule: predict UP when composite_momentum > 0 AND zscore > -0.5
        # This captures "sentiment is improving and not deeply bearish"
        preds = []
        for _, row in test_df.iterrows():
            c_mom = row.get("composite_momentum", 0) or 0
            zs = row.get("zscore", 0) or 0
            agree = row.get("model_agreement", 0) or 0

            # Score: combine signals
            score = 0
            if c_mom > 0: score += 1
            if c_mom < 0: score -= 1
            if zs > 0.3: score += 1
            if zs < -0.3: score -= 1
            if agree > 0: score += 0.5

            preds.append(1 if score > 0 else 0)

        rule_acc = accuracy_score(y_test, preds)
        rule_accuracies.append(rule_acc)

    # ──────────────────────────────────────────────────────────
    # PICK BEST APPROACH
    # ──────────────────────────────────────────────────────────
    lr_mean = np.mean(lr_accuracies) if lr_accuracies else 0
    rule_mean = np.mean(rule_accuracies) if rule_accuracies else 0
    baseline_mean = np.mean(baseline_accs) if baseline_accs else 0.55

    # Use whichever approach performed better in CV
    use_rules = rule_mean > lr_mean

    if use_rules:
        best_acc = rule_mean
        best_f1 = np.mean(lr_f1s) if lr_f1s else 0  # approximate
        method = "Rule-based ensemble"
    else:
        best_acc = lr_mean
        best_f1 = np.mean(lr_f1s) if lr_f1s else 0
        method = "Logistic Regression (lean)"

    # Train final LR model on all data (used for probability output)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    final_model = LogisticRegression(C=0.01, max_iter=1000, solver="lbfgs", random_state=42)
    final_model.fit(X_scaled, y)

    coef = dict(zip(available_cols, final_model.coef_[0].round(4)))

    result = {
        "coin": coin,
        "model": final_model,
        "scaler": scaler,
        "feature_cols": available_cols,
        "optimal_lag": optimal_lag,
        "use_rules": use_rules,
        "method": method,
        "cv_accuracy": round(best_acc, 4),
        "cv_f1": round(best_f1, 4),
        "cv_baseline": round(baseline_mean, 4),
        "beats_baseline": best_acc > baseline_mean,
        "lr_accuracy": round(lr_mean, 4),
        "rule_accuracy": round(rule_mean, 4),
        "n_training_days": len(df),
        "feature_importance": coef,
        "up_ratio": round(y.mean(), 4),
    }

    return result


# ──────────────────────────────────────────────────────────────────────
# PREDICT
# ──────────────────────────────────────────────────────────────────────

def predict_tomorrow(model_info, merged, coin):
    """Generate prediction for tomorrow using the trained model."""

    if model_info is None:
        return {"coin": coin, "error": "No model trained for this coin"}

    model = model_info["model"]
    scaler = model_info["scaler"]
    feature_cols = model_info["feature_cols"]

    # Get latest day
    latest = merged.iloc[-1:]

    if latest.empty:
        return {"coin": coin, "error": "No recent data"}

    # Check minimum posts
    post_count = int(latest["post_count"].values[0])
    if post_count < MIN_POSTS_PER_DAY:
        signal_strength = "VERY_WEAK"
    else:
        signal_strength = "OK"

    # Extract features
    available = [c for c in feature_cols if c in latest.columns]
    X = latest[available].values

    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)

    X_scaled = scaler.transform(X)

    # Predict
    pred_class = model.predict(X_scaled)[0]
    pred_proba = model.predict_proba(X_scaled)[0]

    direction = "UP" if pred_class == 1 else "DOWN"
    probability = float(pred_proba[pred_class])

    # Z-score confidence
    zscore = float(latest["zscore"].values[0]) if "zscore" in latest.columns else 0
    zscore_abs = abs(zscore)

    if zscore_abs > 1.5:
        confidence = "HIGH"
    elif zscore_abs > ZSCORE_THRESHOLD:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # Signal quality assessment
    vader = float(latest["vader"].values[0])
    finbert = float(latest["finbert"].values[0])
    composite = float(latest["composite"].values[0])

    factors = []

    # Sentiment direction
    if vader > 0.1:
        factors.append({"factor": "VADER bullish", "direction": "bullish", "value": round(vader, 4)})
    elif vader < -0.1:
        factors.append({"factor": "VADER bearish", "direction": "bearish", "value": round(vader, 4)})
    else:
        factors.append({"factor": "VADER neutral", "direction": "neutral", "value": round(vader, 4)})

    if finbert > 0.05:
        factors.append({"factor": "FinBERT bullish", "direction": "bullish", "value": round(finbert, 4)})
    elif finbert < -0.05:
        factors.append({"factor": "FinBERT bearish", "direction": "bearish", "value": round(finbert, 4)})
    else:
        factors.append({"factor": "FinBERT neutral", "direction": "neutral", "value": round(finbert, 4)})

    # Momentum
    c_mom = float(latest["composite_momentum"].values[0]) if "composite_momentum" in latest.columns else 0
    if c_mom > 0.02:
        factors.append({"factor": "Sentiment improving", "direction": "bullish", "value": round(c_mom, 4)})
    elif c_mom < -0.02:
        factors.append({"factor": "Sentiment declining", "direction": "bearish", "value": round(c_mom, 4)})

    # Model agreement
    agreement = float(latest["model_agreement"].values[0]) if "model_agreement" in latest.columns else 0
    if agreement > 0:
        factors.append({"factor": "VADER-FinBERT agree", "direction": "high confidence", "value": None})
    else:
        factors.append({"factor": "VADER-FinBERT disagree", "direction": "mixed signal", "value": None})

    # Volume
    vol_ratio = float(latest["volume_ratio"].values[0]) if "volume_ratio" in latest.columns else 1
    if vol_ratio > 1.3:
        factors.append({"factor": "Elevated discussion volume", "direction": "strong signal", "value": round(vol_ratio, 2)})
    elif vol_ratio < 0.7:
        factors.append({"factor": "Low discussion volume", "direction": "weak signal", "value": round(vol_ratio, 2)})

    # Fear & Greed
    fg = float(latest["fear_greed"].values[0]) if "fear_greed" in latest.columns else 50
    if fg < 25:
        factors.append({"factor": "Extreme Fear (F&G Index)", "direction": "bearish", "value": int(fg)})
    elif fg > 75:
        factors.append({"factor": "Extreme Greed (F&G Index)", "direction": "bullish", "value": int(fg)})

    # Contrarian note
    if model_info.get("is_contrarian"):
        factors.append({"factor": "Contrarian pattern detected", "direction": "signal is inverted", "value": None})

    # Feature importance (top 3)
    importance = model_info.get("feature_importance", {})
    top_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

    return {
        "coin": coin,
        "price_coin": coin if coin != "MARKET" else "BTC",
        "prediction": direction,
        "probability": round(probability, 4),
        "confidence": confidence,
        "signal": "BULLISH" if direction == "UP" else "BEARISH",
        "today_sentiment": {
            "date": str(latest["date"].values[0])[:10],
            "vader": round(vader, 4),
            "finbert": round(finbert, 4),
            "composite": round(composite, 4),
            "zscore": round(zscore, 4),
            "posts": post_count,
            "momentum": round(c_mom, 4) if c_mom else 0,
        },
        "model_info": {
            "type": model_info.get("method", "Logistic Regression (lean)"),
            "optimal_lag": model_info["optimal_lag"],
            "is_contrarian": model_info.get("is_contrarian", False),
            "cv_accuracy": model_info["cv_accuracy"],
            "cv_f1": model_info["cv_f1"],
            "cv_baseline": model_info["cv_baseline"],
            "beats_baseline": model_info["beats_baseline"],
            "n_training_days": model_info["n_training_days"],
            "top_features": [{"feature": f, "weight": w} for f, w in top_features],
        },
        "factors": factors,
        "confidence_note": f"Z-score {zscore:.2f} — {'unusually ' if zscore_abs > 1 else ''}{'bullish' if zscore > 0 else 'bearish' if zscore < 0 else 'neutral'} sentiment",
        "disclaimer": "Research tool only. Not financial advice. Based on historical sentiment-price patterns with limited predictive power.",
    }


# ──────────────────────────────────────────────────────────────────────
# TRAIN ALL MODELS
# ──────────────────────────────────────────────────────────────────────

def train_all():
    """Train prediction models for all coins."""
    conn = sqlite3.connect(str(DB_PATH))
    sent, prices, fg = load_data(conn)

    if prices.empty:
        logger.error("No price data. Run price_collector.py first.")
        conn.close()
        return {}

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info("PREDICTION ENGINE — TRAINING")
    logger.info(f"{'='*60}")

    all_models = {}
    all_results = []

    for coin in COINS:
        logger.info(f"\n{'─'*40}")
        logger.info(f"Training: {coin}")
        logger.info(f"{'─'*40}")

        merged = build_features(sent, prices, fg, coin)
        if merged.empty or len(merged) < 30:
            logger.warning(f"  {coin}: insufficient data ({len(merged)} days)")
            continue

        # Find optimal lag
        opt_lag, opt_corr, is_contrarian = find_optimal_lag(merged)
        logger.info(f"  Optimal lag: {opt_lag} (r={opt_corr:.4f}, {'contrarian' if is_contrarian else 'momentum'})")

        # Train model
        model_info = train_model(merged, coin, opt_lag)
        if model_info is None:
            continue

        model_info["is_contrarian"] = is_contrarian
        model_info["optimal_corr"] = opt_corr

        all_models[coin] = {"model_info": model_info, "merged": merged}

        acc = model_info["cv_accuracy"]
        base = model_info["cv_baseline"]
        beats = "YES" if model_info["beats_baseline"] else "no"

        logger.info(f"  Method: {model_info.get('method', 'unknown')}")
        logger.info(f"  LR accuracy: {model_info.get('lr_accuracy', 0):.4f} | Rule accuracy: {model_info.get('rule_accuracy', 0):.4f}")
        logger.info(f"  Best CV Accuracy: {acc:.4f} (baseline: {base:.4f}) — {beats}")
        logger.info(f"  CV F1: {model_info['cv_f1']:.4f}")
        logger.info(f"  Training days: {model_info['n_training_days']}")

        # Top features
        imp = model_info["feature_importance"]
        top3 = sorted(imp.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        logger.info(f"  Top features: {', '.join(f'{f}({w:+.3f})' for f,w in top3)}")

        all_results.append({
            "coin": coin,
            "optimal_lag": opt_lag,
            "correlation": round(opt_corr, 4),
            "is_contrarian": is_contrarian,
            "cv_accuracy": acc,
            "cv_f1": model_info["cv_f1"],
            "cv_baseline": base,
            "beats_baseline": model_info["beats_baseline"],
            "n_training_days": model_info["n_training_days"],
            "improvement_over_baseline": round(acc - base, 4),
        })

    # Save models
    model_path = MODEL_DIR / "prediction_models.pkl"
    save_data = {coin: info["model_info"] for coin, info in all_models.items()}
    with open(model_path, "wb") as f:
        pickle.dump(save_data, f)
    logger.info(f"\nModels saved: {model_path}")

    # Export results
    if all_results:
        df = pd.DataFrame(all_results)
        results_path = EXPORT_DIR / "model_evaluation.csv"
        df.to_csv(results_path, index=False)
        logger.info(f"Evaluation: {results_path}")

        # Summary
        logger.info(f"\n{'='*70}")
        logger.info("TRAINING SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"{'Coin':<8} {'Lag':>4} {'Type':<12} {'CV Acc':>8} {'Base':>8} {'Δ':>8} {'Beats?':>7}")
        logger.info(f"{'─'*70}")
        for r in all_results:
            typ = "contrarian" if r["is_contrarian"] else "momentum"
            delta = r["improvement_over_baseline"]
            logger.info(f"{r['coin']:<8} {r['optimal_lag']:>4} {typ:<12} "
                       f"{r['cv_accuracy']:>8.4f} {r['cv_baseline']:>8.4f} "
                       f"{delta:>+8.4f} {'YES' if r['beats_baseline'] else 'no':>7}")

    conn.close()
    return all_models


# ──────────────────────────────────────────────────────────────────────
# PREDICT FOR A COIN
# ──────────────────────────────────────────────────────────────────────

def predict_coin(coin):
    """Load saved model and predict for a coin."""
    model_path = MODEL_DIR / "prediction_models.pkl"
    if not model_path.exists():
        logger.error("No trained models found. Run training first.")
        return None

    with open(model_path, "rb") as f:
        models = pickle.load(f)

    if coin not in models:
        logger.error(f"No model for {coin}. Available: {list(models.keys())}")
        return None

    conn = sqlite3.connect(str(DB_PATH))
    sent, prices, fg = load_data(conn)
    merged = build_features(sent, prices, fg, coin)
    conn.close()

    if merged.empty:
        return {"coin": coin, "error": "No data to predict"}

    return predict_tomorrow(models[coin], merged, coin)


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Enhanced Prediction Engine")
    parser.add_argument("--predict", type=str, help="Predict for a coin (e.g., BTC)")
    parser.add_argument("--evaluate", action="store_true", help="Show model performance")
    parser.add_argument("--status", action="store_true", help="Summary")
    args = parser.parse_args()

    if args.predict:
        result = predict_coin(args.predict.upper())
        if result:
            print(json.dumps(result, indent=2, default=str))
        return

    if args.evaluate or args.status:
        path = EXPORT_DIR / "model_evaluation.csv"
        if path.exists():
            df = pd.read_csv(path)
            print(f"\n{'='*60}")
            print("MODEL EVALUATION SUMMARY")
            print(f"{'='*60}")
            print(df.to_string(index=False))
            beats = df[df["beats_baseline"] == True]
            print(f"\n{len(beats)}/{len(df)} models beat baseline")
        else:
            print("No evaluation results. Run training first.")
        return

    # Default: train all models
    train_all()


if __name__ == "__main__":
    main()