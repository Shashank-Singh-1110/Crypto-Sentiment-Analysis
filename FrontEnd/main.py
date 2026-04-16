import argparse
import json
import logging
import os
import sqlite3
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, request, send_from_directory

DB_PATH = Path("data") / "project.db"
STATIC_DIR = Path("static")
MODULES_DIR = Path("modules")
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

app = Flask(__name__, static_folder=str(STATIC_DIR))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
refresh_status = {"running": False, "last_run": None, "message": ""}


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def safe_json(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return round(float(obj), 6)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if pd.isna(obj):
        return None
    return obj


def has_table(conn, table_name):
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return c.fetchone() is not None


def has_column(conn, table_name, col_name):
    c = conn.cursor()
    c.execute(f"PRAGMA table_info({table_name})")
    return col_name in {row[1] for row in c.fetchall()}

@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(str(STATIC_DIR), filename)


@app.route("/api/stats")
def api_stats():
    conn = get_db()
    c = conn.cursor()

    stats = {}
    c.execute("SELECT COUNT(*) FROM posts")
    stats["total_posts"] = c.fetchone()[0]

    c.execute("SELECT MIN(date), MAX(date) FROM posts")
    row = c.fetchone()
    stats["date_min"] = row[0]
    stats["date_max"] = row[1]
    stats["days_covered"] = 0

    c.execute("SELECT COUNT(DISTINCT date) FROM posts")
    stats["days_covered"] = c.fetchone()[0]

    c.execute("SELECT COUNT(DISTINCT subreddit) FROM posts")
    stats["subreddits"] = c.fetchone()[0]

    per_coin = {}
    for row in c.execute("SELECT coin_target, COUNT(*) FROM posts GROUP BY coin_target"):
        per_coin[row[0]] = row[1]
    stats["per_coin"] = per_coin

    if has_column(conn, "posts", "sentiment_processed"):
        c.execute("SELECT COUNT(*) FROM posts WHERE sentiment_processed > 0")
        stats["sentiment_scored"] = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM posts WHERE sentiment_processed = 1")
        stats["finbert_scored"] = c.fetchone()[0]
    else:
        stats["sentiment_scored"] = 0
        stats["finbert_scored"] = 0

    if has_column(conn, "posts", "nlp_processed"):
        c.execute("SELECT COUNT(*) FROM posts WHERE nlp_processed = 1")
        stats["nlp_processed"] = c.fetchone()[0]
    else:
        stats["nlp_processed"] = 0

    if has_column(conn, "posts", "topics_processed"):
        c.execute("SELECT COUNT(DISTINCT topic_id) FROM posts WHERE topics_processed = 1 AND topic_id != -1")
        stats["topics_discovered"] = c.fetchone()[0]
    else:
        stats["topics_discovered"] = 0

    stats["refresh"] = refresh_status

    conn.close()
    return jsonify(stats)


@app.route("/api/sentiment/<coin>")
def api_sentiment(coin):
    conn = get_db()
    df = pd.read_sql_query("""
        SELECT
            date,
            COUNT(*) as posts,
            AVG(vader_compound) as vader,
            AVG(finbert_score) as finbert,
            AVG(composite_score) as composite,
            AVG(composite_zscore) as zscore,
            AVG(weighted_score) as weighted,
            SUM(upvotes) as upvotes,
            SUM(num_comments) as comments
        FROM posts
        WHERE coin_target = ? AND sentiment_processed > 0
        GROUP BY date ORDER BY date
    """, conn, params=[coin])
    conn.close()

    return jsonify({
        "coin": coin,
        "dates": df["date"].tolist(),
        "posts": df["posts"].tolist(),
        "vader": [safe_json(v) for v in df["vader"]],
        "finbert": [safe_json(v) for v in df["finbert"]],
        "composite": [safe_json(v) for v in df["composite"]],
        "zscore": [safe_json(v) for v in df["zscore"]],
        "weighted": [safe_json(v) for v in df["weighted"]],
        "upvotes": [safe_json(v) for v in df["upvotes"]],
        "comments": [safe_json(v) for v in df["comments"]],
    })

@app.route("/api/prices/<coin>")
def api_prices(coin):
    conn = get_db()

    if not has_table(conn, "prices"):
        conn.close()
        return jsonify({"error": "no price data", "dates": [], "close": []})

    df = pd.read_sql_query("""
        SELECT date, open, high, low, close, volume
        FROM prices WHERE coin = ? ORDER BY date
    """, conn, params=[coin])
    conn.close()

    if df.empty:
        return jsonify({"coin": coin, "dates": [], "close": []})

    df["daily_return"] = df["close"].pct_change()

    return jsonify({
        "coin": coin,
        "dates": df["date"].tolist(),
        "open": [safe_json(v) for v in df["open"]],
        "high": [safe_json(v) for v in df["high"]],
        "low": [safe_json(v) for v in df["low"]],
        "close": [safe_json(v) for v in df["close"]],
        "volume": [safe_json(v) for v in df["volume"]],
        "daily_return": [safe_json(v) for v in df["daily_return"]],
    })

@app.route("/api/ablation")
def api_ablation():
    path = Path("data") / "correlation" / "ablation_study.csv"
    if not path.exists():
        return jsonify({"error": "Run correlation_engine.py first", "data": []})

    df = pd.read_csv(path)
    records = []
    for _, row in df.iterrows():
        records.append({k: safe_json(v) for k, v in row.to_dict().items()})
    return jsonify({"data": records})

@app.route("/api/emotions/<coin>")
def api_emotions(coin):
    conn = get_db()

    if not has_column(conn, "posts", "emotion_label"):
        conn.close()
        return jsonify({"coin": coin, "distribution": {}, "timeline": []})
    c = conn.cursor()
    c.execute("""
        SELECT emotion_label, COUNT(*) FROM posts
        WHERE coin_target = ? AND emotions_processed = 1
        GROUP BY emotion_label ORDER BY COUNT(*) DESC
    """, (coin,))
    distribution = {row[0]: row[1] for row in c.fetchall()}
    df = pd.read_sql_query("""
        SELECT date, emotion_label, COUNT(*) as count
        FROM posts
        WHERE coin_target = ? AND emotions_processed = 1
        GROUP BY date, emotion_label ORDER BY date
    """, conn, params=[coin])
    conn.close()

    if not df.empty:
        pivot = df.pivot_table(index="date", columns="emotion_label",
                               values="count", fill_value=0).reset_index()
        timeline = pivot.to_dict(orient="records")
    else:
        timeline = []

    return jsonify({"coin": coin, "distribution": distribution, "timeline": timeline})

@app.route("/api/ner/top")
def api_ner_top():
    path = Path("data") / "advanced_nlp" / "ner_entity_summary.csv"
    if not path.exists():
        return jsonify({"entities": []})

    df = pd.read_csv(path)
    entities = []
    for _, row in df.head(50).iterrows():
        entities.append({"entity": row["entity"], "type": row["type"], "count": int(row["count"])})

    return jsonify({"entities": entities})

@app.route("/api/topics/trends")
def api_topics():
    path = Path("data") / "advanced_nlp" / "topic_trends_daily.csv"
    if not path.exists():
        return jsonify({"trends": []})

    df = pd.read_csv(path)
    top = df.groupby("topic_id")["post_count"].sum().nlargest(10).index.tolist()
    filtered = df[df["topic_id"].isin(top)]

    trends = []
    for _, row in filtered.iterrows():
        trends.append({
            "date": row["date"],
            "topic_id": int(row["topic_id"]),
            "topic_label": row.get("topic_label", f"topic_{row['topic_id']}"),
            "count": int(row["post_count"]),
        })

    return jsonify({"trends": trends})


@app.route("/api/correlations")
def api_correlations():
    results = {}
    path = Path("data") / "correlation" / "pearson_correlations.csv"
    if path.exists():
        df = pd.read_csv(path)
        results["pearson"] = [{k: safe_json(v) for k, v in row.to_dict().items()} for _, row in df.iterrows()]
    path = Path("data") / "correlation" / "granger_causality.csv"
    if path.exists():
        df = pd.read_csv(path)
        results["granger"] = [{k: safe_json(v) for k, v in row.to_dict().items()} for _, row in df.iterrows()]
    path = Path("data") / "correlation" / "correlation_heatmap_data.csv"
    if path.exists():
        df = pd.read_csv(path, index_col=0)
        results["heatmap"] = {
            "coins": df.index.tolist(),
            "metrics": df.columns.tolist(),
            "values": [[safe_json(v) for v in row] for row in df.values],
        }

    return jsonify(results)


@app.route("/api/insight", methods=["POST"])
def api_insight():
    data = request.get_json()
    user_query = data.get("query", "")
    coin = data.get("coin", "BTC")

    if not user_query:
        return jsonify({"error": "No query provided"})
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT date, AVG(vader_compound), AVG(finbert_score), AVG(composite_score), COUNT(*)
        FROM posts WHERE coin_target = ? AND sentiment_processed > 0
        GROUP BY date ORDER BY date DESC LIMIT 7
    """, (coin,))
    recent_sent = c.fetchall()
    emotion_ctx = ""
    if has_column(conn, "posts", "emotion_label"):
        c.execute("""
            SELECT emotion_label, COUNT(*) FROM posts
            WHERE coin_target = ? AND emotions_processed = 1
            GROUP BY emotion_label ORDER BY COUNT(*) DESC LIMIT 5
        """, (coin,))
        emotions = c.fetchall()
        if emotions:
            emotion_ctx = "Emotion distribution: " + ", ".join(f"{e[0]}({e[1]})" for e in emotions)
    price_ctx = ""
    if has_table(conn, "prices"):
        price_coin = coin if coin != "MARKET" else "BTC"
        c.execute("""
            SELECT date, close FROM prices WHERE coin = ?
            ORDER BY date DESC LIMIT 7
        """, (price_coin,))
        prices = c.fetchall()
        if prices:
            price_ctx = f"Recent {price_coin} prices: " + ", ".join(
                f"{p[0]}=${p[1]:,.2f}" for p in reversed(prices))

    conn.close()
    sent_ctx = ""
    if recent_sent:
        sent_ctx = f"Recent {coin} sentiment (last 7 days):\n"
        for row in reversed(recent_sent):
            sent_ctx += f"  {row[0]}: VADER={row[1]:.3f}, FinBERT={row[2]:.3f}, Composite={row[3]:.3f}, Posts={row[4]}\n"

    prompt = f"""You are a crypto market research analyst. Analyze the following data and answer the user's question.
Be concise, factual, and data-driven. Do NOT give financial advice. Frame insights as research observations.

CONTEXT:
{sent_ctx}
{price_ctx}
{emotion_ctx}

USER QUESTION: {user_query}

Provide a clear, structured analysis in 3-5 sentences."""

    # Call Ollama
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 300}
        }, timeout=60)

        if resp.status_code == 200:
            result = resp.json()
            return jsonify({"insight": result.get("response", "No response from model.")})
        else:
            return jsonify({"error": f"Ollama returned status {resp.status_code}: {resp.text[:200]}"})

    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot connect to Ollama. Make sure it's running: ollama serve"})
    except requests.exceptions.Timeout:
        return jsonify({"error": "Ollama timed out. The model might still be loading."})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/predict/<coin>")
def api_predict(coin):
    import math
    conn = get_db()
    c = conn.cursor()

    price_coin = coin if coin != "MARKET" else "BTC"
    result = {
        "coin": coin, "price_coin": price_coin,
        "prediction": None, "probability": None,
        "confidence": None, "signal": None,
        "today_sentiment": {}, "historical_accuracy": {},
        "factors": []
    }

    c.execute("""
              SELECT date, AVG (vader_compound) as vader, AVG (finbert_score) as finbert, AVG (composite_score) as composite, AVG (composite_zscore) as zscore, COUNT (*) as posts
              FROM posts
              WHERE coin_target = ? AND sentiment_processed > 0
              GROUP BY date
              ORDER BY date DESC LIMIT 1
              """, (coin,))
    today = c.fetchone()

    if not today:
        conn.close()
        result["error"] = "No sentiment data available"
        return jsonify(result)

    today_date = today[0]
    today_vader = today[1] or 0
    today_finbert = today[2] or 0
    today_composite = today[3] or 0
    today_zscore = today[4] or 0
    today_posts = today[5] or 0

    result["today_sentiment"] = {
        "date": today_date,
        "vader": safe_json(today_vader),
        "finbert": safe_json(today_finbert),
        "composite": safe_json(today_composite),
        "zscore": safe_json(today_zscore),
        "posts": today_posts,
    }

    if not has_table(conn, "prices"):
        conn.close()
        result["error"] = "No price data available"
        return jsonify(result)

    sent_df = pd.read_sql_query("""
                                SELECT date, AVG (composite_score) as composite, AVG (vader_compound) as vader, AVG (finbert_score) as finbert
                                FROM posts
                                WHERE coin_target = ? AND sentiment_processed > 0
                                GROUP BY date
                                ORDER BY date
                                """, conn, params=[coin])

    price_df = pd.read_sql_query("""
                                 SELECT date, close
                                 FROM prices
                                 WHERE coin = ?
                                 ORDER BY date
                                 """, conn, params=[price_coin])

    if sent_df.empty or price_df.empty or len(price_df) < 5:
        conn.close()
        result["error"] = "Insufficient data for prediction"
        return jsonify(result)

    price_df["next_return"] = price_df["close"].pct_change().shift(-1)
    price_df["next_up"] = (price_df["next_return"] > 0).astype(int)

    merged = sent_df.merge(price_df[["date", "next_return", "next_up", "close"]], on="date", how="inner")
    merged = merged.dropna(subset=["next_return"])

    if len(merged) < 10:
        conn.close()
        result["error"] = f"Only {len(merged)} matched days — need at least 10"
        return jsonify(result)

    if today_composite > 0:
        same_dir = merged[merged["composite"] > 0]
        predicted_dir = "UP"
    else:
        same_dir = merged[merged["composite"] <= 0]
        predicted_dir = "DOWN"

    if len(same_dir) >= 3:
        if predicted_dir == "UP":
            accuracy = same_dir["next_up"].mean()
        else:
            accuracy = 1 - same_dir["next_up"].mean()
        hit_count = int(len(same_dir) * accuracy)
    else:
        overall_up = merged["next_up"].mean()
        accuracy = max(overall_up, 1 - overall_up)
        hit_count = int(len(merged) * accuracy)
        same_dir = merged

    zscore_abs = abs(today_zscore)
    if zscore_abs > 1.5:
        confidence = "HIGH"
        confidence_note = f"Z-score {today_zscore:.2f} — sentiment is unusually {'bullish' if today_zscore > 0 else 'bearish'}"
    elif zscore_abs > 0.5:
        confidence = "MEDIUM"
        confidence_note = f"Z-score {today_zscore:.2f} — moderately {'bullish' if today_zscore > 0 else 'bearish'} sentiment"
    else:
        confidence = "LOW"
        confidence_note = f"Z-score {today_zscore:.2f} — sentiment is near average"

    factors = []
    if today_vader > 0.1:
        factors.append({"factor": "VADER positive", "direction": "bullish", "value": safe_json(today_vader)})
    elif today_vader < -0.1:
        factors.append({"factor": "VADER negative", "direction": "bearish", "value": safe_json(today_vader)})

    if today_finbert > 0.05:
        factors.append({"factor": "FinBERT positive", "direction": "bullish", "value": safe_json(today_finbert)})
    elif today_finbert < -0.05:
        factors.append({"factor": "FinBERT negative", "direction": "bearish", "value": safe_json(today_finbert)})

    if today_posts > merged["posts"].median() if "posts" in merged.columns else 50:
        factors.append({"factor": "High post volume", "direction": "strong signal", "value": today_posts})
    else:
        factors.append({"factor": "Low post volume", "direction": "weak signal", "value": today_posts})

    if (today_vader > 0 and today_finbert > 0) or (today_vader < 0 and today_finbert < 0):
        factors.append({"factor": "VADER-FinBERT agreement", "direction": "high confidence", "value": None})
    else:
        factors.append({"factor": "VADER-FinBERT disagreement", "direction": "mixed signal", "value": None})

    c.execute("SELECT close, date FROM prices WHERE coin = ? ORDER BY date DESC LIMIT 1", (price_coin,))
    latest_price_row = c.fetchone()
    latest_price = latest_price_row[0] if latest_price_row else None
    latest_price_date = latest_price_row[1] if latest_price_row else None

    conn.close()
    result["prediction"] = predicted_dir
    result["probability"] = safe_json(round(accuracy, 4))
    result["confidence"] = confidence
    result["confidence_note"] = confidence_note
    result["signal"] = "BULLISH" if predicted_dir == "UP" else "BEARISH"
    result["factors"] = factors
    result["latest_price"] = safe_json(latest_price)
    result["latest_price_date"] = latest_price_date
    result["historical_accuracy"] = {
        "total_days": len(merged),
        "same_direction_days": len(same_dir),
        "hit_count": hit_count,
        "accuracy": safe_json(round(accuracy, 4)),
    }
    result[
        "disclaimer"] = "This is a research tool, not financial advice. Predictions are based on historical sentiment-price correlation patterns and should not be used for investment decisions."

    return jsonify(result)


def run_refresh():
    global refresh_status
    refresh_status = {"running": True, "last_run": None, "message": "Starting..."}

    modules = [
        ("Reddit Collection (7 days)", [sys.executable, str(MODULES_DIR / "reddit_collector.py"), "--days", "7"]),
        ("Price Collection", [sys.executable, str(MODULES_DIR / "price_collector.py"), "--days", "30"]),
        ("NLP Preprocessing", [sys.executable, str(MODULES_DIR / "nlp_preprocessor.py")]),
        ("Sentiment Scoring", [sys.executable, str(MODULES_DIR / "sentiment_scorer.py")]),
        ("Advanced NLP", [sys.executable, str(MODULES_DIR / "advanced_nlp.py")]),
        ("Correlation Engine", [sys.executable, str(MODULES_DIR / "correlation_engine.py")]),
    ]

    for name, cmd in modules:
        refresh_status["message"] = f"Running: {name}"
        logger.info(f"Refresh: {name}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                logger.warning(f"  {name} returned code {result.returncode}")
                logger.warning(f"  stderr: {result.stderr[:300]}")
        except subprocess.TimeoutExpired:
            logger.error(f"  {name} timed out (10 min)")
        except FileNotFoundError:
            logger.warning(f"  {name}: module not found, skipping")

    refresh_status = {
        "running": False,
        "last_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": "Complete",
    }
    logger.info("Refresh complete!")


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    if refresh_status["running"]:
        return jsonify({"status": "already_running", "message": refresh_status["message"]})

    thread = threading.Thread(target=run_refresh, daemon=True)
    thread.start()
    return jsonify({"status": "started", "message": "Pipeline refresh started"})


@app.route("/api/refresh/status")
def api_refresh_status():
    return jsonify(refresh_status)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto Sentiment Dashboard Server")
    parser.add_argument("--port", type=int, default=5002, help="Port (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Crypto Sentiment Dashboard")
    print(f"  http://localhost:{args.port}")
    print(f"  Database: {DB_PATH}")
    print(f"  Ollama:   {OLLAMA_URL} ({OLLAMA_MODEL})\n")

    app.run(host="0.0.0.0", port=args.port, debug=args.debug)