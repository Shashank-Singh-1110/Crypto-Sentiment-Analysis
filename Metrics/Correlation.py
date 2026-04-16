import argparse
import csv
import json
import logging
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DB_PATH = Path("data") / "project.db"
EXPORT_DIR = Path("data") / "correlation"
LOG_DIR = Path("logs")

COINS = ["BTC", "ETH", "SOL", "DOGE", "SHIB", "BNB", "MARKET"]
LAGS = [0, 1, 2, 3]
GRANGER_MAX_LAG = 3
MIN_OBSERVATIONS = 10

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "correlation_engine.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def load_sentiment_daily(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query("""
                           SELECT
                               date, coin_target, COUNT (*) as post_count, AVG (vader_compound) as vader_mean, AVG (finbert_score) as finbert_mean, AVG (composite_score) as composite_mean, AVG (composite_zscore) as zscore_mean, AVG (weighted_score) as weighted_mean, AVG (engagement_weight) as engagement_mean, SUM (upvotes) as upvotes_total, SUM (num_comments) as comments_total
                           FROM posts
                           WHERE sentiment_processed > 0
                           GROUP BY date, coin_target
                           ORDER BY date, coin_target
                           """, conn)

    df["date"] = pd.to_datetime(df["date"])
    return df


def load_prices(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load daily price data."""
    df = pd.read_sql_query("""
                           SELECT coin, date, open, high, low, close, volume
                           FROM prices
                           ORDER BY coin, date
                           """, conn)

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    df["daily_return"] = df.groupby("coin")["close"].pct_change()
    df["price_direction"] = (df["daily_return"] > 0).astype(int)  # 1=up, 0=down
    df["log_return"] = np.log(df["close"] / df.groupby("coin")["close"].shift(1))
    df["volatility"] = df.groupby("coin")["daily_return"].transform(
        lambda x: x.rolling(3, min_periods=1).std()
    )

    return df


def load_macro(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load macro indicator data."""
    # Check if tables exist
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='macro_indicators'")
    if not c.fetchone():
        return pd.DataFrame()

    df = pd.read_sql_query("""
                           SELECT indicator, date, close as value
                           FROM macro_indicators
                           ORDER BY indicator, date
                           """, conn)

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot_table(index="date", columns="indicator", values="value", aggfunc="first")
    pivot = pivot.reset_index()

    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fear_greed'")
    if c.fetchone():
        fg = pd.read_sql_query("SELECT date, value as FEAR_GREED FROM fear_greed", conn)
        if not fg.empty:
            fg["date"] = pd.to_datetime(fg["date"])
            pivot = pivot.merge(fg, on="date", how="outer")

    return pivot


def load_emotions_daily(conn: sqlite3.Connection) -> pd.DataFrame:
    c = conn.cursor()
    c.execute("PRAGMA table_info(posts)")
    cols = {row[1] for row in c.fetchall()}
    if "emotion_label" not in cols:
        return pd.DataFrame()

    df = pd.read_sql_query("""
                           SELECT date, coin_target, emotion_label, COUNT (*) as count
                           FROM posts
                           WHERE emotions_processed = 1
                           GROUP BY date, coin_target, emotion_label
                           """, conn)

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot_table(
        index=["date", "coin_target"],
        columns="emotion_label",
        values="count",
        fill_value=0,
    ).reset_index()
    emotion_cols = [c for c in pivot.columns if c not in ("date", "coin_target")]
    total = pivot[emotion_cols].sum(axis=1)
    for col in emotion_cols:
        pivot[f"{col}_ratio"] = pivot[col] / total.replace(0, 1)
    return pivot


def merge_sentiment_price(sentiment_df: pd.DataFrame,
                          price_df: pd.DataFrame,
                          coin: str) -> pd.DataFrame:
    sent = sentiment_df[sentiment_df["coin_target"] == coin].copy()
    price_coin = coin if coin != "MARKET" else None

    if price_coin:
        price = price_df[price_df["coin"] == price_coin].copy()
    else:
        price = price_df[price_df["coin"] == "BTC"].copy()

    if sent.empty or price.empty:
        return pd.DataFrame()

    merged = sent.merge(price, on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)

    return merged


def lagged_pearson(merged: pd.DataFrame, sentiment_col: str,
                   price_col: str = "daily_return") -> list[dict]:
    """
    Compute Pearson correlation at multiple lags.
    Positive lag = sentiment leads price (what we want to test).
    """
    results = []

    for lag in LAGS:
        if lag == 0:
            s = merged[sentiment_col].values
            p = merged[price_col].values
        else:
            # Sentiment at time t correlated with price at time t+lag
            s = merged[sentiment_col].iloc[:-lag].values
            p = merged[price_col].iloc[lag:].values

        # Drop NaN pairs
        mask = ~(np.isnan(s) | np.isnan(p))
        s_clean = s[mask]
        p_clean = p[mask]

        if len(s_clean) < MIN_OBSERVATIONS:
            continue

        r, p_value = stats.pearsonr(s_clean, p_clean)

        results.append({
            "lag": lag,
            "correlation": round(r, 6),
            "p_value": round(p_value, 6),
            "significant": p_value < 0.05,
            "n_observations": len(s_clean),
        })

    return results


def granger_causality(merged: pd.DataFrame, sentiment_col: str,
                      price_col: str = "daily_return") -> dict:
    """
    Granger causality test: does sentiment Granger-cause price returns?
    Returns test statistics and p-values for each lag.
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        logger.warning("statsmodels not installed. Skipping Granger causality.")
        return {}

    # Prepare data: need both columns without NaN
    data = merged[[price_col, sentiment_col]].dropna()

    if len(data) < MIN_OBSERVATIONS + GRANGER_MAX_LAG:
        return {"error": "insufficient_data", "n": len(data)}

    try:
        result = grangercausalitytests(data, maxlag=GRANGER_MAX_LAG, verbose=False)

        granger_results = {}
        for lag in range(1, GRANGER_MAX_LAG + 1):
            test_result = result[lag]
            # ssr_ftest is the standard F-test
            f_stat = test_result[0]["ssr_ftest"][0]
            p_value = test_result[0]["ssr_ftest"][1]
            granger_results[f"lag_{lag}"] = {
                "f_statistic": round(f_stat, 4),
                "p_value": round(p_value, 6),
                "significant": p_value < 0.05,
            }

        return granger_results

    except Exception as e:
        return {"error": str(e)}


def binary_classification(merged: pd.DataFrame,
                          sentiment_col: str) -> dict:

    sent = merged[sentiment_col].iloc[:-1].values
    direction = merged["price_direction"].iloc[1:].values
    mask = ~(np.isnan(sent) | np.isnan(direction))
    sent = sent[mask]
    direction = direction[mask]

    if len(sent) < MIN_OBSERVATIONS:
        return {"error": "insufficient_data", "n": len(sent)}

    predicted = (sent > 0).astype(int)
    actual = direction.astype(int)
    correct = (predicted == actual).sum()
    total = len(actual)
    accuracy = correct / total
    true_pos = ((predicted == 1) & (actual == 1)).sum()
    false_pos = ((predicted == 1) & (actual == 0)).sum()
    false_neg = ((predicted == 0) & (actual == 1)).sum()

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    baseline = max(actual.mean(), 1 - actual.mean())

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "baseline_accuracy": round(baseline, 4),
        "beats_baseline": accuracy > baseline,
        "n_observations": total,
        "up_ratio": round(actual.mean(), 4),
    }


def run_ablation(merged: pd.DataFrame, coin: str) -> list[dict]:
    variants = [
        ("VADER_only", "vader_mean"),
        ("FinBERT_only", "finbert_mean"),
        ("Composite", "composite_mean"),
        ("Composite_zscore", "zscore_mean"),
        ("Full_weighted", "weighted_mean"),
    ]

    results = []

    for variant_name, col in variants:
        if col not in merged.columns or merged[col].isna().all():
            continue

        pearson_results = lagged_pearson(merged, col, "daily_return")
        lag1 = next((r for r in pearson_results if r["lag"] == 1), None)
        binary = binary_classification(merged, col)

        result = {
            "coin": coin,
            "variant": variant_name,
            "sentiment_column": col,
        }

        if lag1:
            result["lag1_correlation"] = lag1["correlation"]
            result["lag1_p_value"] = lag1["p_value"]
            result["lag1_significant"] = lag1["significant"]

        if "accuracy" in binary:
            result["accuracy"] = binary["accuracy"]
            result["f1_score"] = binary["f1_score"]
            result["baseline_accuracy"] = binary["baseline_accuracy"]
            result["beats_baseline"] = binary["beats_baseline"]

        results.append(result)
    return results


def run_macro_correlation(price_df: pd.DataFrame,
                          macro_df: pd.DataFrame) -> list[dict]:
    if macro_df.empty:
        logger.info("No macro data available. Skipping macro correlation.")
        return []

    results = []
    macro_cols = [c for c in macro_df.columns if c != "date"]

    for coin in ["BTC", "ETH", "SOL"]:  # Main coins only
        price = price_df[price_df["coin"] == coin][["date", "daily_return"]].dropna()

        for macro_col in macro_cols:
            macro = macro_df[["date", macro_col]].dropna()
            macro = macro.sort_values("date")
            macro[f"{macro_col}_return"] = macro[macro_col].pct_change()

            merged = price.merge(macro, on="date", how="inner")

            if len(merged) < MIN_OBSERVATIONS:
                continue

            r, p_value = stats.pearsonr(
                merged["daily_return"].values,
                merged[f"{macro_col}_return"].dropna().values[:len(merged)]
            )

            results.append({
                "coin": coin,
                "macro_indicator": macro_col,
                "correlation": round(r, 6),
                "p_value": round(p_value, 6),
                "significant": p_value < 0.05,
                "n_observations": len(merged),
            })

    return results


def run_full_analysis(target_coin: str = None):
    conn = sqlite3.connect(str(DB_PATH))
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"{'=' * 60}")
    logger.info("CORRELATION ENGINE — FULL ANALYSIS")
    logger.info(f"{'=' * 60}")
    logger.info("Loading data...")
    sentiment_df = load_sentiment_daily(conn)
    price_df = load_prices(conn)
    macro_df = load_macro(conn)
    emotions_df = load_emotions_daily(conn)

    logger.info(f"  Sentiment: {len(sentiment_df)} daily records")
    logger.info(f"  Prices:    {len(price_df)} daily records")
    logger.info(f"  Macro:     {len(macro_df)} daily records")
    logger.info(f"  Emotions:  {len(emotions_df)} daily records")

    if sentiment_df.empty or price_df.empty:
        logger.error("Insufficient data for analysis!")
        conn.close()
        return

    coins = [target_coin] if target_coin else COINS
    all_pearson = []
    all_granger = []
    all_binary = []
    all_ablation = []

    for coin in coins:
        price_coin = coin if coin != "MARKET" else "BTC"
        if price_coin not in price_df["coin"].values:
            logger.warning(f"  No price data for {price_coin}. Skipping {coin}.")
            continue

        merged = merge_sentiment_price(sentiment_df, price_df, coin)

        if merged.empty or len(merged) < MIN_OBSERVATIONS:
            logger.warning(f"  {coin}: insufficient merged data ({len(merged)} days). Skipping.")
            continue

        logger.info(f"\n{'─' * 50}")
        logger.info(f"ANALYSIS: {coin} ({len(merged)} days)")
        logger.info(f"{'─' * 50}")
        sentiment_cols = ["vader_mean", "finbert_mean", "composite_mean",
                          "zscore_mean", "weighted_mean"]

        for scol in sentiment_cols:
            if scol not in merged.columns:
                continue
            results = lagged_pearson(merged, scol, "daily_return")
            for r in results:
                r["coin"] = coin
                r["sentiment_metric"] = scol
                all_pearson.append(r)
            if results:
                best = max(results, key=lambda x: abs(x["correlation"]))
                sig = "*" if best["significant"] else ""
                logger.info(f"  Pearson ({scol}): best r={best['correlation']:.4f}{sig} at lag {best['lag']}")

        granger = granger_causality(merged, "composite_mean", "daily_return")
        if "error" not in granger:
            for lag_key, lag_result in granger.items():
                all_granger.append({
                    "coin": coin,
                    "lag": lag_key,
                    "f_statistic": lag_result["f_statistic"],
                    "p_value": lag_result["p_value"],
                    "significant": lag_result["significant"],
                })
                if lag_result["significant"]:
                    logger.info(f"  Granger ({lag_key}): F={lag_result['f_statistic']:.2f}, "
                                f"p={lag_result['p_value']:.4f} ***SIGNIFICANT***")
        else:
            logger.info(f"  Granger: {granger.get('error', 'failed')}")
        for scol in sentiment_cols:
            if scol not in merged.columns:
                continue
            binary = binary_classification(merged, scol)
            if "accuracy" in binary:
                binary["coin"] = coin
                binary["sentiment_metric"] = scol
                all_binary.append(binary)

                beats = "BEATS BASELINE" if binary["beats_baseline"] else "below baseline"
                logger.info(f"  Binary ({scol}): acc={binary['accuracy']:.3f} "
                            f"(baseline={binary['baseline_accuracy']:.3f}) — {beats}")

        ablation = run_ablation(merged, coin)
        all_ablation.extend(ablation)
    logger.info(f"\n{'─' * 50}")
    logger.info("MACRO INDICATOR CORRELATION")
    logger.info(f"{'─' * 50}")

    macro_results = run_macro_correlation(price_df, macro_df)
    if macro_results:
        for mr in macro_results:
            if mr["significant"]:
                logger.info(f"  {mr['coin']} vs {mr['macro_indicator']}: "
                            f"r={mr['correlation']:.4f} ***SIGNIFICANT***")


    logger.info(f"\n{'=' * 60}")
    logger.info("EXPORTING RESULTS")
    logger.info(f"{'=' * 60}")
    if all_pearson:
        df = pd.DataFrame(all_pearson)
        path = EXPORT_DIR / "pearson_correlations.csv"
        df.to_csv(path, index=False)
        logger.info(f"  Pearson: {path} ({len(df)} rows)")
    if all_granger:
        df = pd.DataFrame(all_granger)
        path = EXPORT_DIR / "granger_causality.csv"
        df.to_csv(path, index=False)
        logger.info(f"  Granger: {path} ({len(df)} rows)")

    if all_binary:
        df = pd.DataFrame(all_binary)
        path = EXPORT_DIR / "binary_classification.csv"
        df.to_csv(path, index=False)
        logger.info(f"  Binary: {path} ({len(df)} rows)")

    if all_ablation:
        df = pd.DataFrame(all_ablation)
        path = EXPORT_DIR / "ablation_study.csv"
        df.to_csv(path, index=False)
        logger.info(f"  Ablation: {path} ({len(df)} rows)")
        logger.info(f"\n{'─' * 70}")
        logger.info("ABLATION STUDY SUMMARY")
        logger.info(f"{'─' * 70}")
        logger.info(f"{'Coin':<8} {'Variant':<22} {'Lag1 r':>8} {'p-value':>9} {'Acc':>7} {'F1':>7} {'Beats?':>7}")
        logger.info(f"{'─' * 70}")

        for row in all_ablation:
            r = row.get("lag1_correlation", float("nan"))
            p = row.get("lag1_p_value", float("nan"))
            acc = row.get("accuracy", float("nan"))
            f1 = row.get("f1_score", float("nan"))
            beats = "YES" if row.get("beats_baseline", False) else "no"
            logger.info(f"{row['coin']:<8} {row['variant']:<22} {r:>8.4f} {p:>9.4f} "
                        f"{acc:>7.3f} {f1:>7.3f} {beats:>7}")

    if macro_results:
        df = pd.DataFrame(macro_results)
        path = EXPORT_DIR / "macro_correlations.csv"
        df.to_csv(path, index=False)
        logger.info(f"  Macro: {path} ({len(df)} rows)")

    if all_pearson:
        pearson_df = pd.DataFrame(all_pearson)
        lag1 = pearson_df[pearson_df["lag"] == 1]
        if not lag1.empty:
            heatmap = lag1.pivot_table(
                index="coin", columns="sentiment_metric",
                values="correlation", aggfunc="first",
            )
            heatmap_path = EXPORT_DIR / "correlation_heatmap_data.csv"
            heatmap.to_csv(heatmap_path)
            logger.info(f"  Heatmap: {heatmap_path}")

    conn.close()
    logger.info(f"\n{'=' * 60}")
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"{'=' * 60}")


def print_status():
    if not EXPORT_DIR.exists():
        print("No correlation results found. Run analysis first.")
        return

    print(f"\n{'=' * 60}")
    print("CORRELATION ENGINE — RESULTS SUMMARY")
    print(f"{'=' * 60}")

    pearson_path = EXPORT_DIR / "pearson_correlations.csv"
    if pearson_path.exists():
        df = pd.read_csv(pearson_path)
        sig = df[df["significant"] == True]
        print(f"\nPearson correlations: {len(df)} tests, {len(sig)} significant")

        if not sig.empty:
            print("\n  Significant results:")
            for _, row in sig.iterrows():
                print(f"    {row['coin']:<8} {row['sentiment_metric']:<18} lag={row['lag']} "
                      f"r={row['correlation']:>7.4f} p={row['p_value']:.4f}")

    ablation_path = EXPORT_DIR / "ablation_study.csv"
    if ablation_path.exists():
        df = pd.read_csv(ablation_path)
        print(f"\nAblation study: {len(df)} variant-coin combinations")

        print(f"\n  {'Coin':<8} {'Best Variant':<22} {'Accuracy':>9} {'Beats Baseline?'}")
        print(f"  {'─' * 55}")

        for coin in df["coin"].unique():
            coin_df = df[df["coin"] == coin]
            if "accuracy" in coin_df.columns:
                best = coin_df.loc[coin_df["accuracy"].idxmax()]
                beats = "YES" if best.get("beats_baseline", False) else "no"
                print(f"  {coin:<8} {best['variant']:<22} {best['accuracy']:>9.4f} {beats}")

    binary_path = EXPORT_DIR / "binary_classification.csv"
    if binary_path.exists():
        df = pd.read_csv(binary_path)
        beats = df[df["beats_baseline"] == True]
        print(f"\nBinary classification: {len(df)} tests, {len(beats)} beat baseline")

    granger_path = EXPORT_DIR / "granger_causality.csv"
    if granger_path.exists():
        df = pd.read_csv(granger_path)
        sig = df[df["significant"] == True]
        print(f"\nGranger causality: {len(df)} tests, {len(sig)} significant")
        if not sig.empty:
            for _, row in sig.iterrows():
                print(f"  {row['coin']}: {row['lag']} — F={row['f_statistic']:.2f}, p={row['p_value']:.4f}")

    macro_path = EXPORT_DIR / "macro_correlations.csv"
    if macro_path.exists():
        df = pd.read_csv(macro_path)
        sig = df[df["significant"] == True]
        print(f"\nMacro correlations: {len(df)} tests, {len(sig)} significant")


def main():
    parser = argparse.ArgumentParser(
        description="Correlation Engine + Ablation Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python correlation_engine.py                  # Full analysis
  python correlation_engine.py --coin BTC       # BTC only
  python correlation_engine.py --status         # Show results
        """,
    )
    parser.add_argument("--coin", type=str, default=None,
                        help="Analyse single coin (e.g., BTC, ETH)")
    parser.add_argument("--status", action="store_true",
                        help="Show results summary")

    args = parser.parse_args()

    if args.status:
        print_status()
        return

    run_full_analysis(target_coin=args.coin)


if __name__ == "__main__":
    main()
