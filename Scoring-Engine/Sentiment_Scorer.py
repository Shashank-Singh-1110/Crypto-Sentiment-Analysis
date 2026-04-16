import argparse
import csv
import logging
import math
import sqlite3
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

DB_PATH = Path("data") / "project.db"
EXPORT_DIR = Path("data") / "sentiment"
LOG_DIR = Path("logs")

DEFAULT_BATCH_SIZE = 32
FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_MAX_LENGTH = 512
COMPOSITE_WEIGHT_FINBERT = 0.7
COMPOSITE_WEIGHT_VADER = 0.3

CRYPTO_LEXICON = {
    "bullish":      2.5,
    "bearish":     -2.5,
    "moon":         2.0,
    "mooning":      2.5,
    "dump":        -2.0,
    "dumping":     -2.5,
    "pump":         1.5,
    "pumping":      2.0,
    "hodl":         1.5,
    "hold":         0.5,
    "rekt":        -3.0,
    "scam":        -3.0,
    "rug":         -3.5,
    "rugged":      -3.5,
    "fud":         -2.0,
    "fomo":        -1.0,
    "wagmi":        2.5,
    "ngmi":        -2.5,
    "dip":         -1.0,
    "crash":       -3.0,
    "rally":        2.5,
    "surge":        2.0,
    "plunge":      -3.0,
    "tank":        -2.5,
    "tanking":     -2.5,
    "soar":         2.5,
    "soaring":      2.5,
    "breakout":     2.0,
    "breakdown":   -2.0,
    "whale":        0.5,
    "accumulate":   1.5,
    "sell":        -1.0,
    "buy":          1.0,
    "long":         1.0,
    "short":       -1.0,
    "liquidated":  -3.0,
    "liquidation": -2.5,
    "ath":          2.5,
    "atl":         -2.5,
    "dyor":         0.0,
    "nfa":          0.0,
    "undervalued":  1.5,
    "overvalued":  -1.5,
    "adoption":     2.0,
    "regulation":  -1.0,
    "ban":         -3.0,
    "halving":      1.5,
    "staking":      1.0,
    "airdrop":      1.5,
    "exploit":     -3.0,
    "hack":        -3.5,
    "hacked":      -3.5,
}

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "sentiment_scorer.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("PRAGMA table_info(posts)")
    existing_cols = {row[1] for row in c.fetchall()}

    sentiment_columns = {
        "vader_neg": "REAL DEFAULT 0.0",
        "vader_neu": "REAL DEFAULT 0.0",
        "vader_pos": "REAL DEFAULT 0.0",
        "vader_compound": "REAL DEFAULT 0.0",
        "finbert_positive": "REAL DEFAULT 0.0",
        "finbert_negative": "REAL DEFAULT 0.0",
        "finbert_neutral": "REAL DEFAULT 0.0",
        "finbert_score": "REAL DEFAULT 0.0",  # positive - negative
        "finbert_label": "TEXT",
        "composite_score": "REAL DEFAULT 0.0",  # 0.7*finbert + 0.3*vader
        "composite_zscore": "REAL DEFAULT 0.0",  # z-score corrected
        "engagement_weight": "REAL DEFAULT 1.0",  # 1 + log1p(up) + log1p(com)
        "weighted_score": "REAL DEFAULT 0.0",  # composite × engagement
        "sentiment_processed": "INTEGER DEFAULT 0",  # 0=no, 1=vader+finbert, 2=vader only
    }

    for col, dtype in sentiment_columns.items():
        if col not in existing_cols:
            c.execute(f"ALTER TABLE posts ADD COLUMN {col} {dtype}")
            logger.info(f"  Added column: {col}")

    conn.commit()
    return conn

def init_vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    for word, score in CRYPTO_LEXICON.items():
        analyzer.lexicon[word] = score

        logger.info(f"VADER initialised with {len(CRYPTO_LEXICON)} crypto lexicon terms")
        return analyzer

def score_vader(analyzer, text: str)->dict:
    scores = analyzer.polarity_scores(text)
    return {
        "vader_neg": scores["neg"],
        "vader_neu": scores["neu"],
        "vader_pos": scores["pos"],
        "vader_compound": scores["compound"],
    }

def init_finbert(device: str = None):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    if device is None:
        device ='cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Loading FinBERT ({FINBERT_MODEL}) on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    model.to(device)
    model.eval()

    logger.info("FinBERT loaded successfully")
    return tokenizer, model, device

def score_finbert(tokenizer, model, device, texts: list[str])-> list[dict]:
    import torch
    inputs = tokenizer(texts, padding=True, truncation=True, max_length = FINBERT_MAX_LENGTH, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    results = []

    for i in range(len(texts)):
        pos = probs[i][0].item()
        neg = probs[i][1].item()
        neu = probs[i][2].item()

        score = pos - neg
        if pos > neg and pos > neu:
            label = "positive"
        elif neg > pos and neg > neu:
            label = "negative"
        else:
            label = "neutral"

        results.append({
            "finbert_positive": round(pos, 6),
            "finbert_negative": round(neg, 6),
            "finbert_neutral": round(neu, 6),
            "finbert_score": round(score, 6),
            "finbert_label": label,
        })

    return results


def compute_composite(vader_compound: float, finbert_score: float) -> float:
    return (COMPOSITE_WEIGHT_FINBERT * finbert_score +
            COMPOSITE_WEIGHT_VADER * vader_compound)


def compute_engagement_weight(upvotes: int, num_comments: int) -> float:
    """
    Engagement weight: higher-engagement posts get more influence.
    Formula: 1 + log1p(upvotes) + log1p(comments)
    """
    return 1.0 + math.log1p(max(upvotes, 0)) + math.log1p(max(num_comments, 0))


def apply_zscore_correction(conn: sqlite3.Connection):
    c = conn.cursor()

    logger.info("Applying z-score bias correction per coin...")
    c.execute("""
              SELECT coin_target,
                     AVG(composite_score),
                     CASE
                         WHEN COUNT(*) > 1
                             THEN SQRT(SUM((composite_score - sub.mean_score) *
                                           (composite_score - sub.mean_score)) / (COUNT(*) - 1))
                         ELSE 1.0 END
              FROM posts,
                   (SELECT coin_target AS ct, AVG(composite_score) AS mean_score
                    FROM posts
                    WHERE sentiment_processed > 0
                    GROUP BY coin_target) sub
              WHERE posts.coin_target = sub.ct
                AND sentiment_processed > 0
              GROUP BY posts.coin_target
              """)

    df = pd.read_sql_query("""
                           SELECT id, coin_target, composite_score
                           FROM posts
                           WHERE sentiment_processed > 0
                           """, conn)

    if df.empty:
        return
    for coin in df["coin_target"].unique():
        mask = df["coin_target"] == coin
        scores = df.loc[mask, "composite_score"]

        mean = scores.mean()
        std = scores.std()
        if std == 0 or pd.isna(std):
            std = 1.0

        zscores = (scores - mean) / std
        for idx in df.loc[mask].index:
            post_id = df.loc[idx, "id"]
            zscore = round(float(zscores.loc[idx]), 6)
            c.execute("UPDATE posts SET composite_zscore = ? WHERE id = ?",
                      (zscore, post_id))

        logger.info(f"  {coin}: mean={mean:.4f}, std={std:.4f}, "
                    f"n={mask.sum()}")

    conn.commit()
    logger.info("Z-score correction applied")


def run_scoring(batch_size: int = DEFAULT_BATCH_SIZE,
                rescore: bool = False,
                vader_only: bool = False):
    conn = init_db()
    c = conn.cursor()
    if rescore:
        c.execute("UPDATE posts SET sentiment_processed = 0")
        conn.commit()

    # Count work
    c.execute("""
              SELECT COUNT(*)
              FROM posts
              WHERE nlp_processed = 1
                AND sentiment_processed = 0
              """)
    total = c.fetchone()[0]

    if total == 0:
        logger.info("No posts to score! (Only NLP-processed posts are scored)")
        conn.close()
        return

    logger.info(f"{'=' * 60}")
    logger.info(f"SENTIMENT SCORING — {total:,} posts")
    logger.info(f"Mode: {'VADER only' if vader_only else 'VADER + FinBERT'}")
    logger.info(f"Composite weights: FinBERT={COMPOSITE_WEIGHT_FINBERT}, VADER={COMPOSITE_WEIGHT_VADER}")
    logger.info(f"{'=' * 60}")

    # Init models
    vader = init_vader()

    finbert_tokenizer = None
    finbert_model = None
    finbert_device = None

    if not vader_only:
        try:
            finbert_tokenizer, finbert_model, finbert_device = init_finbert()
        except Exception as e:
            logger.warning(f"FinBERT failed to load: {e}")
            logger.warning("Falling back to VADER-only mode")
            vader_only = True

    total_scored = 0
    batch_num = 0
    start_time = time.time()

    while True:
        # Fetch batch
        c.execute("""
                  SELECT id, text_clean, upvotes, num_comments
                  FROM posts
                  WHERE nlp_processed = 1
                    AND sentiment_processed = 0 LIMIT ?
                  """, (batch_size,))

        batch = c.fetchall()
        if not batch:
            break

        batch_num += 1
        post_ids = [row[0] for row in batch]
        texts = [row[1] or "" for row in batch]
        upvotes_list = [row[2] or 0 for row in batch]
        comments_list = [row[3] or 0 for row in batch]
        vader_results = [score_vader(vader, t) for t in texts]

        if not vader_only:
            finbert_texts = [t[:2000] for t in texts]  # rough char limit
            finbert_results = score_finbert(
                finbert_tokenizer, finbert_model, finbert_device, finbert_texts
            )
        else:
            finbert_results = [None] * len(texts)

        for i, post_id in enumerate(post_ids):
            vr = vader_results[i]
            upvotes = upvotes_list[i]
            comments = comments_list[i]
            eng_weight = compute_engagement_weight(upvotes, comments)

            if not vader_only and finbert_results[i] is not None:
                fr = finbert_results[i]
                composite = compute_composite(vr["vader_compound"], fr["finbert_score"])
                weighted = composite * eng_weight

                c.execute("""
                          UPDATE posts
                          SET vader_neg           = ?,
                              vader_neu           = ?,
                              vader_pos           = ?,
                              vader_compound      = ?,
                              finbert_positive    = ?,
                              finbert_negative    = ?,
                              finbert_neutral     = ?,
                              finbert_score       = ?,
                              finbert_label       = ?,
                              composite_score     = ?,
                              engagement_weight   = ?,
                              weighted_score      = ?,
                              sentiment_processed = 1
                          WHERE id = ?
                          """, (
                              vr["vader_neg"], vr["vader_neu"], vr["vader_pos"], vr["vader_compound"],
                              fr["finbert_positive"], fr["finbert_negative"], fr["finbert_neutral"],
                              fr["finbert_score"], fr["finbert_label"],
                              round(composite, 6),
                              round(eng_weight, 6),
                              round(weighted, 6),
                              post_id,
                          ))
            else:
                composite = vr["vader_compound"]
                weighted = composite * eng_weight

                c.execute("""
                          UPDATE posts
                          SET vader_neg           = ?,
                              vader_neu           = ?,
                              vader_pos           = ?,
                              vader_compound      = ?,
                              composite_score     = ?,
                              engagement_weight   = ?,
                              weighted_score      = ?,
                              sentiment_processed = 2
                          WHERE id = ?
                          """, (
                              vr["vader_neg"], vr["vader_neu"], vr["vader_pos"], vr["vader_compound"],
                              round(composite, 6),
                              round(eng_weight, 6),
                              round(weighted, 6),
                              post_id,
                          ))

        conn.commit()
        total_scored += len(batch)

        elapsed = time.time() - start_time
        rate = total_scored / elapsed if elapsed > 0 else 0
        remaining = (total - total_scored) / rate if rate > 0 else 0

        logger.info(
            f"  Batch {batch_num}: {len(batch)} scored | "
            f"Total: {total_scored}/{total} | "
            f"Rate: {rate:.1f} posts/sec | ETA: {remaining:.0f}s"
        )

    apply_zscore_correction(conn)
    elapsed = time.time() - start_time
    logger.info(f"\n{'=' * 60}")
    logger.info(f"SCORING COMPLETE")
    logger.info(f"Scored: {total_scored:,} posts")
    if elapsed > 0:
        logger.info(f"Time:   {elapsed:.1f}s ({total_scored / elapsed:.1f} posts/sec)")
    logger.info(f"{'=' * 60}")

    conn.close()


def export_csvs():
    conn = sqlite3.connect(str(DB_PATH))
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_sql_query("""
                           SELECT id,
                                  subreddit,
                                  coin_target,
                                  text_clean, date, upvotes, upvote_ratio, num_comments, word_count, vader_neg, vader_neu, vader_pos, vader_compound, finbert_positive, finbert_negative, finbert_neutral, finbert_score, finbert_label, composite_score, composite_zscore, engagement_weight, weighted_score, sentiment_processed
                           FROM posts
                           WHERE sentiment_processed > 0
                           ORDER BY date, coin_target
                           """, conn)

    if df.empty:
        logger.warning("No scored posts to export!")
        conn.close()
        return

    logger.info(f"Exporting {len(df):,} sentiment-scored posts...")
    full_path = EXPORT_DIR / "nlp_sentiment_scored.csv"
    df.to_csv(full_path, index=False, quoting=csv.QUOTE_MINIMAL)
    logger.info(f"  Full: {full_path}")
    daily_agg = df.groupby(["date", "coin_target"]).agg(
        post_count=("id", "count"),
        vader_mean=("vader_compound", "mean"),
        finbert_mean=("finbert_score", "mean"),
        composite_mean=("composite_score", "mean"),
        zscore_mean=("composite_zscore", "mean"),
        weighted_mean=("weighted_score", "mean"),
        engagement_mean=("engagement_weight", "mean"),
        upvotes_total=("upvotes", "sum"),
        comments_total=("num_comments", "sum"),
    ).reset_index()

    daily_path = EXPORT_DIR / "sentiment_daily_by_coin.csv"
    daily_agg.to_csv(daily_path, index=False)
    logger.info(f"  Daily by coin: {daily_path} ({len(daily_agg)} rows)")

    daily_all = df.groupby("date").agg(
        post_count=("id", "count"),
        vader_mean=("vader_compound", "mean"),
        finbert_mean=("finbert_score", "mean"),
        composite_mean=("composite_score", "mean"),
        zscore_mean=("composite_zscore", "mean"),
        weighted_mean=("weighted_score", "mean"),
    ).reset_index()

    daily_all_path = EXPORT_DIR / "sentiment_daily_overall.csv"
    daily_all.to_csv(daily_all_path, index=False)
    logger.info(f"  Daily overall: {daily_all_path} ({len(daily_all)} rows)")

    conn.close()
    logger.info("CSV export complete!")


def print_status():
    """Print sentiment scoring statistics."""
    if not DB_PATH.exists():
        print("No database found.")
        return

    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("PRAGMA table_info(posts)")
    cols = {row[1] for row in c.fetchall()}
    if "sentiment_processed" not in cols:
        print("Sentiment scoring has not been run yet.")
        conn.close()
        return

    print(f"\n{'=' * 60}")
    print("SENTIMENT SCORING STATUS")
    print(f"{'=' * 60}")

    c.execute("SELECT COUNT(*) FROM posts")
    total = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM posts WHERE sentiment_processed = 1")
    full_scored = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM posts WHERE sentiment_processed = 2")
    vader_only = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM posts WHERE sentiment_processed = 0 AND nlp_processed = 1")
    pending = c.fetchone()[0]

    print(f"\nTotal posts:           {total:,}")
    print(f"VADER + FinBERT:       {full_scored:,}")
    print(f"VADER only:            {vader_only:,}")
    print(f"Pending (NLP-ready):   {pending:,}")

    scored = full_scored + vader_only
    if scored > 0:
        c.execute("""
                  SELECT ROUND(AVG(vader_compound), 4),
                         ROUND(AVG(finbert_score), 4),
                         ROUND(AVG(composite_score), 4),
                         ROUND(AVG(engagement_weight), 2),
                         ROUND(AVG(weighted_score), 4)
                  FROM posts
                  WHERE sentiment_processed > 0
                  """)
        avg_vader, avg_finbert, avg_comp, avg_eng, avg_weighted = c.fetchone()

        print(f"\nOverall averages:")
        print(f"  VADER compound:    {avg_vader}")
        print(f"  FinBERT score:     {avg_finbert}")
        print(f"  Composite:         {avg_comp}")
        print(f"  Engagement weight: {avg_eng}")
        print(f"  Weighted score:    {avg_weighted}")
        print(f"\n{'─' * 70}")
        print(f"{'Coin':<8} {'Posts':>7} {'VADER':>8} {'FinBERT':>9} {'Composite':>10} {'Weighted':>10}")
        print(f"{'─' * 70}")

        for row in c.execute("""
                             SELECT coin_target,
                                    COUNT(*),
                                    ROUND(AVG(vader_compound), 4),
                                    ROUND(AVG(finbert_score), 4),
                                    ROUND(AVG(composite_score), 4),
                                    ROUND(AVG(weighted_score), 4)
                             FROM posts
                             WHERE sentiment_processed > 0
                             GROUP BY coin_target
                             ORDER BY coin_target
                             """):
            coin, cnt, v, f, comp, w = row
            print(f"{coin:<8} {cnt:>7,} {v or 0:>8.4f} {f or 0:>9.4f} {comp or 0:>10.4f} {w or 0:>10.4f}")
        if full_scored > 0:
            print(f"\nFinBERT label distribution:")
            for row in c.execute("""
                                 SELECT finbert_label,
                                        COUNT(*),
                                        ROUND(COUNT(*) * 100.0 / ?, 1)
                                 FROM posts
                                 WHERE sentiment_processed = 1
                                   AND finbert_label IS NOT NULL
                                 GROUP BY finbert_label
                                 ORDER BY COUNT(*) DESC
                                 """, (full_scored,)):
                label, cnt, pct = row
                print(f"  {label:<12} {cnt:>8,} ({pct}%)")

        print(f"\nSentiment distribution (composite):")
        for row in c.execute("""
                             SELECT SUM(CASE WHEN composite_score > 0.05 THEN 1 ELSE 0 END),
                                    SUM(CASE WHEN composite_score BETWEEN -0.05 AND 0.05 THEN 1 ELSE 0 END),
                                    SUM(CASE WHEN composite_score < -0.05 THEN 1 ELSE 0 END)
                             FROM posts
                             WHERE sentiment_processed > 0
                             """):
            pos, neu, neg = row
            total_s = pos + neu + neg
            print(f"  Positive (>0.05):  {pos:>8,} ({pos / total_s * 100:.1f}%)")
            print(f"  Neutral:           {neu:>8,} ({neu / total_s * 100:.1f}%)")
            print(f"  Negative (<-0.05): {neg:>8,} ({neg / total_s * 100:.1f}%)")

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Sentiment Scoring (VADER + FinBERT)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sentiment_scorer.py                      # Score all (VADER + FinBERT)
  python sentiment_scorer.py --vader-only         # VADER only (fast, no GPU)
  python sentiment_scorer.py --batch-size 16      # Smaller batches (low GPU mem)
  python sentiment_scorer.py --rescore            # Redo all scoring
  python sentiment_scorer.py --export-only        # Export CSVs
  python sentiment_scorer.py --status             # Show stats
        """,
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"FinBERT batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--rescore", action="store_true",
                        help="Rescore all posts from scratch")
    parser.add_argument("--vader-only", action="store_true",
                        help="Only use VADER (fast, no GPU needed)")
    parser.add_argument("--export-only", action="store_true",
                        help="Skip scoring, just export CSVs")
    parser.add_argument("--status", action="store_true",
                        help="Show scoring statistics")

    args = parser.parse_args()

    if args.status:
        print_status()
        return

    if args.export_only:
        export_csvs()
        return

    run_scoring(
        batch_size=args.batch_size,
        rescore=args.rescore,
        vader_only=args.vader_only,
    )
    export_csvs()


if __name__ == "__main__":
    main()

