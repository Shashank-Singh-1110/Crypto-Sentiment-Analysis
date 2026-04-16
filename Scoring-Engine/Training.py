import argparse
import csv
import json
import logging
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path("data") / "project.db"
EXPORT_DIR = Path("data") / "advanced_nlp"
LOG_DIR = Path("logs")

DEFAULT_BATCH_SIZE = 32
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "advanced_nlp.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("PRAGMA table_info(posts)")
    existing_cols = {row[1] for row in c.fetchall()}

    adv_columns = {
        "topic_id": "INTEGER DEFAULT -1",
        "topic_label": "TEXT",
        "topic_probability": "REAL DEFAULT 0.0",
        "emotion_label": "TEXT",
        "emotion_score": "REAL DEFAULT 0.0",
        "emotion_all": "TEXT",
        "ner_orgs": "TEXT",
        "ner_people": "TEXT",
        "ner_crypto": "TEXT",
        "topics_processed": "INTEGER DEFAULT 0",
        "emotions_processed": "INTEGER DEFAULT 0",
        "ner_processed": "INTEGER DEFAULT 0",
    }

    for col, dtype in adv_columns.items():
        if col not in existing_cols:
            c.execute(f"ALTER TABLE posts ADD COLUMN {col} {dtype}")
            logger.info(f"  Added column: {col}")

    conn.commit()
    return conn


def run_topic_modeling(num_topics: int = None):
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer

    conn = init_db()
    c = conn.cursor()
    df = pd.read_sql_query("""
                           SELECT id, lemmas_clean, coin_target, subreddit
                           FROM posts
                           WHERE nlp_processed = 1
                             AND lemmas_clean IS NOT NULL
                             AND lemmas_clean != ''
                           """, conn)

    if df.empty:
        logger.warning("No processed posts for topic modeling!")
        conn.close()
        return

    logger.info(f"{'=' * 60}")
    logger.info(f"BERTOPIC TOPIC MODELING — {len(df):,} posts")
    logger.info(f"{'=' * 60}")

    docs = df["lemmas_clean"].tolist()

    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        min_df=5,
        max_df=0.95,
    )
    nr_topics = num_topics if num_topics else "auto"
    topic_model = BERTopic(
        vectorizer_model=vectorizer,
        nr_topics=nr_topics,
        min_topic_size=10,
        verbose=True,
        calculate_probabilities=True,
    )

    logger.info("Fitting BERTopic model...")
    start = time.time()
    topics, probs = topic_model.fit_transform(docs)
    elapsed = time.time() - start
    logger.info(f"BERTopic fitted in {elapsed:.1f}s")
    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info) - 1
    logger.info(f"Discovered {n_topics} topics (+ outlier topic -1)")

    topic_labels = {}
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            topic_labels[tid] = "outlier"
        else:
            topic_words = topic_model.get_topic(tid)
            if topic_words:
                label = "_".join([w for w, _ in topic_words[:3]])
                topic_labels[tid] = label
            else:
                topic_labels[tid] = f"topic_{tid}"

    logger.info("Storing topic assignments...")
    for i, post_id in enumerate(df["id"].tolist()):
        topic_id = int(topics[i])
        prob = float(probs[i].max()) if hasattr(probs[i], 'max') else float(probs[i]) if probs is not None else 0.0
        label = topic_labels.get(topic_id, f"topic_{topic_id}")

        c.execute("""
                  UPDATE posts
                  SET topic_id          = ?,
                      topic_label       = ?,
                      topic_probability = ?,
                      topics_processed  = 1
                  WHERE id = ?
                  """, (topic_id, label, round(prob, 6), post_id))

    conn.commit()
    logger.info(f"\nTop 10 topics:")
    for _, row in topic_info.head(11).iterrows():
        tid = row["Topic"]
        count = row["Count"]
        label = topic_labels.get(tid, "")
        logger.info(f"  Topic {tid:>3} ({count:>5} posts): {label}")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    topic_info_path = EXPORT_DIR / "topic_info.csv"
    topic_info["label"] = topic_info["Topic"].map(topic_labels)
    topic_info.to_csv(topic_info_path, index=False)
    logger.info(f"\nTopic info saved: {topic_info_path}")

    conn.close()
    logger.info("Topic modeling complete!")


def run_emotion_detection(batch_size: int = DEFAULT_BATCH_SIZE):
    import torch
    from transformers import pipeline

    conn = init_db()
    c = conn.cursor()
    c.execute("""
              SELECT COUNT(*)
              FROM posts
              WHERE nlp_processed = 1
                AND emotions_processed = 0
              """)
    total = c.fetchone()[0]

    if total == 0:
        logger.info("No posts for emotion detection!")
        conn.close()
        return

    logger.info(f"{'=' * 60}")
    logger.info(f"EMOTION DETECTION — {total:,} posts")
    logger.info(f"Model: {EMOTION_MODEL}")
    logger.info(f"{'=' * 60}")
    device = 0 if torch.cuda.is_available() else -1
    device_name = "cuda" if device == 0 else "cpu"
    logger.info(f"Loading emotion model on {device_name}...")

    classifier = pipeline(
        "text-classification",
        model=EMOTION_MODEL,
        top_k=None,
        device=device,
        truncation=True,
        max_length=512,
    )

    total_processed = 0
    batch_num = 0
    start_time = time.time()

    while True:
        c.execute("""
                  SELECT id, text_clean
                  FROM posts
                  WHERE nlp_processed = 1
                    AND emotions_processed = 0 LIMIT ?
                  """, (batch_size,))

        batch = c.fetchall()
        if not batch:
            break

        batch_num += 1
        post_ids = [row[0] for row in batch]
        texts = [row[1][:1000] or "" for row in batch]  # truncate for speed
        results = classifier(texts)

        for i, post_id in enumerate(post_ids):
            emotions = results[i]
            dominant = max(emotions, key=lambda x: x["score"])
            emotion_label = dominant["label"]
            emotion_score = dominant["score"]
            emotion_dict = {e["label"]: round(e["score"], 6) for e in emotions}

            c.execute("""
                      UPDATE posts
                      SET emotion_label      = ?,
                          emotion_score      = ?,
                          emotion_all        = ?,
                          emotions_processed = 1
                      WHERE id = ?
                      """, (emotion_label, round(emotion_score, 6),
                            json.dumps(emotion_dict), post_id))

        conn.commit()
        total_processed += len(batch)

        elapsed = time.time() - start_time
        rate = total_processed / elapsed if elapsed > 0 else 0
        remaining = (total - total_processed) / rate if rate > 0 else 0

        logger.info(
            f"  Batch {batch_num}: {len(batch)} classified | "
            f"Total: {total_processed}/{total} | "
            f"Rate: {rate:.1f}/sec | ETA: {remaining:.0f}s"
        )

    elapsed = time.time() - start_time
    logger.info(f"\nEmotion detection complete: {total_processed:,} posts in {elapsed:.1f}s")
    conn.close()


CRYPTO_ENTITIES = {
    "bitcoin", "btc", "ethereum", "eth", "solana", "sol",
    "dogecoin", "doge", "shib", "shiba", "bnb", "binance",
    "cardano", "ada", "xrp", "ripple", "polkadot", "dot",
    "avalanche", "avax", "polygon", "matic", "chainlink", "link",
    "uniswap", "uni", "aave", "maker", "mkr", "tether", "usdt",
    "usdc", "dai", "coinbase", "kraken", "ftx", "opensea",
    "metamask", "ledger", "trezor",
}

KNOWN_ORGS = {
    "sec", "cftc", "federal reserve", "fed", "imf", "world bank",
    "jp morgan", "goldman sachs", "blackrock", "fidelity",
    "grayscale", "microstrategy", "tesla", "paypal", "visa",
    "mastercard", "square", "block", "stripe", "robinhood",
}

KNOWN_PEOPLE = {
    "elon musk", "musk", "vitalik", "vitalik buterin",
    "cz", "changpeng zhao", "gary gensler", "gensler",
    "michael saylor", "saylor", "satoshi", "satoshi nakamoto",
    "cathie wood", "sam bankman", "sbf", "brian armstrong",
    "jack dorsey", "powell", "jerome powell",
}


def run_ner_extraction():
    import spacy

    conn = init_db()
    c = conn.cursor()
    c.execute("""
              SELECT COUNT(*)
              FROM posts
              WHERE nlp_processed = 1
                AND ner_processed = 0
              """)
    total = c.fetchone()[0]

    if total == 0:
        logger.info("No posts for NER extraction!")
        conn.close()
        return

    logger.info(f"{'=' * 60}")
    logger.info(f"NER EXTRACTION — {total:,} posts")
    logger.info(f"{'=' * 60}")
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser"])
        nlp.max_length = 200_000
    except OSError:
        logger.error("spaCy model not found! Run: python -m spacy download en_core_web_sm")
        conn.close()
        return

    total_processed = 0
    batch_size = 200
    batch_num = 0
    start_time = time.time()

    while True:
        c.execute("""
                  SELECT id, text_clean
                  FROM posts
                  WHERE nlp_processed = 1
                    AND ner_processed = 0 LIMIT ?
                  """, (batch_size,))

        batch = c.fetchall()
        if not batch:
            break

        batch_num += 1

        for post_id, text in batch:
            text = text or ""
            text_lower = text.lower()
            doc = nlp(text)
            spacy_orgs = set()
            spacy_people = set()

            for ent in doc.ents:
                if ent.label_ == "ORG":
                    spacy_orgs.add(ent.text)
                elif ent.label_ == "PERSON":
                    spacy_people.add(ent.text)

            found_crypto = set()
            for entity in CRYPTO_ENTITIES:
                if entity in text_lower:
                    found_crypto.add(entity)
            for org in KNOWN_ORGS:
                if org in text_lower:
                    spacy_orgs.add(org.title())
            for person in KNOWN_PEOPLE:
                if person in text_lower:
                    spacy_people.add(person.title())

            c.execute("""
                      UPDATE posts
                      SET ner_orgs      = ?,
                          ner_people    = ?,
                          ner_crypto    = ?,
                          ner_processed = 1
                      WHERE id = ?
                      """, (
                          json.dumps(sorted(spacy_orgs)),
                          json.dumps(sorted(spacy_people)),
                          json.dumps(sorted(found_crypto)),
                          post_id,
                      ))

        conn.commit()
        total_processed += len(batch)

        elapsed = time.time() - start_time
        rate = total_processed / elapsed if elapsed > 0 else 0

        logger.info(
            f"  Batch {batch_num}: {len(batch)} extracted | "
            f"Total: {total_processed}/{total} | "
            f"Rate: {rate:.0f}/sec"
        )

    elapsed = time.time() - start_time
    logger.info(f"\nNER extraction complete: {total_processed:,} posts in {elapsed:.1f}s")
    conn.close()


def export_csvs():
    conn = sqlite3.connect(str(DB_PATH))
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    df_topics = pd.read_sql_query("""
                                  SELECT id,
                                         subreddit,
                                         coin_target, date, topic_id, topic_label, topic_probability, upvotes, num_comments
                                  FROM posts
                                  WHERE topics_processed = 1
                                  ORDER BY date
                                  """, conn)

    if not df_topics.empty:
        path = EXPORT_DIR / "nlp_topics.csv"
        df_topics.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
        logger.info(f"Topics: {path} ({len(df_topics)} posts)")
        topic_daily = df_topics.groupby(["date", "topic_id", "topic_label"]).agg(
            post_count=("id", "count"),
        ).reset_index()
        trend_path = EXPORT_DIR / "topic_trends_daily.csv"
        topic_daily.to_csv(trend_path, index=False)
        logger.info(f"Topic trends: {trend_path} ({len(topic_daily)} rows)")
    df_emotions = pd.read_sql_query("""
                                    SELECT id,
                                           subreddit,
                                           coin_target, date, emotion_label, emotion_score, emotion_all, upvotes, num_comments
                                    FROM posts
                                    WHERE emotions_processed = 1
                                    ORDER BY date
                                    """, conn)

    if not df_emotions.empty:
        path = EXPORT_DIR / "nlp_emotions.csv"
        df_emotions.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
        logger.info(f"Emotions: {path} ({len(df_emotions)} posts)")

        # Emotion timeline (daily distribution)
        emotion_daily = df_emotions.groupby(["date", "coin_target", "emotion_label"]).agg(
            post_count=("id", "count"),
            avg_score=("emotion_score", "mean"),
        ).reset_index()
        timeline_path = EXPORT_DIR / "emotion_timeline_daily.csv"
        emotion_daily.to_csv(timeline_path, index=False)
        logger.info(f"Emotion timeline: {timeline_path} ({len(emotion_daily)} rows)")
    df_ner = pd.read_sql_query("""
                               SELECT id,
                                      subreddit,
                                      coin_target, date, ner_orgs, ner_people, ner_crypto
                               FROM posts
                               WHERE ner_processed = 1
                               ORDER BY date
                               """, conn)

    if not df_ner.empty:
        path = EXPORT_DIR / "nlp_ner.csv"
        df_ner.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
        logger.info(f"NER: {path} ({len(df_ner)} posts)")
        org_counter = Counter()
        people_counter = Counter()
        crypto_counter = Counter()

        for _, row in df_ner.iterrows():
            try:
                for org in json.loads(row["ner_orgs"] or "[]"):
                    org_counter[org] += 1
                for person in json.loads(row["ner_people"] or "[]"):
                    people_counter[person] += 1
                for crypto in json.loads(row["ner_crypto"] or "[]"):
                    crypto_counter[crypto] += 1
            except (json.JSONDecodeError, TypeError):
                pass

        entities = []
        for name, count in org_counter.most_common(50):
            entities.append({"entity": name, "type": "ORG", "count": count})
        for name, count in people_counter.most_common(50):
            entities.append({"entity": name, "type": "PERSON", "count": count})
        for name, count in crypto_counter.most_common(50):
            entities.append({"entity": name, "type": "CRYPTO", "count": count})

        if entities:
            ent_path = EXPORT_DIR / "ner_entity_summary.csv"
            pd.DataFrame(entities).to_csv(ent_path, index=False)
            logger.info(f"Entity summary: {ent_path} ({len(entities)} entities)")

    conn.close()
    logger.info("Export complete!")


def print_status():
    if not DB_PATH.exists():
        print("No database found.")
        return

    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("PRAGMA table_info(posts)")
    cols = {row[1] for row in c.fetchall()}

    print(f"\n{'=' * 60}")
    print("ADVANCED NLP STATUS")
    print(f"{'=' * 60}")

    if "topics_processed" in cols:
        c.execute("SELECT COUNT(*) FROM posts WHERE topics_processed = 1")
        t_done = c.fetchone()[0]
        c.execute("SELECT COUNT(DISTINCT topic_id) FROM posts WHERE topics_processed = 1 AND topic_id != -1")
        n_topics = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM posts WHERE topics_processed = 1 AND topic_id = -1")
        n_outliers = c.fetchone()[0]

        print(f"\nTOPICS: {t_done:,} posts processed, {n_topics} topics discovered, {n_outliers:,} outliers")

        if t_done > 0:
            print(f"\n  Top 10 topics:")
            for row in c.execute("""
                                 SELECT topic_id, topic_label, COUNT(*) as cnt
                                 FROM posts
                                 WHERE topics_processed = 1
                                   AND topic_id != -1
                                 GROUP BY topic_id
                                 ORDER BY cnt DESC LIMIT 10
                                 """):
                tid, label, cnt = row
                print(f"    Topic {tid:>3} ({cnt:>5} posts): {label}")
    if "emotions_processed" in cols:
        c.execute("SELECT COUNT(*) FROM posts WHERE emotions_processed = 1")
        e_done = c.fetchone()[0]

        print(f"\nEMOTIONS: {e_done:,} posts processed")

        if e_done > 0:
            print(f"\n  Emotion distribution:")
            for row in c.execute("""
                                 SELECT emotion_label,
                                        COUNT(*),
                                        ROUND(COUNT(*) * 100.0 / ?, 1),
                                        ROUND(AVG(emotion_score), 4)
                                 FROM posts
                                 WHERE emotions_processed = 1
                                 GROUP BY emotion_label
                                 ORDER BY COUNT(*) DESC
                                 """, (e_done,)):
                label, cnt, pct, avg = row
                print(f"    {label:<12} {cnt:>7,} ({pct:>5.1f}%)  avg_confidence: {avg}")

            # Per coin emotion breakdown
            print(f"\n  Dominant emotion per coin:")
            for row in c.execute("""
                                 SELECT coin_target, emotion_label, COUNT(*) as cnt
                                 FROM posts
                                 WHERE emotions_processed = 1
                                 GROUP BY coin_target, emotion_label
                                 ORDER BY coin_target, cnt DESC
                                 """):
                coin, label, cnt = row
                # Only show top emotion per coin
                pass

            for row in c.execute("""
                                 SELECT coin_target,
                                        (SELECT emotion_label
                                         FROM posts p2
                                         WHERE p2.coin_target = p1.coin_target
                                           AND p2.emotions_processed = 1
                                         GROUP BY emotion_label
                                         ORDER BY COUNT(*) DESC LIMIT 1) as top_emo,
                    COUNT(*)
                                 FROM posts p1
                                 WHERE emotions_processed = 1
                                 GROUP BY coin_target
                                 """):
                coin, emo, cnt = row
                print(f"    {coin:<8} → {emo} ({cnt} posts)")

    if "ner_processed" in cols:
        c.execute("SELECT COUNT(*) FROM posts WHERE ner_processed = 1")
        n_done = c.fetchone()[0]

        print(f"\nNER: {n_done:,} posts processed")

        if n_done > 0:
            c.execute("SELECT ner_orgs, ner_people, ner_crypto FROM posts WHERE ner_processed = 1 LIMIT 5000")
            org_c = Counter()
            people_c = Counter()
            crypto_c = Counter()

            for orgs, people, crypto in c.fetchall():
                try:
                    for o in json.loads(orgs or "[]"):
                        org_c[o] += 1
                    for p in json.loads(people or "[]"):
                        people_c[p] += 1
                    for cr in json.loads(crypto or "[]"):
                        crypto_c[cr] += 1
                except (json.JSONDecodeError, TypeError):
                    pass

            print(f"\n  Top organisations: {', '.join(f'{o}({c})' for o, c in org_c.most_common(5))}")
            print(f"  Top people:        {', '.join(f'{p}({c})' for p, c in people_c.most_common(5))}")
            print(f"  Top crypto:        {', '.join(f'{cr}({c})' for cr, c in crypto_c.most_common(5))}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Advanced NLP: Topics, Emotions, NER",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python advanced_nlp.py                     # Run all analyses
  python advanced_nlp.py --topics-only       # Only BERTopic
  python advanced_nlp.py --emotions-only     # Only emotion detection
  python advanced_nlp.py --ner-only          # Only NER
  python advanced_nlp.py --num-topics 20     # Force 20 topics
  python advanced_nlp.py --export-only       # Export CSVs
  python advanced_nlp.py --status            # Show stats
        """,
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Emotion model batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--num-topics", type=int, default=None,
                        help="Force specific number of topics (default: auto)")
    parser.add_argument("--topics-only", action="store_true",
                        help="Only run BERTopic")
    parser.add_argument("--emotions-only", action="store_true",
                        help="Only run emotion detection")
    parser.add_argument("--ner-only", action="store_true",
                        help="Only run NER extraction")
    parser.add_argument("--export-only", action="store_true",
                        help="Skip processing, export CSVs")
    parser.add_argument("--status", action="store_true",
                        help="Show statistics")

    args = parser.parse_args()

    if args.status:
        print_status()
        return

    if args.export_only:
        export_csvs()
        return

    run_specific = args.topics_only or args.emotions_only or args.ner_only

    if not run_specific or args.topics_only:
        try:
            run_topic_modeling(num_topics=args.num_topics)
        except Exception as e:
            logger.error(f"Topic modeling failed: {e}")
            logger.error("Try: pip install bertopic hdbscan umap-learn")

    if not run_specific or args.emotions_only:
        try:
            run_emotion_detection(batch_size=args.batch_size)
        except Exception as e:
            logger.error(f"Emotion detection failed: {e}")
            logger.error("Try: pip install transformers torch")

    if not run_specific or args.ner_only:
        try:
            run_ner_extraction()
        except Exception as e:
            logger.error(f"NER extraction failed: {e}")
            logger.error("Try: python -m spacy download en_core_web_sm")

    export_csvs()


if __name__ == "__main__":
    main()
