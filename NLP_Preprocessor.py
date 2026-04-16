import argparse
import csv
import logging
import re
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

DB_PATH = Path( "data") / "project.db"
EXPORT_DIR = Path("exports")
LOG_DIR =  Path("logs")

DEFAULT_BATCH_SIZE = 200
MIN_WORD_COUNT = 5

CRYPTO_SLANG = {
    r'\bhodl\b':        'hold',
    r'\bhodling\b':     'holding',
    r'\bhodler\b':      'holder',
    r'\bhodlers\b':     'holders',
    r'\bfud\b':         'fear uncertainty doubt',
    r'\brekt\b':        'wrecked',
    r'\bgm\b':          'good morning',
    r'\bwagmi\b':       'we are going to make it',
    r'\bngmi\b':        'not going to make it',
    r'\bfomo\b':        'fear of missing out',
    r'\bbtfd\b':        'buy the dip',
    r'\bbuidl\b':       'build',
    r'\bsafu\b':        'safe',
    r'\bnocoiner\b':    'non crypto investor',
    r'\bnoob\b':        'beginner',
    r'\bwhale\b':       'large investor',
    r'\bwhales\b':      'large investors',
    r'\brug\s*pull\b':  'scam',
    r'\brugged\b':      'scammed',
    r'\brug\b':         'scam',
    r'\bpump\b':        'price surge',
    r'\bpumping\b':     'price surging',
    r'\bdump\b':        'price crash',
    r'\bdumping\b':     'price crashing',
    r'\bmoon\b':        'price surge',
    r'\bmooning\b':     'price surging',
    r'\bape\b':         'invest recklessly',
    r'\baped\b':        'invested recklessly',
    r'\baping\b':       'investing recklessly',
    r'\bbullish\b':     'optimistic',
    r'\bbearish\b':     'pessimistic',
    r'\bbull\b':        'optimist',
    r'\bbear\b':        'pessimist',
    r'\bdegen\b':       'speculator',
    r'\bdegens\b':      'speculators',
    r'\bgas\b':         'fee',
    r'\bgwei\b':        'fee',
    r'\byield\b':       'return',
    r'\bairdrop\b':     'token distribution',
    r'\bstaking\b':     'staking',
    r'\bflippening\b':  'overtaking',
    r'\baltseason\b':   'altcoin rally',
    r'\bshitcoin\b':    'bad coin',
    r'\bshitcoins\b':   'bad coins',
    r'\bmemecoin\b':    'meme coin',
    r'\bmemecoins\b':   'meme coins',
}

COMPILED_SLANG = {re.compile(k, re.IGNORECASE): v for k, v in CRYPTO_SLANG.items()}
KEEP_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "nlp_preprocessor.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("PRAGMA table_info(posts);")
    existing_cols = {row[1] for row in c.fetchall()}

    nlp_columns = {
        "text_normalised": "TEXT",
        "tokens": "TEXT",
        "lemmas": "TEXT",
        "lemmas_clean": "TEXT",
        "pos_filtered": "TEXT",
        "pos_tags": "TEXT",
        "entities": "TEXT",
        "adj_count": "INTEGER DEFAULT 0",
        "verb_count": "INTEGER DEFAULT 0",
        "noun_count": "INTEGER DEFAULT 0",
        "unique_lemmas": "INTEGER DEFAULT 0",
        "nlp_processed": "INTEGER DEFAULT 0",
    }

    for col, dtype in nlp_columns.items():
        if col not in existing_cols:
            c.execute(f"ALTER TABLE posts ADD COLUMN {col} {dtype}")
            logger.info(f"  Added column: {col}")

    conn.commit()
    return conn

def normalise_slang(text: str) -> str:
    for pattern, replacement in COMPILED_SLANG.items():
        text = pattern.sub(replacement, text)
    return text

def load_spacy():
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser"])
        logger.info("Loaded en_core_web_sm (parser disabled)")
    except OSError:
        logger.error(
            "spaCy model not found! Install it with:\n"
            "  python -m spacy download en_core_web_sm"
        )
        sys.exit(1)

    nlp.max_length = 200000
    return nlp

def process_text(nlp, text: str) -> dict:
    doc  = nlp(text)
    tokens = []
    lemmas = []
    pos_filtered = []
    pos_tags = []
    adj_count = 0
    verb_count = 0
    noun_count = 0

    for token in doc:
        if token.is_punct or token.is_space:
            continue
        tok_text = token.text.lower()
        lem_text = token.lemma_.lower()

        tokens.append(tok_text)
        pos_tags.append([tok_text,token.pos_])

        if not token.is_stop:
            lemmas.append(lem_text)
            if token.pos_ in KEEP_POS:
                pos_filtered.append(lem_text)

        if token.pos_ == "ADJ":
            adj_count += 1
        elif token.pos_ == "VERB":
            verb_count += 1
        elif token.pos_ == "NOUN":
            noun_count += 1

    entities = [[ent.text, ent.label_] for ent in doc.ents]
    lemmas_clean = " ".join(lemmas)

    return {
        "tokens": tokens,
        "lemmas": lemmas,
        "lemmas_clean": lemmas_clean,
        "pos_filtered": pos_filtered,
        "pos_tags": pos_tags,
        "entities": entities,
        "adj_count": adj_count,
        "verb_count": verb_count,
        "noun_count": noun_count,
        "unique_lemmas": len(set(lemmas)),
    }

def process_batch(conn: sqlite3.Connection, nlp, batch: list[tuple]) -> tuple[int, int]:
    import json
    c = conn.cursor()
    processed = 0
    filtered = 0

    for row in batch:
        post_id = row[0]
        text_clean = row[1] or ""
        word_count = row[2] or 0
        if word_count < MIN_WORD_COUNT:
            c.execute("""
                      UPDATE posts
                      SET nlp_processed = -1
                      WHERE id = ?
                      """, (post_id,))
            filtered += 1
            continue

        text_normalised = normalise_slang(text_clean)
        nlp_result = process_text(nlp, text_normalised)
        c.execute("""
                  UPDATE posts
                  SET text_normalised = ?,
                      tokens          = ?,
                      lemmas          = ?,
                      lemmas_clean    = ?,
                      pos_filtered    = ?,
                      pos_tags        = ?,
                      entities        = ?,
                      adj_count       = ?,
                      verb_count      = ?,
                      noun_count      = ?,
                      unique_lemmas   = ?,
                      nlp_processed   = 1
                  WHERE id = ?
                  """, (
                      text_normalised,
                      json.dumps(nlp_result["tokens"]),
                      json.dumps(nlp_result["lemmas"]),
                      nlp_result["lemmas_clean"],
                      json.dumps(nlp_result["pos_filtered"]),
                      json.dumps(nlp_result["pos_tags"]),
                      json.dumps(nlp_result["entities"]),
                      nlp_result["adj_count"],
                      nlp_result["verb_count"],
                      nlp_result["noun_count"],
                      nlp_result["unique_lemmas"],
                      post_id,
                  ))
        processed += 1

    conn.commit()
    return processed, filtered

def run_preprocessing(batch_size: int = DEFAULT_BATCH_SIZE, reprocess: bool = False):
    conn = init_db()
    nlp = load_spacy()
    c = conn.cursor()
    if reprocess:
        c.execute("UPDATE posts SET nlp_processed = 0")
        conn.commit()
        c.execute("SELECT COUNT(*) FROM posts")
    else:
        c.execute("SELECT COUNT(*) FROM posts WHERE nlp_processed = 0")

    total = c.fetchone()[0]

    if total == 0:
        logger.info("No posts to process!")
        conn.close()
        return

    logger.info(f"{'='*60}")
    logger.info(f"NLP PREPROCESSING — {total:,} posts to process")
    logger.info(f"Batch size: {batch_size} | Min words: {MIN_WORD_COUNT}")
    logger.info(f"Slang terms: {len(CRYPTO_SLANG)} | POS filter: {KEEP_POS}")
    logger.info(f"{'='*60}")

    total_processed = 0
    total_filtered = 0
    batch_num = 0
    start_time = time.time()

    while True:
        if reprocess:
            c.execute("""
                SELECT id, text_clean, word_count FROM posts
                WHERE nlp_processed = 0
                LIMIT ?
            """, (batch_size,))
        else:
            c.execute("""
                SELECT id, text_clean, word_count FROM posts
                WHERE nlp_processed = 0
                LIMIT ?
            """, (batch_size,))

        batch = c.fetchall()
        if not batch:
            break

        batch_num += 1
        processed, filtered = process_batch(conn, nlp, batch)
        total_processed += processed
        total_filtered += filtered

        elapsed = time.time() - start_time
        rate = total_processed / elapsed if elapsed > 0 else 0
        remaining = (total - total_processed - total_filtered) / rate if rate > 0 else 0

        logger.info(
            f"  Batch {batch_num}: {processed} processed, {filtered} filtered | "
            f"Total: {total_processed + total_filtered}/{total} | "
            f"Rate: {rate:.0f} posts/sec | ETA: {remaining:.0f}s"
        )

    elapsed = time.time() - start_time

    logger.info(f"\n{'='*60}")
    logger.info(f"PREPROCESSING COMPLETE")
    logger.info(f"Processed: {total_processed:,} posts")
    logger.info(f"Filtered:  {total_filtered:,} posts (< {MIN_WORD_COUNT} words)")
    logger.info(f"Time:      {elapsed:.1f}s ({total_processed/elapsed:.0f} posts/sec)" if elapsed > 0 else "")
    logger.info(f"{'='*60}")

    conn.close()


def run_tfidf():
    from sklearn.feature_extraction.text import TfidfVectorizer

    conn = sqlite3.connect(str(DB_PATH))
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_sql_query("""
                           SELECT id, subreddit, coin_target, lemmas_clean
                           FROM posts
                           WHERE nlp_processed = 1
                             AND lemmas_clean IS NOT NULL
                             AND lemmas_clean != ''
                           """, conn)

    if df.empty:
        logger.warning("No processed posts for TF-IDF!")
        conn.close()
        return

    logger.info(f"\n{'=' * 60}")
    logger.info(f"TF-IDF ANALYSIS — {len(df):,} posts")
    logger.info(f"{'=' * 60}")

    top_n = 30

    all_results = []
    logger.info("\nTF-IDF by coin_target:")
    for coin in df["coin_target"].unique():
        coin_docs = df[df["coin_target"] == coin]["lemmas_clean"].tolist()
        if len(coin_docs) < 5:
            logger.info(f"  {coin}: skipped (only {len(coin_docs)} posts)")
            continue

        tfidf = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2),
        )
        matrix = tfidf.fit_transform(coin_docs)
        feature_names = tfidf.get_feature_names_out()

        # Mean TF-IDF score per term across all docs in this group
        mean_scores = matrix.mean(axis=0).A1
        top_indices = mean_scores.argsort()[::-1][:top_n]

        logger.info(f"  {coin} ({len(coin_docs)} posts): top terms = "
                    f"{', '.join(feature_names[i] for i in top_indices[:5])}")

        for rank, idx in enumerate(top_indices, 1):
            all_results.append({
                "group_type": "coin_target",
                "group_name": coin,
                "rank": rank,
                "term": feature_names[idx],
                "tfidf_score": round(mean_scores[idx], 6),
                "doc_count": len(coin_docs),
            })
    logger.info("\nTF-IDF by subreddit:")
    for sub in df["subreddit"].unique():
        sub_docs = df[df["subreddit"] == sub]["lemmas_clean"].tolist()
        if len(sub_docs) < 5:
            logger.info(f"  r/{sub}: skipped (only {len(sub_docs)} posts)")
            continue

        tfidf = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2),
        )
        matrix = tfidf.fit_transform(sub_docs)
        feature_names = tfidf.get_feature_names_out()

        mean_scores = matrix.mean(axis=0).A1
        top_indices = mean_scores.argsort()[::-1][:top_n]

        logger.info(f"  r/{sub} ({len(sub_docs)} posts): top terms = "
                    f"{', '.join(feature_names[i] for i in top_indices[:5])}")

        for rank, idx in enumerate(top_indices, 1):
            all_results.append({
                "group_type": "subreddit",
                "group_name": sub,
                "rank": rank,
                "term": feature_names[idx],
                "tfidf_score": round(mean_scores[idx], 6),
                "doc_count": len(sub_docs),
            })
    if all_results:
        tfidf_df = pd.DataFrame(all_results)
        tfidf_path = EXPORT_DIR / "nlp_tfidf_terms.csv"
        tfidf_df.to_csv(tfidf_path, index=False, quoting=csv.QUOTE_MINIMAL)
        logger.info(f"\nTF-IDF results exported: {tfidf_path}")

    conn.close()

def export_csvs():
    conn = sqlite3.connect(str(DB_PATH))
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_sql_query("""
                           SELECT id,
                                  subreddit,
                                  coin_target,
                                  title,
                                  text_clean,
                                  text_normalised,
                                  word_count,
                                  tokens,
                                  lemmas,
                                  lemmas_clean,
                                  pos_filtered,
                                  pos_tags,
                                  entities,
                                  adj_count,
                                  verb_count,
                                  noun_count,
                                  unique_lemmas,
                                  flair,
                                  datetime_utc, date, time, upvotes, upvote_ratio, num_comments, tickers, post_type, nlp_processed
                           FROM posts
                           WHERE nlp_processed = 1
                           ORDER BY datetime_utc
                           """, conn)

    if df.empty:
        logger.warning("No processed posts to export!")
        conn.close()
        return

    logger.info(f"Exporting {len(df):,} NLP-processed posts...")
    nlp_path = EXPORT_DIR / "nlp_preprocessed.csv"
    df.to_csv(nlp_path, index=False, quoting=csv.QUOTE_ALL)
    logger.info(f"  Full NLP export: {nlp_path}")
    lite_cols = [
        "id", "subreddit", "coin_target", "text_clean", "lemmas_clean",
        "date", "upvotes", "num_comments", "word_count", "unique_lemmas",
    ]
    lite_df = df[lite_cols]
    lite_path = EXPORT_DIR / "nlp_lite.csv"
    lite_df.to_csv(lite_path, index=False, quoting=csv.QUOTE_ALL)
    logger.info(f"  Lite export: {lite_path}")

    conn.close()
    logger.info("CSV export complete!")

def print_status():
    if not DB_PATH.exists():
        print("No database found.")
        return

    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("PRAGMA table_info(posts)")
    cols = {row[1] for row in c.fetchall()}
    if "nlp_processed" not in cols:
        print("NLP preprocessing has not been run yet.")
        conn.close()
        return

    print(f"\n{'=' * 60}")
    print("NLP PREPROCESSING STATUS")
    print(f"{'=' * 60}")

    c.execute("SELECT COUNT(*) FROM posts")
    total = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM posts WHERE nlp_processed = 1")
    processed = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM posts WHERE nlp_processed = -1")
    filtered = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM posts WHERE nlp_processed = 0")
    pending = c.fetchone()[0]

    print(f"\nTotal posts:      {total:,}")
    print(f"NLP processed:    {processed:,}")
    print(f"Filtered (<{MIN_WORD_COUNT}w):  {filtered:,}")
    print(f"Pending:          {pending:,}")

    if processed > 0:
        c.execute("""
                  SELECT ROUND(AVG(adj_count), 1),
                         ROUND(AVG(verb_count), 1),
                         ROUND(AVG(noun_count), 1),
                         ROUND(AVG(unique_lemmas), 1),
                         ROUND(AVG(word_count), 1)
                  FROM posts
                  WHERE nlp_processed = 1
                  """)
        avg_adj, avg_verb, avg_noun, avg_unique, avg_words = c.fetchone()

        print(f"\nAverage per post:")
        print(f"  Words:         {avg_words}")
        print(f"  Unique lemmas: {avg_unique}")
        print(f"  Adjectives:    {avg_adj}")
        print(f"  Verbs:         {avg_verb}")
        print(f"  Nouns:         {avg_noun}")

        # Per coin
        print(f"\n{'─' * 45}")
        print(f"{'Coin':<10} {'Processed':>10} {'Filtered':>10} {'Avg Words':>10}")
        print(f"{'─' * 45}")

        for coin_row in c.execute("""
                                  SELECT coin_target,
                                         SUM(CASE WHEN nlp_processed = 1 THEN 1 ELSE 0 END),
                                         SUM(CASE WHEN nlp_processed = -1 THEN 1 ELSE 0 END),
                                         ROUND(AVG(CASE WHEN nlp_processed = 1 THEN word_count END), 1)
                                  FROM posts
                                  GROUP BY coin_target
                                  ORDER BY coin_target
                                  """):
            coin, proc, filt, avg_w = coin_row
            print(f"{coin:<10} {proc or 0:>10,} {filt or 0:>10,} {avg_w or 0:>10}")

        import json
        c.execute("""
                  SELECT entities
                  FROM posts
                  WHERE nlp_processed = 1
                    AND entities IS NOT NULL
                    AND entities != '[]'
            LIMIT 5000
                  """)

        entity_counter = Counter()
        for (ents_json,) in c.fetchall():
            try:
                ents = json.loads(ents_json)
                for text, label in ents:
                    entity_counter[(text, label)] += 1
            except (json.JSONDecodeError, ValueError):
                pass

        if entity_counter:
            print(f"\nTop 15 named entities (from up to 5000 posts):")
            for (text, label), count in entity_counter.most_common(15):
                print(f"  {text:<25} {label:<10} {count:>5} mentions")

    conn.close()

def main():
    parser = argparse.ArgumentParser(description="NLP preprocessing", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Posts per batch (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--reprocess", action="store_true",
                        help="Reprocess all posts (ignore existing NLP data)")
    parser.add_argument("--export-only", action="store_true",
                        help="Skip processing, just export CSVs")
    parser.add_argument("--tfidf-only", action="store_true",
                        help="Only run TF-IDF analysis on existing processed data")
    parser.add_argument("--status", action="store_true",
                        help="Show preprocessing statistics")

    args = parser.parse_args()
    if args.status:
        print_status()
        return
    if args.tfidf_only:
        run_tfidf()
        return
    if args.export_only:
        export_csvs()
        return

    run_preprocessing(batch_size=args.batch_size, reprocess=args.reprocess)
    run_tfidf()
    export_csvs()

if __name__ == "__main__":
    main()