import argparse
import csv
import json
import logging
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

DB_PATH = Path("data") / "project.db"
EXPORT_DIR = Path("exports")
LOG_DIR = Path ("logs")
COINS = ["BTC", "ETH", "SOL", "DOGE", "SHIB", "BNB"]
SUBREDDIT_COIN_MAP = {
    "Bitcoin":          "BTC",
    "ethereum":         "ETH",
    "solana":           "SOL",
    "dogecoin":         "DOGE",
    "SHIBArmy":         "SHIB",
    "binance":          "BNB",
    "CryptoCurrency":   "MARKET",
    "CryptoMarkets":    "MARKET",
    "Bitcoinmarkets":   "MARKET",
    "CryptoMoonShots":  "MARKET",
    "wallstreetbets":   "MARKET",
}

TICKER_PATTERNS = {
    "BTC":  r'\b(?:BTC|Bitcoin|btc)\b',
    "ETH":  r'\b(?:ETH|Ethereum|Ether|eth)\b',
    "SOL":  r'\b(?:SOL|Solana|sol)\b',
    "DOGE": r'\b(?:DOGE|Dogecoin|doge)\b',
    "SHIB": r'\b(?:SHIB|Shiba\s*Inu|shib)\b',
    "BNB":  r'\b(?:BNB|Binance\s*Coin|bnb)\b',
}

ARCTIC_SHIFT_BASE = "https://arctic-shift.photon-reddit.com/api/posts/search"
REQUEST_DELAY = 1.2
MAX_RETRIES = 5
BATCH_SIZE = 100
DEFAULT_DAYS = 180

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "reddit_collector.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id              TEXT PRIMARY KEY,
            subreddit       TEXT NOT NULL,
            coin_target     TEXT NOT NULL,
            title           TEXT,
            text_clean      TEXT,
            word_count      INTEGER DEFAULT 0,
            flair           TEXT,
            datetime_utc    TEXT NOT NULL,
            date            TEXT NOT NULL,
            time            TEXT NOT NULL,
            upvotes         INTEGER DEFAULT 0,
            upvote_ratio    REAL DEFAULT 0.0,
            num_comments    INTEGER DEFAULT 0,
            tickers         TEXT,
            post_type       TEXT DEFAULT 'text'
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS checkpoints (
            subreddit       TEXT PRIMARY KEY,
            last_timestamp  INTEGER NOT NULL,
            posts_collected INTEGER DEFAULT 0,
            updated_at      TEXT NOT NULL
        )
    """)

    c.execute("CREATE INDEX IF NOT EXISTS idx_posts_date ON posts(date)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_posts_coin ON posts(coin_target)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_posts_subreddit ON posts(subreddit)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_posts_datetime ON posts(datetime_utc)")

    conn.commit()
    return conn

def clean_text(title: str, selftext: str) -> str:
    title = title or ""
    selftext = selftext or ""

    if selftext.lower() in ("[removed]", "[deleted]", ""):
        text = title
    else:
        text = f"{title}. {selftext}"
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'[*_~`#>]+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def detect_tickers(text: str) -> list:
    found = []
    for ticker, pattern in TICKER_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            found.append(ticker)
    return found


def determine_post_type(post: dict) -> str:
    if post.get("is_video"):
        return "video"
    url = post.get("url", "")
    if any(ext in url for ext in [".jpg", ".png", ".gif", ".jpeg", ".webp"]):
        return "image"
    domain = post.get("domain", "")
    if domain and "self." not in domain and "reddit.com" not in domain:
        return "link"
    return "text"


def fetch_posts_arctic_shift(subreddit: str, after: int, before: int) -> list:
    all_posts = []
    current_after = after

    while True:
        params = {
            "subreddit": subreddit,
            "after": current_after,
            "before": before,
            "sort": "asc",
            "limit": BATCH_SIZE,
        }

        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(
                    ARCTIC_SHIFT_BASE,
                    params=params,
                    timeout=30,
                    headers={"User-Agent": "CryptoSentimentResearch/1.0"}
                )
                if resp.status_code == 429:
                    wait = min(2 ** attempt * 5, 60)
                    logger.warning(f"Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if resp.status_code == 400:
                    logger.error(f"Bad request for r/{subreddit}. URL: {resp.url}")
                    logger.error(f"Response: {resp.text[:500]}")
                    return all_posts
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.exceptions.RequestException as e:
                wait = min(2 ** attempt * 2, 30)
                logger.warning(f"Request error (attempt {attempt+1}/{MAX_RETRIES}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"Failed after {MAX_RETRIES} retries for r/{subreddit}")
                    return all_posts
                continue

        posts = data.get("data", [])
        if not posts:
            break

        all_posts.extend(posts)
        last_ts = posts[-1].get("created_utc", 0)
        if isinstance(last_ts, str):
            last_ts = int(float(last_ts))
        if last_ts <= current_after:
            break
        current_after = last_ts

        if len(posts) < BATCH_SIZE:
            break

        time.sleep(REQUEST_DELAY)

    return all_posts


def process_post(post: dict, subreddit: str) -> dict | None:
    post_id = post.get("id")
    if not post_id:
        return None

    title = post.get("title", "")
    selftext = post.get("selftext", "")
    text_clean = clean_text(title, selftext)

    word_count = len(text_clean.split())
    if word_count == 0:
        return None

    created_utc = post.get("created_utc", 0)
    if isinstance(created_utc, str):
        try:
            created_utc = int(float(created_utc))
        except (ValueError, TypeError):
            return None

    dt = datetime.fromtimestamp(created_utc, tz=timezone.utc)

    coin_target = SUBREDDIT_COIN_MAP.get(subreddit, "MARKET")
    tickers = detect_tickers(text_clean)
    flair = post.get("link_flair_text", "") or ""
    post_type = determine_post_type(post)

    return {
        "id":           post_id,
        "subreddit":    subreddit,
        "coin_target":  coin_target,
        "title":        title.strip(),
        "text_clean":   text_clean,
        "word_count":   word_count,
        "flair":        flair,
        "datetime_utc": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "date":         dt.strftime("%Y-%m-%d"),
        "time":         dt.strftime("%H:%M:%S"),
        "upvotes":      post.get("score", 0) or 0,
        "upvote_ratio": post.get("upvote_ratio", 0.0) or 0.0,
        "num_comments": post.get("num_comments", 0) or 0,
        "tickers":      ",".join(tickers) if tickers else "",
        "post_type":    post_type,
    }


def store_posts(conn: sqlite3.Connection, posts: list[dict]) -> int:
    if not posts:
        return 0

    c = conn.cursor()
    inserted = 0
    for p in posts:
        try:
            c.execute("""
                INSERT OR IGNORE INTO posts
                (id, subreddit, coin_target, title, text_clean, word_count,
                 flair, datetime_utc, date, time, upvotes, upvote_ratio,
                 num_comments, tickers, post_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                p["id"], p["subreddit"], p["coin_target"], p["title"],
                p["text_clean"], p["word_count"], p["flair"],
                p["datetime_utc"], p["date"], p["time"],
                p["upvotes"], p["upvote_ratio"], p["num_comments"],
                p["tickers"], p["post_type"],
            ))
            if c.rowcount > 0:
                inserted += 1
        except sqlite3.IntegrityError:
            pass

    conn.commit()
    return inserted


def update_checkpoint(conn: sqlite3.Connection, subreddit: str,
                      last_ts: int, total: int):
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO checkpoints (subreddit, last_timestamp, posts_collected, updated_at)
        VALUES (?, ?, ?, ?)
    """, (subreddit, last_ts, total, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()


def get_checkpoint(conn: sqlite3.Connection, subreddit: str) -> int | None:
    c = conn.cursor()
    c.execute("SELECT last_timestamp FROM checkpoints WHERE subreddit = ?", (subreddit,))
    row = c.fetchone()
    return row[0] if row else None


def collect_subreddit(conn: sqlite3.Connection, subreddit: str,
                      start_ts: int, end_ts: int, resume: bool = True) -> int:
    if resume:
        checkpoint_ts = get_checkpoint(conn, subreddit)
        if checkpoint_ts and checkpoint_ts > start_ts:
            logger.info(f"  Resuming r/{subreddit} from checkpoint "
                        f"({datetime.fromtimestamp(checkpoint_ts, tz=timezone.utc).strftime('%Y-%m-%d')})")
            start_ts = checkpoint_ts

    logger.info(f"  Collecting r/{subreddit}: "
                f"{datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime('%Y-%m-%d')} → "
                f"{datetime.fromtimestamp(end_ts, tz=timezone.utc).strftime('%Y-%m-%d')}")

    raw_posts = fetch_posts_arctic_shift(subreddit, start_ts, end_ts)
    logger.info(f"  Fetched {len(raw_posts)} raw posts from API")

    processed = [p for rp in raw_posts if (p := process_post(rp, subreddit))]

    inserted = store_posts(conn, processed)
    logger.info(f"  Stored {inserted} new posts ({len(processed)} processed, "
                f"{len(processed) - inserted} duplicates)")

    if raw_posts:
        last_ts = max(
            (int(float(rp.get("created_utc", 0))) if isinstance(rp.get("created_utc"), str)
             else rp.get("created_utc", 0))
            for rp in raw_posts
        )
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM posts WHERE subreddit = ?", (subreddit,))
        total = c.fetchone()[0]
        update_checkpoint(conn, subreddit, last_ts, total)

    return inserted


def run_collection(days: int = DEFAULT_DAYS) -> dict:
    conn = init_db()

    now = datetime.now(timezone.utc)
    end_ts = int(now.timestamp())
    start_ts = int((now - timedelta(days=days)).timestamp())

    logger.info(f"{'='*60}")
    logger.info(f"REDDIT DATA COLLECTION — {days} days")
    logger.info(f"Period: {datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime('%Y-%m-%d')} → "
                f"{datetime.fromtimestamp(end_ts, tz=timezone.utc).strftime('%Y-%m-%d')}")
    logger.info(f"Subreddits: {len(SUBREDDIT_COIN_MAP)}")
    logger.info(f"{'='*60}")

    stats = {}
    total_new = 0

    for subreddit in SUBREDDIT_COIN_MAP:
        try:
            new = collect_subreddit(conn, subreddit, start_ts, end_ts)
            stats[subreddit] = new
            total_new += new
        except Exception as e:
            logger.error(f"  ERROR collecting r/{subreddit}: {e}")
            stats[subreddit] = -1

        time.sleep(REQUEST_DELAY)

    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM posts")
    total_posts = c.fetchone()[0]

    logger.info(f"\n{'='*60}")
    logger.info(f"COLLECTION COMPLETE")
    logger.info(f"New posts this run: {total_new}")
    logger.info(f"Total posts in DB:  {total_posts}")
    logger.info(f"{'='*60}")

    conn.close()
    return stats



def export_csvs():
    conn = init_db()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_sql_query("SELECT * FROM posts ORDER BY datetime_utc", conn)

    if df.empty:
        logger.warning("No posts in database to export!")
        conn.close()
        return

    logger.info(f"Exporting {len(df)} posts to CSV...")
    combined_path = EXPORT_DIR / "reddit_all_posts.csv"
    df.to_csv(combined_path, index=False, quoting=csv.QUOTE_ALL)
    logger.info(f"  Combined: {combined_path} ({len(df)} posts)")

    for coin in COINS + ["MARKET"]:
        coin_df = df[df["coin_target"] == coin]
        if not coin_df.empty:
            coin_path = EXPORT_DIR / f"reddit_{coin.lower()}.csv"
            coin_df.to_csv(coin_path, index=False, quoting=csv.QUOTE_ALL)
            logger.info(f"  {coin}: {coin_path} ({len(coin_df)} posts)")

    for sub in SUBREDDIT_COIN_MAP:
        sub_df = df[df["subreddit"] == sub]
        if not sub_df.empty:
            sub_path = EXPORT_DIR / f"reddit_r_{sub.lower()}.csv"
            sub_df.to_csv(sub_path, index=False, quoting=csv.QUOTE_ALL)
            logger.info(f"  r/{sub}: {sub_path} ({len(sub_df)} posts)")

    conn.close()
    logger.info("CSV export complete!")



def detect_date_gaps():
    conn = init_db()
    df = pd.read_sql_query("SELECT date, coin_target, subreddit FROM posts", conn)
    conn.close()

    if df.empty:
        logger.warning("No data for gap detection.")
        return {}

    df["date"] = pd.to_datetime(df["date"])
    date_range = pd.date_range(df["date"].min(), df["date"].max())

    gaps = {}

    logger.info(f"\n{'─'*50}")
    logger.info("DATE GAP DETECTION")
    logger.info(f"{'─'*50}")

    for coin in COINS + ["MARKET"]:
        coin_dates = set(df[df["coin_target"] == coin]["date"].dt.date)
        expected = set(d.date() for d in date_range)
        missing = sorted(expected - coin_dates)
        if missing:
            gaps[coin] = [str(d) for d in missing]
            logger.warning(f"  {coin}: {len(missing)} missing days "
                          f"(first: {missing[0]}, last: {missing[-1]})")
        else:
            logger.info(f"  {coin}: No gaps")

    for sub in SUBREDDIT_COIN_MAP:
        sub_dates = set(df[df["subreddit"] == sub]["date"].dt.date)
        expected = set(d.date() for d in date_range)
        missing = sorted(expected - sub_dates)
        if missing:
            gaps[f"r/{sub}"] = [str(d) for d in missing]
            logger.warning(f"  r/{sub}: {len(missing)} missing days")

    return gaps


def print_status():
    """Print detailed collection statistics."""
    if not DB_PATH.exists():
        print("No database found. Run collection first.")
        return

    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    print(f"\n{'='*60}")
    print("REDDIT COLLECTION STATUS")
    print(f"{'='*60}")

    c.execute("SELECT COUNT(*) FROM posts")
    total = c.fetchone()[0]
    print(f"\nTotal posts: {total:,}")

    c.execute("SELECT MIN(date), MAX(date) FROM posts")
    min_date, max_date = c.fetchone()
    print(f"Date range:  {min_date} -> {max_date}")

    print(f"\n{'─'*40}")
    print(f"{'Coin':<10} {'Posts':>8} {'Date Range':<25}")
    print(f"{'─'*40}")
    for coin in COINS + ["MARKET"]:
        c.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM posts WHERE coin_target = ?", (coin,))
        count, d_min, d_max = c.fetchone()
        if count:
            print(f"{coin:<10} {count:>8,} {d_min} -> {d_max}")

    print(f"\n{'─'*50}")
    print(f"{'Subreddit':<22} {'Posts':>8} {'Last Post':<12}")
    print(f"{'─'*50}")
    for sub in SUBREDDIT_COIN_MAP:
        c.execute("SELECT COUNT(*), MAX(date) FROM posts WHERE subreddit = ?", (sub,))
        count, last = c.fetchone()
        print(f"r/{sub:<20} {count:>8,} {last or 'N/A':<12}")

    print(f"\n{'─'*30}")
    print("Post types:")
    c.execute("SELECT post_type, COUNT(*) FROM posts GROUP BY post_type ORDER BY COUNT(*) DESC")
    for ptype, count in c.fetchall():
        print(f"  {ptype:<10} {count:>8,}")

    db_size = DB_PATH.stat().st_size / (1024 * 1024)
    print(f"\nDatabase size: {db_size:.1f} MB")

    conn.close()
    detect_date_gaps()


def run_scheduled(interval_hours: int, days: int = DEFAULT_DAYS):
    """Run collection on a schedule."""
    logger.info(f"Scheduler started: collecting every {interval_hours} hours")
    while True:
        try:
            run_collection(days=days)
            export_csvs()
            detect_date_gaps()
        except Exception as e:
            logger.error(f"Scheduled run failed: {e}")

        logger.info(f"Next run in {interval_hours} hours...")
        time.sleep(interval_hours * 3600)


def main():
    parser = argparse.ArgumentParser(
        description="Reddit Crypto Data Collector (Arctic Shift API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reddit_collector.py                  # 180 days backfill
  python reddit_collector.py --days 30        # Last 30 days
  python reddit_collector.py --schedule 6     # Run every 6 hours
  python reddit_collector.py --export-only    # Export CSVs only
  python reddit_collector.py --status         # Show stats
        """,
    )
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"Days to collect (default: {DEFAULT_DAYS})")
    parser.add_argument("--schedule", type=int, default=None,
                        help="Run every N hours (continuous mode)")
    parser.add_argument("--export-only", action="store_true",
                        help="Skip collection, just export CSVs")
    parser.add_argument("--status", action="store_true",
                        help="Show collection statistics")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore checkpoints, re-collect everything")

    args = parser.parse_args()

    if args.status:
        print_status()
        return

    if args.export_only:
        export_csvs()
        detect_date_gaps()
        return

    if args.schedule:
        run_scheduled(args.schedule, days=args.days)
    else:
        run_collection(days=args.days)
        export_csvs()
        detect_date_gaps()


if __name__ == "__main__":
    main()