import argparse
import csv
import json
import logging
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
LOG_DIR = Path("logs")
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

ARCTIC_SHIFT_URL = "https://arctic-shift.photon-reddit.com/api/posts/search"
PULLPUSH_URL = "https://api.pullpush.io/reddit/search/submission/"

REQUEST_DELAY = 1.5
MAX_RETRIES = 5
BATCH_SIZE = 100
DEFAULT_DAYS = 180
CHUNK_DAYS = 7
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
    conn.commit()
    return conn


def clean_text(title, selftext):
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

def detect_tickers(text):
    return [t for t, p in TICKER_PATTERNS.items() if re.search(p, text, re.IGNORECASE)]

def determine_post_type(post):
    if post.get("is_video"): return "video"
    url = post.get("url", "")
    if any(ext in url for ext in [".jpg", ".png", ".gif", ".jpeg", ".webp"]): return "image"
    domain = post.get("domain", "")
    if domain and "self." not in domain and "reddit.com" not in domain: return "link"
    return "text"

def process_post(post, subreddit):
    post_id = post.get("id")
    if not post_id: return None
    text_clean = clean_text(post.get("title", ""), post.get("selftext", ""))
    word_count = len(text_clean.split())
    if word_count == 0: return None
    created_utc = post.get("created_utc", 0)
    if isinstance(created_utc, str):
        try: created_utc = int(float(created_utc))
        except: return None
    dt = datetime.fromtimestamp(created_utc, tz=timezone.utc)
    return {
        "id": post_id, "subreddit": subreddit,
        "coin_target": SUBREDDIT_COIN_MAP.get(subreddit, "MARKET"),
        "title": (post.get("title", "") or "").strip(),
        "text_clean": text_clean, "word_count": word_count,
        "flair": post.get("link_flair_text", "") or "",
        "datetime_utc": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "date": dt.strftime("%Y-%m-%d"), "time": dt.strftime("%H:%M:%S"),
        "upvotes": post.get("score", 0) or 0,
        "upvote_ratio": post.get("upvote_ratio", 0.0) or 0.0,
        "num_comments": post.get("num_comments", 0) or 0,
        "tickers": ",".join(detect_tickers(text_clean)),
        "post_type": determine_post_type(post),
    }

def fetch_arctic_shift(subreddit, after, before):
    all_posts = []
    current_after = after

    while True:
        params = {"subreddit": subreddit, "after": current_after, "before": before,
                  "sort": "asc", "limit": BATCH_SIZE}

        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(ARCTIC_SHIFT_URL, params=params, timeout=30,
                                    headers={"User-Agent": "CryptoSentimentResearch/1.0"})
                if resp.status_code == 429:
                    wait = min(2 ** attempt * 5, 60)
                    logger.warning(f"Arctic Shift rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if resp.status_code == 400:
                    logger.error(f"Arctic Shift bad request: {resp.text[:200]}")
                    return all_posts
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.exceptions.RequestException as e:
                wait = min(2 ** attempt * 2, 30)
                logger.warning(f"Arctic Shift error (attempt {attempt+1}): {e}")
                time.sleep(wait)
                if attempt == MAX_RETRIES - 1:
                    return all_posts
                continue

        posts = data.get("data", [])
        if not posts:
            break
        all_posts.extend(posts)

        last_ts = posts[-1].get("created_utc", 0)
        if isinstance(last_ts, str): last_ts = int(float(last_ts))
        if last_ts <= current_after: break
        current_after = last_ts
        if len(posts) < BATCH_SIZE: break
        time.sleep(REQUEST_DELAY)

    return all_posts


def fetch_pullpush(subreddit, after, before):
    all_posts = []
    current_after = after

    while True:
        params = {"subreddit": subreddit, "after": current_after, "before": before,
                  "sort": "asc", "sort_type": "created_utc", "size": BATCH_SIZE}

        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(PULLPUSH_URL, params=params, timeout=30,
                                    headers={"User-Agent": "CryptoSentimentResearch/1.0"})
                if resp.status_code == 429:
                    wait = min(2 ** attempt * 10, 120)
                    logger.warning(f"PullPush rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.exceptions.RequestException as e:
                wait = min(2 ** attempt * 4, 60)
                logger.warning(f"PullPush error (attempt {attempt+1}): {e}")
                time.sleep(wait)
                if attempt == MAX_RETRIES - 1:
                    return all_posts
                continue

        posts = data.get("data", [])
        if not posts:
            break
        all_posts.extend(posts)

        last_ts = posts[-1].get("created_utc", 0)
        if isinstance(last_ts, str): last_ts = int(float(last_ts))
        if last_ts <= current_after: break
        current_after = last_ts
        if len(posts) < BATCH_SIZE: break
        time.sleep(max(REQUEST_DELAY, 4.0))

    return all_posts


def store_posts(conn, posts):
    if not posts: return 0
    c = conn.cursor()
    inserted = 0
    for p in posts:
        try:
            c.execute("""INSERT OR IGNORE INTO posts
                (id, subreddit, coin_target, title, text_clean, word_count,
                 flair, datetime_utc, date, time, upvotes, upvote_ratio,
                 num_comments, tickers, post_type)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (p["id"], p["subreddit"], p["coin_target"], p["title"],
                 p["text_clean"], p["word_count"], p["flair"],
                 p["datetime_utc"], p["date"], p["time"],
                 p["upvotes"], p["upvote_ratio"], p["num_comments"],
                 p["tickers"], p["post_type"]))
            if c.rowcount > 0: inserted += 1
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    return inserted

def update_checkpoint(conn, subreddit, last_ts, total):
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO checkpoints VALUES (?,?,?,?)",
              (subreddit, last_ts, total, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()

def get_checkpoint(conn, subreddit):
    c = conn.cursor()
    c.execute("SELECT last_timestamp FROM checkpoints WHERE subreddit = ?", (subreddit,))
    row = c.fetchone()
    return row[0] if row else None


def collect_subreddit(conn, subreddit, start_ts, end_ts,
                      source="auto", resume=True):
    if resume:
        ckpt = get_checkpoint(conn, subreddit)
        if ckpt and ckpt > start_ts:
            logger.info(f"  Resuming r/{subreddit} from {datetime.fromtimestamp(ckpt, tz=timezone.utc).strftime('%Y-%m-%d')}")
            start_ts = ckpt

    start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)
    total_days = (end_dt - start_dt).days

    logger.info(f"  Collecting r/{subreddit}: {start_dt.strftime('%Y-%m-%d')} → {end_dt.strftime('%Y-%m-%d')} ({total_days} days)")

    total_inserted = 0
    chunk_start = start_dt

    while chunk_start < end_dt:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), end_dt)
        chunk_start_ts = int(chunk_start.timestamp())
        chunk_end_ts = int(chunk_end.timestamp())
        raw_posts = []

        if source in ("auto", "arctic"):
            raw_posts = fetch_arctic_shift(subreddit, chunk_start_ts, chunk_end_ts)

        if not raw_posts and source in ("auto", "pullpush"):
            if source == "auto":
                logger.info(f"    Arctic Shift empty for {chunk_start.strftime('%Y-%m-%d')}–{chunk_end.strftime('%Y-%m-%d')}, trying PullPush...")
            raw_posts = fetch_pullpush(subreddit, chunk_start_ts, chunk_end_ts)

        if raw_posts:
            processed = [p for rp in raw_posts if (p := process_post(rp, subreddit))]
            inserted = store_posts(conn, processed)
            total_inserted += inserted
            logger.info(f"    {chunk_start.strftime('%m-%d')}→{chunk_end.strftime('%m-%d')}: "
                       f"{len(raw_posts)} fetched, {inserted} new")

            # Update checkpoint
            last_ts = max(
                (int(float(rp.get("created_utc", 0))) if isinstance(rp.get("created_utc"), str)
                 else rp.get("created_utc", 0)) for rp in raw_posts)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM posts WHERE subreddit = ?", (subreddit,))
            update_checkpoint(conn, subreddit, last_ts, c.fetchone()[0])
        else:
            logger.info(f"    {chunk_start.strftime('%m-%d')}→{chunk_end.strftime('%m-%d')}: no data from any source")

        chunk_start = chunk_end
        time.sleep(REQUEST_DELAY)

    return total_inserted


def run_collection(days=DEFAULT_DAYS, source="auto"):
    conn = init_db()
    now = datetime.now(timezone.utc)
    end_ts = int(now.timestamp())
    start_ts = int((now - timedelta(days=days)).timestamp())

    logger.info(f"{'='*60}")
    logger.info(f"REDDIT DATA COLLECTION — {days} days")
    logger.info(f"Period: {datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime('%Y-%m-%d')} → "
                f"{datetime.fromtimestamp(end_ts, tz=timezone.utc).strftime('%Y-%m-%d')}")
    logger.info(f"Source: {source} | Chunk size: {CHUNK_DAYS} days")
    logger.info(f"Subreddits: {len(SUBREDDIT_COIN_MAP)}")
    logger.info(f"{'='*60}")

    stats = {}
    total_new = 0
    for subreddit in SUBREDDIT_COIN_MAP:
        try:
            new = collect_subreddit(conn, subreddit, start_ts, end_ts, source=source)
            stats[subreddit] = new
            total_new += new
        except Exception as e:
            logger.error(f"  ERROR r/{subreddit}: {e}")
            stats[subreddit] = -1
        time.sleep(REQUEST_DELAY)

    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM posts")
    total = c.fetchone()[0]
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE — {total_new} new posts, {total} total in DB")
    logger.info(f"{'='*60}")
    conn.close()
    return stats


def export_csvs():
    conn = init_db()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_sql_query("SELECT * FROM posts ORDER BY datetime_utc", conn)
    if df.empty:
        logger.warning("No posts to export!")
        conn.close()
        return
    logger.info(f"Exporting {len(df)} posts...")
    df.to_csv(EXPORT_DIR / "reddit_all_posts.csv", index=False, quoting=csv.QUOTE_ALL)
    for coin in COINS + ["MARKET"]:
        cdf = df[df["coin_target"] == coin]
        if not cdf.empty:
            cdf.to_csv(EXPORT_DIR / f"reddit_{coin.lower()}.csv", index=False, quoting=csv.QUOTE_ALL)
    conn.close()
    logger.info("Export complete!")

def detect_date_gaps():
    conn = init_db()
    df = pd.read_sql_query("SELECT date, coin_target FROM posts", conn)
    conn.close()
    if df.empty: return {}
    df["date"] = pd.to_datetime(df["date"])
    dr = pd.date_range(df["date"].min(), df["date"].max())
    gaps = {}
    logger.info(f"\n{'─'*50}\nDATE GAP DETECTION\n{'─'*50}")
    for coin in COINS + ["MARKET"]:
        cd = set(df[df["coin_target"] == coin]["date"].dt.date)
        missing = sorted(set(d.date() for d in dr) - cd)
        if missing:
            gaps[coin] = [str(d) for d in missing]
            logger.warning(f"  {coin}: {len(missing)} missing days")
        else:
            logger.info(f"  {coin}: No gaps")
    return gaps

def print_status():
    if not DB_PATH.exists():
        print("No database found."); return
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    print(f"\n{'='*60}\nREDDIT COLLECTION STATUS\n{'='*60}")
    c.execute("SELECT COUNT(*) FROM posts")
    print(f"\nTotal posts: {c.fetchone()[0]:,}")
    c.execute("SELECT MIN(date), MAX(date) FROM posts")
    mn, mx = c.fetchone()
    print(f"Date range:  {mn} → {mx}")
    print(f"\n{'─'*50}")
    for coin in COINS + ["MARKET"]:
        c.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM posts WHERE coin_target = ?", (coin,))
        cnt, d1, d2 = c.fetchone()
        if cnt: print(f"  {coin:<8} {cnt:>7,} posts  {d1} → {d2}")
    print(f"\n{'─'*50}")
    for sub in SUBREDDIT_COIN_MAP:
        c.execute("SELECT COUNT(*), MAX(date) FROM posts WHERE subreddit = ?", (sub,))
        cnt, last = c.fetchone()
        print(f"  r/{sub:<20} {cnt:>7,}  last: {last or 'N/A'}")
    conn.close()
    detect_date_gaps()


def main():
    parser = argparse.ArgumentParser(description="Reddit Collector (Arctic Shift + PullPush)")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--source", choices=["auto", "arctic", "pullpush"], default="auto",
                        help="Data source: auto tries Arctic Shift then PullPush")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.status: print_status(); return
    if args.export_only: export_csvs(); detect_date_gaps(); return

    run_collection(days=args.days, source=args.source)
    export_csvs()
    detect_date_gaps()

if __name__ == "__main__":
    main()