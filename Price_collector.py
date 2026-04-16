import argparse
import csv
import logging
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

DB_PATH = Path("data") / "project.db"
EXPORT_DIR = Path("data") / "prices"
LOG_DIR = Path("logs")
COIN_SYMBOLS = {
    "BTC":  "BTCUSDT",
    "ETH":  "ETHUSDT",
    "SOL":  "SOLUSDT",
    "DOGE": "DOGEUSDT",
    "SHIB": "SHIBUSDT",
    "BNB":  "BNBUSDT",
}

# Coin → yfinance ticker mapping (fallback)
COIN_YFINANCE = {
    "BTC":  "BTC-USD",
    "ETH":  "ETH-USD",
    "SOL":  "SOL-USD",
    "DOGE": "DOGE-USD",
    "SHIB": "SHIB-USD",
    "BNB":  "BNB-USD",
}

BINANCE_BASE = "https://data-api.binance.vision"
BINANCE_KLINES = f"{BINANCE_BASE}/api/v3/klines"
BINANCE_FALLBACKS = [
    "https://api.binance.com/api/v3/klines",
    "https://api1.binance.com/api/v3/klines",
    "https://api2.binance.com/api/v3/klines",
]

REQUEST_DELAY = 0.5
MAX_RETRIES = 3
DEFAULT_DAYS = 180
KLINE_LIMIT = 1000

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "price_collector.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            coin            TEXT NOT NULL,
            date            TEXT NOT NULL,
            open            REAL NOT NULL,
            high            REAL NOT NULL,
            low             REAL NOT NULL,
            close           REAL NOT NULL,
            volume          REAL NOT NULL,
            quote_volume    REAL DEFAULT 0,
            num_trades      INTEGER DEFAULT 0,
            price_change    REAL DEFAULT 0,
            price_change_pct REAL DEFAULT 0,
            source          TEXT DEFAULT 'binance',
            PRIMARY KEY (coin, date)
        )
    """)

    c.execute("CREATE INDEX IF NOT EXISTS idx_prices_coin ON prices(coin)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_prices_coin_date ON prices(coin, date)")

    conn.commit()
    return conn


def fetch_binance_klines(symbol: str, start_ts_ms: int, end_ts_ms: int) -> list[dict] | None:
    params = {
        "symbol": symbol,
        "interval": "1d",
        "startTime": start_ts_ms,
        "endTime": end_ts_ms,
        "limit": KLINE_LIMIT,
    }

    endpoints = [BINANCE_KLINES] + BINANCE_FALLBACKS

    for endpoint in endpoints:
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(
                    endpoint,
                    params=params,
                    timeout=15,
                    headers={"User-Agent": "CryptoSentimentResearch/1.0"}
                )

                if resp.status_code == 429:
                    wait = min(2 ** attempt * 5, 30)
                    logger.warning(f"Rate limited on {endpoint}. Waiting {wait}s...")
                    time.sleep(wait)
                    continue

                if resp.status_code == 451:
                    logger.warning(f"Geographic restriction on {endpoint}, trying fallback...")
                    break

                if resp.status_code != 200:
                    logger.warning(f"HTTP {resp.status_code} from {endpoint}: {resp.text[:200]}")
                    break

                raw_klines = resp.json()
                if not raw_klines or not isinstance(raw_klines, list):
                    logger.warning(f"Empty/invalid response from {endpoint} for {symbol}")
                    break

                records = []
                for k in raw_klines:
                    open_time_ms = k[0]
                    dt = datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc)
                    date_str = dt.strftime("%Y-%m-%d")

                    open_p = float(k[1])
                    high_p = float(k[2])
                    low_p = float(k[3])
                    close_p = float(k[4])
                    volume = float(k[5])
                    quote_vol = float(k[7])
                    num_trades = int(k[8])

                    price_change = close_p - open_p
                    price_change_pct = (price_change / open_p * 100) if open_p != 0 else 0

                    records.append({
                        "date":             date_str,
                        "open":             open_p,
                        "high":             high_p,
                        "low":              low_p,
                        "close":            close_p,
                        "volume":           volume,
                        "quote_volume":     quote_vol,
                        "num_trades":       num_trades,
                        "price_change":     round(price_change, 8),
                        "price_change_pct": round(price_change_pct, 4),
                        "source":           "binance",
                    })

                logger.info(f"  Binance: fetched {len(records)} daily candles for {symbol}")
                return records

            except requests.exceptions.RequestException as e:
                wait = min(2 ** attempt * 2, 15)
                logger.warning(f"Request error ({endpoint}, attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
                continue

    return None


def fetch_yfinance_klines(ticker: str, start_date: str, end_date: str) -> list[dict] | None:
    """Fetch daily OHLCV from Yahoo Finance as a fallback."""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Install with: pip install yfinance")
        return None

    try:
        tk = yf.Ticker(ticker)
        df = tk.history(start=start_date, end=end_date, interval="1d")

        if df.empty:
            logger.warning(f"yfinance returned no data for {ticker}")
            return None

        records = []
        for idx, row in df.iterrows():
            date_str = idx.strftime("%Y-%m-%d")
            open_p = float(row["Open"])
            close_p = float(row["Close"])
            price_change = close_p - open_p
            price_change_pct = (price_change / open_p * 100) if open_p != 0 else 0

            records.append({
                "date":             date_str,
                "open":             open_p,
                "high":             float(row["High"]),
                "low":              float(row["Low"]),
                "close":            close_p,
                "volume":           float(row["Volume"]),
                "quote_volume":     0.0,
                "num_trades":       0,
                "price_change":     round(price_change, 8),
                "price_change_pct": round(price_change_pct, 4),
                "source":           "yfinance",
            })

        logger.info(f"  yfinance: fetched {len(records)} daily candles for {ticker}")
        return records

    except Exception as e:
        logger.error(f"yfinance error for {ticker}: {e}")
        return None


def store_prices(conn: sqlite3.Connection, coin: str, records: list[dict]) -> int:
    """Insert price records with upsert on conflict."""
    if not records:
        return 0

    c = conn.cursor()
    stored = 0
    for r in records:
        c.execute("""
            INSERT INTO prices
                (coin, date, open, high, low, close, volume,
                 quote_volume, num_trades, price_change, price_change_pct, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(coin, date) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume,
                quote_volume = excluded.quote_volume,
                num_trades = excluded.num_trades,
                price_change = excluded.price_change,
                price_change_pct = excluded.price_change_pct,
                source = excluded.source
        """, (
            coin, r["date"], r["open"], r["high"], r["low"], r["close"],
            r["volume"], r["quote_volume"], r["num_trades"],
            r["price_change"], r["price_change_pct"], r["source"],
        ))
        stored += 1

    conn.commit()
    return stored

def collect_coin(conn: sqlite3.Connection, coin: str,
                 start_ts_ms: int, end_ts_ms: int,
                 start_date: str, end_date: str,
                 force_source: str = None) -> int:
    """Collect price data for a single coin. Binance first, yfinance fallback."""
    records = None

    if force_source != "yfinance":
        symbol = COIN_SYMBOLS.get(coin)
        if symbol:
            records = fetch_binance_klines(symbol, start_ts_ms, end_ts_ms)

    if records is None:
        ticker = COIN_YFINANCE.get(coin)
        if ticker:
            logger.info(f"  Falling back to yfinance for {coin}...")
            records = fetch_yfinance_klines(ticker, start_date, end_date)

    if records is None:
        logger.error(f"  FAILED to collect price data for {coin} from any source")
        return 0

    stored = store_prices(conn, coin, records)
    logger.info(f"  {coin}: stored {stored} daily records")
    return stored


def run_collection(days: int = DEFAULT_DAYS, force_source: str = None) -> dict:
    """Run price collection for all coins."""
    conn = init_db()

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)

    start_ts_ms = int(start.timestamp() * 1000)
    end_ts_ms = int(now.timestamp() * 1000)

    start_date = start.strftime("%Y-%m-%d")
    end_date = now.strftime("%Y-%m-%d")

    logger.info(f"{'='*60}")
    logger.info(f"PRICE DATA COLLECTION — {days} days")
    logger.info(f"Period: {start_date} -> {end_date}")
    logger.info(f"Coins: {', '.join(COIN_SYMBOLS.keys())}")
    if force_source:
        logger.info(f"Forced source: {force_source}")
    logger.info(f"{'='*60}")

    stats = {}
    for coin in COIN_SYMBOLS:
        try:
            stored = collect_coin(
                conn, coin,
                start_ts_ms, end_ts_ms,
                start_date, end_date,
                force_source=force_source,
            )
            stats[coin] = stored
        except Exception as e:
            logger.error(f"  ERROR collecting {coin}: {e}")
            stats[coin] = -1

        time.sleep(REQUEST_DELAY)

    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM prices")
    total = c.fetchone()[0]

    logger.info(f"\n{'='*60}")
    logger.info(f"PRICE COLLECTION COMPLETE")
    logger.info(f"Total records in DB: {total}")
    logger.info(f"{'='*60}")

    conn.close()
    return stats


def export_csvs():
    """Export per-coin price CSVs."""
    conn = init_db()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_sql_query("SELECT * FROM prices ORDER BY coin, date", conn)

    if df.empty:
        logger.warning("No price data to export!")
        conn.close()
        return

    logger.info(f"Exporting {len(df)} price records to CSV...")

    for coin in COIN_SYMBOLS:
        coin_df = df[df["coin"] == coin].copy()
        if not coin_df.empty:
            coin_path = EXPORT_DIR / f"{coin.lower()}_daily.csv"
            coin_df.to_csv(coin_path, index=False, quoting=csv.QUOTE_MINIMAL)
            logger.info(f"  {coin}: {coin_path} ({len(coin_df)} days)")

    combined_path = EXPORT_DIR / "all_prices_daily.csv"
    df.to_csv(combined_path, index=False, quoting=csv.QUOTE_MINIMAL)
    logger.info(f"  Combined: {combined_path} ({len(df)} records)")

    conn.close()
    logger.info("Price CSV export complete!")


def print_status():
    """Print price data statistics."""
    if not DB_PATH.exists():
        print("No database found. Run collection first.")
        return

    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='prices'")
    if not c.fetchone():
        print("No price data collected yet.")
        conn.close()
        return

    print(f"\n{'='*60}")
    print("PRICE DATA STATUS")
    print(f"{'='*60}")

    c.execute("SELECT COUNT(*) FROM prices")
    total = c.fetchone()[0]
    print(f"\nTotal records: {total:,}")

    c.execute("SELECT MIN(date), MAX(date) FROM prices")
    min_date, max_date = c.fetchone()
    print(f"Date range:    {min_date} -> {max_date}")

    print(f"\n{'─'*65}")
    print(f"{'Coin':<6} {'Days':>5} {'Date Range':<25} {'Latest Close':>14} {'Source':<10}")
    print(f"{'─'*65}")

    for coin in COIN_SYMBOLS:
        c.execute("""
            SELECT COUNT(*), MIN(date), MAX(date),
                   (SELECT close FROM prices WHERE coin = ? ORDER BY date DESC LIMIT 1),
                   (SELECT source FROM prices WHERE coin = ? ORDER BY date DESC LIMIT 1)
            FROM prices WHERE coin = ?
        """, (coin, coin, coin))
        count, d_min, d_max, last_close, source = c.fetchone()
        if count:
            if last_close and last_close < 0.01:
                price_str = f"${last_close:.8f}"
            elif last_close and last_close < 1:
                price_str = f"${last_close:.4f}"
            else:
                price_str = f"${last_close:,.2f}" if last_close else "N/A"
            print(f"{coin:<6} {count:>5} {d_min} -> {d_max} {price_str:>14} {source or 'N/A':<10}")

    # Gap detection
    print(f"\n{'─'*40}")
    print("DATE GAPS:")
    df = pd.read_sql_query("SELECT coin, date FROM prices ORDER BY coin, date", conn)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        for coin in COIN_SYMBOLS:
            coin_dates = df[df["coin"] == coin]["date"]
            if len(coin_dates) < 2:
                continue
            date_range = pd.date_range(coin_dates.min(), coin_dates.max())
            missing = sorted(set(date_range) - set(coin_dates))
            if missing:
                print(f"  {coin}: {len(missing)} missing days")
            else:
                print(f"  {coin}: No gaps")

    db_size = DB_PATH.stat().st_size / (1024 * 1024)
    print(f"\nDatabase size: {db_size:.1f} MB")
    conn.close()

def main():
    parser = argparse.ArgumentParser(
        description="Crypto Price Data Collector (Binance + yfinance fallback)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python price_collector.py                     # 180 days from Binance
  python price_collector.py --days 30           # Last 30 days
  python price_collector.py --source yfinance   # Force yfinance
  python price_collector.py --export-only       # Export CSVs only
  python price_collector.py --status            # Show stats
        """,
    )
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"Days to collect (default: {DEFAULT_DAYS})")
    parser.add_argument("--source", choices=["binance", "yfinance"], default=None,
                        help="Force a specific data source")
    parser.add_argument("--export-only", action="store_true",
                        help="Skip collection, just export CSVs")
    parser.add_argument("--status", action="store_true",
                        help="Show price data statistics")

    args = parser.parse_args()

    if args.status:
        print_status()
        return

    if args.export_only:
        export_csvs()
        return

    run_collection(days=args.days, force_source=args.source)
    export_csvs()


if __name__ == "__main__":
    main()